#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE lammps-hdnnp
#include "Example_lammps_hdnnp.h"
#include "nnp_test.h"
#include "tolerance.h"
#include "utility.h"

#include <fstream> // std::ifstream
#include <limits>  // std::numeric_limits
#include <string>  // std::string
#include <vector>  // std::vector

using namespace std;
using namespace nnp;

Tolerance const tolerance =
    Tolerance(10.0 * numeric_limits<double>::epsilon());

BoostDataContainer<Example_lammps_hdnnp> container;

// Forces of the unperturbed structure ("forces.dump", columns id type fx fy fz
// sorted by id, one time step) against central differences of the potential
// energies in the thermo rows of step 0 (unperturbed, then +h and -h for each
// atom in fdAtoms along x, y and z).
void finiteDifferenceCheck(Example_lammps_hdnnp const& example)
{
    vector<double> energies;
    string line;
    ifstream file;
    file.open("log.lammps");
    BOOST_REQUIRE(file.is_open());
    bool header = false;
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));
        if (columns.size() == 2 && columns.at(0) == "Step")
        {
            header = true;
            continue;
        }
        if (header && columns.size() == 2 && columns.at(0) == "0")
        {
            energies.push_back(stod(columns.at(1)));
        }
        header = false;
    }
    file.close();
    BOOST_REQUIRE_EQUAL(energies.size(), 1 + 6 * example.fdAtoms.size());

    vector<vector<double>> forces;
    file.open("forces.dump");
    BOOST_REQUIRE(file.is_open());
    bool atoms = false;
    while (getline(file, line))
    {
        if (line.find("ITEM: ATOMS") == 0)
        {
            atoms = true;
            continue;
        }
        vector<string> columns = split(reduce(line));
        if (!atoms || columns.size() != 5) continue;
        forces.push_back({stod(columns.at(2)),
                          stod(columns.at(3)),
                          stod(columns.at(4))});
    }
    file.close();

    Tolerance const fdTolerance(example.fdTolerance);
    for (size_t n = 0; n < example.fdAtoms.size(); ++n)
    {
        size_t const a = example.fdAtoms.at(n);
        BOOST_REQUIRE(a >= 1 && a <= forces.size());
        for (size_t k = 0; k < 3; ++k)
        {
            size_t const e = 1 + 6 * n + 2 * k;
            double const fd = -(energies.at(e) - energies.at(e + 1))
                            / (2.0 * example.fdStep);
            BOOST_TEST_INFO("Atom " << a << " force component " << k);
            REQUIRE_CLOSE_TO_REFERENCE(forces.at(a - 1).at(k), fd, fdTolerance);
        }
    }

    return;
}

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_lammps_hdnnp const example)
{
    if (!example.fdAtoms.empty())
    {
        finiteDifferenceCheck(example);
        return;
    }

    bool timeStepFound = false;
    bool startReading = false;
    string line;
    ifstream file;
    file.open("log.lammps");
    BOOST_REQUIRE(file.is_open());
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));
        if (columns.size() > 0 && columns.at(0) == "Step")
        {
            startReading = true;
        }
        if (startReading == true &&
            columns.size() >= 5 &&
            columns.at(0) == to_string(example.lastTimeStep))
        {
            timeStepFound = true;
            REQUIRE_CLOSE_TO_REFERENCE(example.potentialEnergy,
                                       stod(columns.at(2)),
                                       tolerance);
            REQUIRE_CLOSE_TO_REFERENCE(example.totalEnergy,
                                       stod(columns.at(4)),
                                       tolerance);
            break;
        }
    }
    BOOST_REQUIRE_MESSAGE(timeStepFound,
                          string("ERROR: Epoch information was not "
                                 "found in file."));

    if (example.forcesReference.empty()) return;

    // Forces of the last time step written by the example (dump with columns
    // id type fx fy fz, sorted by id) against the reference (id fx fy fz).
    vector<vector<double>> forces;
    vector<vector<double>> reference;
    file.close();
    file.open("forces.dump");
    BOOST_REQUIRE(file.is_open());
    bool atoms = false;
    while (getline(file, line))
    {
        if (line.find("ITEM: TIMESTEP") == 0)
        {
            forces.clear();
            atoms = false;
            continue;
        }
        if (line.find("ITEM: ATOMS") == 0)
        {
            atoms = true;
            continue;
        }
        vector<string> columns = split(reduce(line));
        if (!atoms || columns.size() != 5) continue;
        forces.push_back({stod(columns.at(2)),
                          stod(columns.at(3)),
                          stod(columns.at(4))});
    }
    file.close();
    file.open(example.forcesReference);
    BOOST_REQUIRE(file.is_open());
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));
        if (columns.size() != 4 || columns.at(0).at(0) == '#') continue;
        reference.push_back({stod(columns.at(1)),
                             stod(columns.at(2)),
                             stod(columns.at(3))});
    }
    BOOST_REQUIRE_EQUAL(forces.size(), reference.size());
    Tolerance const forcesTolerance(example.forcesTolerance);
    for (size_t i = 0; i < forces.size(); ++i)
    {
        for (size_t k = 0; k < 3; ++k)
        {
            BOOST_TEST_INFO("Atom " << i + 1 << " force component " << k);
            REQUIRE_CLOSE_TO_REFERENCE(reference.at(i).at(k),
                                       forces.at(i).at(k),
                                       forcesTolerance);
        }
    }

    return;
}
