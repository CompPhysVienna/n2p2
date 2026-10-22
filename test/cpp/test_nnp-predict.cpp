#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-predict
#include "Example_nnp_predict.h"
#include "ExamplePrediction.h"
#include "nnp_test.h"
#include "tolerance.h"
#include "utility.h"

#include <cstddef> // std::size_t
#include <limits>  // std::numeric_limits

using namespace std;
using namespace nnp;

//double const accuracy = 10.0 * numeric_limits<double>::epsilon();
double const accuracy = 1E-5; // Output

BoostDataContainer<Example_nnp_predict> container;

// Full reference forces of the same examples, already used by
// test_Prediction (library) and checked here for the application output.
BoostDataContainer<ExamplePrediction> referenceForces;

// output.data prints forces and charges with 17 significant digits.
Tolerance const toleranceForces =
    Tolerance(1000.0 * numeric_limits<double>::epsilon());
// The 4G checks were recorded with a later build than the energies above
// (which differ from it by 1E-11 relative): allow a little more.
Tolerance const toleranceCheckAtoms = Tolerance(1.0E-10);

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_predict const example)
{
    BOOST_REQUIRE(bfs::exists("nnp-predict.log"));
    for (auto f : example.createdFiles)
    {
        BOOST_REQUIRE(bfs::exists(f));
    }
    ifstream file;
    file.open("output.data");
    string line;
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));
        if (columns.at(0) != "energy") continue;
        else
        {
            BOOST_REQUIRE_SMALL(example.energy - stod(columns.at(1)),
                                accuracy);
            break;
        }
    }
    file.close();

    // Forces (columns 7 to 9) and charges (column 5) of all atoms.
    vector<Vec3D> forces;
    vector<double> charges;
    file.open("output.data");
    while (getline(file, line))
    {
        vector<string> columns = split(reduce(line));
        if (columns.at(0) != "atom") continue;
        charges.push_back(stod(columns.at(5)));
        forces.push_back(Vec3D(stod(columns.at(7)),
                               stod(columns.at(8)),
                               stod(columns.at(9))));
    }
    file.close();

    for (auto const& ref : referenceForces.examples)
    {
        if (ref.name != example.name) continue;
        BOOST_REQUIRE_EQUAL(forces.size(), ref.forces.size());
        for (size_t i = 0; i < forces.size(); ++i)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                BOOST_TEST_INFO(example.name << " atom " << i
                                << " force component " << c);
                REQUIRE_CLOSE_TO_REFERENCE(ref.forces.at(i)[c],
                                           forces.at(i)[c],
                                           toleranceForces);
            }
        }
    }
    for (size_t k = 0; k < example.checkAtoms.size(); ++k)
    {
        size_t const i = example.checkAtoms.at(k) - 1;
        BOOST_REQUIRE(i < forces.size());
        for (size_t c = 0; c < 3; ++c)
        {
            BOOST_TEST_INFO(example.name << " atom " << i + 1
                            << " force component " << c);
            REQUIRE_CLOSE_TO_REFERENCE(example.checkForces.at(k)[c],
                                       forces.at(i)[c],
                                       toleranceCheckAtoms);
        }
        BOOST_TEST_INFO(example.name << " atom " << i + 1 << " charge");
        REQUIRE_CLOSE_TO_REFERENCE(example.checkCharges.at(k),
                                   charges.at(i),
                                   toleranceCheckAtoms);
    }

    return;
}
