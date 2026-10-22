#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Supercell
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "Prediction.h"
#include "utility.h"
#include <algorithm> // std::max
#include <cmath>     // std::fabs
#include <cstddef>   // std::size_t
#include <fstream>   // std::ifstream, std::ofstream
#include <map>       // std::map
#include <string>    // std::string
#include <utility>   // std::pair
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

// Doubling a periodic cell along its first lattice vector must double the
// total energy and leave every atom's force (and, for 4G-HDNNPs, charge)
// unchanged. This needs no stored reference values and checks periodic
// boundary handling end to end: neighbor lists across cell boundaries
// (including a triclinic cell, Cu2S_PBE), and for 4G the Ewald real- and
// reciprocal-space sums and the charge equilibration with the total charge
// constraint. It also catches any state cached per structure index that is
// not invalidated when a different system is evaluated.

// The Ewald truncation parameters depend on the system, so at the examples'
// ewald_prec 1.e-5 cell and supercell differ at that level; tighten it.
map<string, pair<string, string>> const ewaldPrecisionOverrides = {
    {"AlAuMgO_4G", {"ewald_prec 1.e-5", "ewald_prec 1.e-10"}},
    {"H2O_RPBE-D3_4G", {"ewald_prec 1.e-5", "ewald_prec 1.e-8"}}};

vector<string> const examples = {"AlAuMgO_4G",
                                 "Cu2S_PBE",
                                 "H2O_RPBE-D3",
                                 "H2O_RPBE-D3_4G"};

double const toleranceEnergyRelative = 1.0E-9;
double const toleranceAbsolute = 1.0E-8;
double const toleranceRelative = 1.0E-7;
// Charges come out of the charge equilibration solve, which amplifies the
// remaining Ewald truncation difference between cell and supercell
// (AlAuMgO_4G: ~1E-8 at ewald_prec 1.e-10).
double const toleranceChargeAbsolute = 1.0E-7;

// Writes a copy of the single structure in fileIn, repeated twice along its
// first lattice vector, to fileOut. Reference energy and total charge are
// doubled as well (the total charge is the 4G charge equilibration
// constraint).
void writeDoubledCell(string const& fileIn, string const& fileOut)
{
    ifstream in(fileIn);
    vector<vector<string>> lattice;
    vector<vector<string>> atoms;
    double energy = 0.0;
    double charge = 0.0;
    string line;
    while (getline(in, line))
    {
        vector<string> c = split(reduce(line));
        if (c.empty()) continue;
        if (c.at(0) == "lattice") lattice.push_back(c);
        else if (c.at(0) == "atom") atoms.push_back(c);
        else if (c.at(0) == "energy") energy = stod(c.at(1));
        else if (c.at(0) == "charge") charge = stod(c.at(1));
    }
    in.close();
    BOOST_REQUIRE_EQUAL(lattice.size(), 3);

    ofstream out(fileOut);
    out << "begin\n";
    for (size_t i = 0; i < 3; ++i)
    {
        double const f = (i == 0) ? 2.0 : 1.0;
        out << strpr("lattice %24.16E %24.16E %24.16E\n",
                     f * stod(lattice.at(i).at(1)),
                     f * stod(lattice.at(i).at(2)),
                     f * stod(lattice.at(i).at(3)));
    }
    for (size_t copy = 0; copy < 2; ++copy)
    {
        for (auto const& a : atoms)
        {
            out << strpr("atom %24.16E %24.16E %24.16E %s",
                         stod(a.at(1)) + copy * stod(lattice.at(0).at(1)),
                         stod(a.at(2)) + copy * stod(lattice.at(0).at(2)),
                         stod(a.at(3)) + copy * stod(lattice.at(0).at(3)),
                         a.at(4).c_str());
            for (size_t k = 5; k < a.size(); ++k) out << " " << a.at(k);
            out << "\n";
        }
    }
    out << strpr("energy %24.16E\n", 2.0 * energy);
    out << strpr("charge %24.16E\n", 2.0 * charge);
    out << "end\n";
    out.close();
}

BOOST_AUTO_TEST_SUITE(RegressionTests)

BOOST_DATA_TEST_CASE_F(FixtureRepairDir,
                       DoubledCell_EnergyDoublesForcesAndChargesRepeat,
                       bdata::make(examples),
                       name)
{
    BOOST_REQUIRE(copy_directory_recursively("../../examples/nnp-predict/"
                                             + name, "test"));
    bfs::current_path("test");
    if (ewaldPrecisionOverrides.count(name) > 0)
    {
        auto const& o = ewaldPrecisionOverrides.at(name);
        BOOST_REQUIRE(replaceInFile("input.nn", o.first, o.second));
    }
    writeDoubledCell("input.data", "doubled.data");

    Prediction cell;
    cell.log.writeToStdout = false;
    cell.setup();
    cell.readStructureFromFile("input.data");
    cell.predict();
    Structure const& s1 = cell.structure;

    Prediction doubled;
    doubled.log.writeToStdout = false;
    doubled.setup();
    doubled.readStructureFromFile("doubled.data");
    doubled.predict();
    Structure const& s2 = doubled.structure;

    size_t const n = s1.atoms.size();
    BOOST_REQUIRE_EQUAL(s2.atoms.size(), 2 * n);

    double const e1 = s1.energy;
    double const e2 = s2.energy;
    BOOST_TEST_MESSAGE(name << ": E(cell) = " << e1 << ", E(doubled) / 2 = "
                       << e2 / 2.0 << ", relative deviation "
                       << fabs(e2 - 2.0 * e1) / fabs(2.0 * e1));
    BOOST_REQUIRE_LE(fabs(e2 - 2.0 * e1),
                     toleranceEnergyRelative * fabs(2.0 * e1));

    bool const charges = cell.getNnpType() == Mode::NNPType::HDNNP_4G;
    double maxForce = 0.0;
    double maxCharge = 0.0;
    for (size_t copy = 0; copy < 2; ++copy)
    {
        for (size_t i = 0; i < n; ++i)
        {
            Atom const& a = s1.atoms.at(i);
            Atom const& b = s2.atoms.at(copy * n + i);
            for (size_t c = 0; c < 3; ++c)
            {
                double const d = fabs(b.f[c] - a.f[c]);
                maxForce = max(maxForce, d);
                BOOST_TEST_INFO(name << " copy " << copy << " atom " << i
                                << " force component " << c);
                BOOST_REQUIRE_LE(d, toleranceAbsolute
                                    + toleranceRelative * fabs(a.f[c]));
            }
            if (charges)
            {
                double const d = fabs(b.charge - a.charge);
                maxCharge = max(maxCharge, d);
                BOOST_TEST_INFO(name << " copy " << copy << " atom " << i
                                << " charge");
                BOOST_REQUIRE_LE(d, toleranceChargeAbsolute
                                    + toleranceRelative * fabs(a.charge));
            }
        }
    }
    BOOST_TEST_MESSAGE(name << ": max force deviation " << maxForce
                       << ", max charge deviation " << maxCharge);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()
