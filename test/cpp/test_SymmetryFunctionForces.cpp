#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SymmetryFunctionForces
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "Mode.h"
#include "Structure.h"
#include <algorithm> // std::max
#include <cmath>     // std::fabs, std::sin
#include <cstddef>   // std::size_t
#include <fstream>   // std::ofstream
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bfs = boost::filesystem;

// Symmetry function derivatives of every type, through the production code
// path (Mode::calculateSymmetryFunctionGroups() and Mode::calculateForces()),
// on a periodic, triclinic cell: no neural network is involved. Every
// symmetry function of every atom gets a fixed, arbitrary weight c (stored
// as dEdG), so calculateForces() must return F = -d/dr sum_i sum_k c_ik G_ik,
// which is checked against a 4-point finite difference of that sum.
//
// The upstream analytic-vs-numeric tests (test_SymmetryFunctionGroups.cpp)
// use isolated 3-4 atom clusters. Here the cutoff (8 A) exceeds half the
// shortest cell vector of Cu2S_PBE (11.9 A), so periodic images of the same
// atom appear as separate neighbors, and several functions of the same type
// share a group, as in production setups.

string const settingsFile =
    "number_of_elements 2\n"
    "elements Cu S\n"
    "cutoff_type 6 0.0\n"
    "global_hidden_layers_short 1\n"
    "global_nodes_short 5\n"
    "global_activation_short t l\n"
    "symfunction_short Cu 2 S 0.03 1.0 8.0\n"
    "symfunction_short Cu 2 S 0.10 2.0 8.0\n"
    "symfunction_short S 2 Cu 0.03 1.0 8.0\n"
    "symfunction_short Cu 3 S S 0.03 -1.0 4.0 8.0 0.5\n"
    "symfunction_short S 3 Cu Cu 0.01 1.0 2.0 8.0 0.0\n"
    "symfunction_short S 3 Cu Cu 0.05 -1.0 1.0 8.0 0.0\n"
    "symfunction_short Cu 9 Cu S 0.01 1.0 2.0 8.0 0.0\n"
    "symfunction_short S 12 0.02 0.5 8.0\n"
    "symfunction_short Cu 13 0.01 0.5 1.0 2.0 8.0\n"
    "symfunction_short Cu 20 S 0.0 8.0 p2\n"
    "symfunction_short S 21 Cu Cu 0.0 8.0 30.0 150.0 p2\n"
    "symfunction_short Cu 22 S Cu 1.0 8.0 0.0 180.0 p3\n"
    "symfunction_short S 23 0.0 8.0 p2\n"
    "symfunction_short Cu 24 0.0 8.0 20.0 160.0 p2\n"
    "symfunction_short S 25 1.0 8.0 0.0 180.0 p3\n";

double const h = 1.0E-4;
double const toleranceAbsolute = 1.0E-7;
double const toleranceRelative = 1.0E-6;

// Fixed, irregular weight for symmetry function k of atom i.
double weight(size_t i, size_t k)
{
    return sin(1.3 * i + 0.7 * k + 0.2);
}

double weightedSum(Mode& mode, Structure const& base, size_t j, size_t c,
                   double dx)
{
    Structure s = base;
    s.atoms.at(j).r[c] += dx;
    s.calculateNeighborList(mode.getMaxCutoffRadius());
    mode.calculateSymmetryFunctionGroups(s, false);
    double sum = 0.0;
    for (size_t i = 0; i < s.atoms.size(); ++i)
    {
        Atom const& a = s.atoms.at(i);
        for (size_t k = 0; k < a.G.size(); ++k) sum += weight(i, k) * a.G.at(k);
    }
    return sum;
}

// Cu2S_PBE cell with the all-types settings above, set up without scaling.
struct FixtureAllTypes : public FixtureRepairDir
{
    FixtureAllTypes()
    {
        BOOST_REQUIRE(bfs::create_directory("test"));
        bfs::copy_file("../../examples/nnp-predict/Cu2S_PBE/input.data",
                       "test/input.data");
        bfs::current_path("test");
        ofstream file("input.nn");
        file << settingsFile;
        file.close();

        mode.log.writeToStdout = false;
        mode.initialize();
        mode.loadSettingsFile();
        mode.setupGeneric();
        mode.setupSymmetryFunctionScalingNone();
        mode.setupSymmetryFunctionStatistics(false, false, false, false);

        base.setElementMap(mode.elementMap);
        base.readFromFile("input.data");
    }

    ~FixtureAllTypes()
    {
        bfs::current_path("..");
        bfs::remove_all("test");
    }

    Mode      mode;
    Structure base;
};

BOOST_AUTO_TEST_SUITE(RegressionTests)

BOOST_FIXTURE_TEST_CASE(AllTypesPeriodic_ForcesMatchFiniteDifference,
                        FixtureAllTypes)
{

    Structure s = base;
    s.calculateNeighborList(mode.getMaxCutoffRadius());
    mode.calculateSymmetryFunctionGroups(s, true);
    for (size_t i = 0; i < s.atoms.size(); ++i)
    {
        Atom& a = s.atoms.at(i);
        for (size_t k = 0; k < a.numSymmetryFunctions; ++k)
        {
            a.dEdG.at(k) = weight(i, k);
        }
    }
    mode.calculateForces(s);

    size_t const n = base.atoms.size();
    double maxDeviation = 0.0;
    for (size_t j : {size_t(0), n / 2, n - 1})
    {
        for (size_t c = 0; c < 3; ++c)
        {
            double const sm2 = weightedSum(mode, base, j, c, -2.0 * h);
            double const sm1 = weightedSum(mode, base, j, c, -1.0 * h);
            double const sp1 = weightedSum(mode, base, j, c,  1.0 * h);
            double const sp2 = weightedSum(mode, base, j, c,  2.0 * h);
            double const fd = -(sm2 - 8.0 * sm1 + 8.0 * sp1 - sp2) / (12.0 * h);
            double const f = s.atoms.at(j).f[c];
            maxDeviation = max(maxDeviation, fabs(f - fd));
            BOOST_TEST_INFO("atom " << j << " component " << c
                            << ": analytic " << f << ", finite difference "
                            << fd);
            BOOST_REQUIRE_LE(fabs(f - fd),
                             toleranceAbsolute + toleranceRelative * fabs(f));
        }
    }
    BOOST_TEST_MESSAGE("max |F_analytic - F_fd| = " << maxDeviation);
}

// The grouped evaluation (calculateSymmetryFunctionGroups(), used in
// production) must give
// the same values and derivatives, own and neighbor-side, as evaluating
// every function individually (calculateSymmetryFunctions()).
BOOST_FIXTURE_TEST_CASE(AllTypesPeriodic_GroupsMatchIndividualFunctions,
                        FixtureAllTypes)
{
    Structure grouped = base;
    grouped.calculateNeighborList(mode.getMaxCutoffRadius());
    Structure individual = grouped;
    mode.calculateSymmetryFunctionGroups(grouped, true);
    mode.calculateSymmetryFunctions(individual, true);

    double const tolerance = 1.0E-12;
    double maxDeviation = 0.0;
    for (size_t i = 0; i < grouped.atoms.size(); ++i)
    {
        Atom const& g = grouped.atoms.at(i);
        Atom const& d = individual.atoms.at(i);
        BOOST_REQUIRE_EQUAL(g.G.size(), d.G.size());
        for (size_t k = 0; k < g.G.size(); ++k)
        {
            double const dev = max(fabs(g.G.at(k) - d.G.at(k)),
                                   (g.dGdr.at(k) - d.dGdr.at(k)).norm());
            maxDeviation = max(maxDeviation, dev);
            BOOST_TEST_INFO("atom " << i << " symmetry function " << k);
            BOOST_REQUIRE_LE(dev, tolerance * max(1.0, fabs(d.G.at(k))));
        }
        BOOST_REQUIRE_EQUAL(g.neighbors.size(), d.neighbors.size());
        for (size_t n = 0; n < g.neighbors.size(); ++n)
        {
            auto const& gn = g.neighbors.at(n).dGdr;
            auto const& dn = d.neighbors.at(n).dGdr;
            BOOST_REQUIRE_EQUAL(gn.size(), dn.size());
            for (size_t k = 0; k < gn.size(); ++k)
            {
                double const dev = (gn.at(k) - dn.at(k)).norm();
                maxDeviation = max(maxDeviation, dev);
                BOOST_TEST_INFO("atom " << i << " neighbor " << n
                                << " derivative " << k);
                BOOST_REQUIRE_LE(dev, tolerance * max(1.0, dn.at(k).norm()));
            }
        }
    }
    BOOST_TEST_MESSAGE("max |grouped - individual| = " << maxDeviation);
}

BOOST_AUTO_TEST_SUITE_END()
