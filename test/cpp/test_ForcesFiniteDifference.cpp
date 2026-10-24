#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ForcesFiniteDifference
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "Prediction.h"
#include "Structure.h"
#include <algorithm> // std::max
#include <cmath>     // std::fabs
#include <cstddef>   // std::size_t
#include <fstream>   // std::ofstream
#include <map>       // std::map
#include <string>    // std::string
#include <utility>   // std::pair
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

// Checks that the analytic forces returned by Mode::evaluateNNP() (via
// Prediction) are the negative gradient of the energy it returns, using a
// 4-point central finite difference of the total energy. Unlike the
// reference-value comparisons in test_Prediction.cpp, this needs no stored
// numbers, covers every example in examples/nnp-predict (2G and 4G,
// periodic and non-periodic, including the 4G ones test_Prediction skips),
// and holds for any correct implementation, in any summation order.
// It catches errors in force assembly (dE/dG . dG/dr scatter), symmetry
// function derivatives, the NN backward pass and, for 4G, the charge
// equilibration and electrostatic force terms.

// Displacement for the finite difference (in the length unit of the
// example's input.data). The 4-point stencil has a truncation error of
// order h^4 and a round-off error of order eps * |E| / h. 4G examples use a
// smaller step: their truncated Ewald sum is not perfectly smooth (pairs
// crossing the real-space cutoff), so a wider stencil can straddle a jump.
double const h2G = 1.0E-3;
double const h4G = 1.0E-4;

// A force component passes if |F_analytic - F_fd| <= absolute + relative *
// |F_analytic|, in the force unit of the example.
double const toleranceAbsolute = 1.0E-6;
double const toleranceRelative = 1.0E-5;

// Energy (physical units) of base with atom i displaced by dx along axis c.
double energyDisplaced(Prediction&      p,
                       Structure const& base,
                       size_t           i,
                       size_t           c,
                       double           dx)
{
    p.structure = base;
    p.structure.atoms.at(i).r[c] += dx;
    if (p.useNormalization())
    {
        p.structure.toNormalizedUnits(p.getMeanEnergy(),
                                      p.getConvEnergy(),
                                      p.getConvLength(),
                                      p.getConvCharge());
    }
    p.evaluateNNP(p.structure, false);
    if (p.useNormalization())
    {
        p.structure.toPhysicalUnits(p.getMeanEnergy(),
                                    p.getConvEnergy(),
                                    p.getConvLength(),
                                    p.getConvCharge());
    }
    return p.structure.energy;
}

// The truncated Ewald energy and the analytic forces only agree to roughly
// the requested Ewald precision; at the example's ewald_prec 1.e-5,
// AlAuMgO_4G forces deviate from -dE/dr by up to ~5E-4 (about 1% of the
// largest force). Tighten it so the test checks the force formulas, not the
// truncation. (H2O_RPBE-D3_4G already agrees to ~1E-7 at its setting, and
// tightening it makes the test very slow.)
map<string, pair<string, string>> const ewaldPrecisionOverrides = {
    {"AlAuMgO_4G", {"ewald_prec 1.e-5", "ewald_prec 1.e-10"}}};

vector<string> const examples = {"AlAuMgO_4G",
                                 "Anisole_SCAN",
                                 "Carbon_chain_4G",
                                 "Cu2S_PBE",
                                 "DMABN_SCAN",
                                 "Ethylbenzene_SCAN",
                                 "H2O_RPBE-D3",
                                 "H2O_RPBE-D3_4G"};

// Compares the analytic forces of the structure in input.data (current
// directory) with finite differences, for the first, middle and last atom.
void requireForcesMatchFiniteDifference(string const& name)
{
    Prediction p;
    p.log.writeToStdout = false;
    p.setup();
    // The displaced structures push some symmetry functions marginally past
    // the range in scaling.data (computed from these same structures). That
    // does not affect the force check; do not flood the output with
    // extrapolation warnings (written to stderr, not to the log).
    p.setupSymmetryFunctionStatistics(false, false, false, false);
    p.readStructureFromFile();
    p.predict();
    Structure const analytic = p.structure;

    Structure base;
    base.setElementMap(p.elementMap);
    base.readFromFile("input.data");

    double const h = p.getNnpType() == Mode::NNPType::HDNNP_4G ? h4G : h2G;
    size_t const n = base.atoms.size();
    vector<size_t> const atoms = {0, n / 2, n - 1};
    double maxDeviation = 0.0;
    for (auto i : atoms)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            double const em2 = energyDisplaced(p, base, i, c, -2.0 * h);
            double const em1 = energyDisplaced(p, base, i, c, -1.0 * h);
            double const ep1 = energyDisplaced(p, base, i, c,  1.0 * h);
            double const ep2 = energyDisplaced(p, base, i, c,  2.0 * h);
            double const fd = -(em2 - 8.0 * em1 + 8.0 * ep1 - ep2)
                            / (12.0 * h);
            double const f = analytic.atoms.at(i).f[c];
            double const deviation = fabs(f - fd);
            maxDeviation = max(maxDeviation, deviation);
            BOOST_TEST_INFO(name << " atom " << i << " component " << c
                            << ": analytic " << f << ", finite difference "
                            << fd);
            BOOST_REQUIRE_LE(deviation,
                             toleranceAbsolute + toleranceRelative * fabs(f));
        }
    }
    BOOST_TEST_MESSAGE(name << ": max |F_analytic - F_fd| = "
                       << maxDeviation);

    return;
}

BOOST_AUTO_TEST_SUITE(RegressionTests)

BOOST_DATA_TEST_CASE_F(FixtureRepairDir,
                       AnalyticForces_MatchFiniteDifference,
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

    requireForcesMatchFiniteDifference(name);

    bfs::current_path("..");
    bfs::remove_all("test");
}

// A cell vector shorter than the symmetry function cutoff: each atom has
// periodic images of itself among its neighbors, whose contributions must
// be included in the forces. One water molecule (from the H2O_RPBE-D3
// example) in a cubic box of 10 bohr, cutoff 12 bohr, with the 2G and 4G
// water models.
vector<string> const smallCellModels = {"H2O_RPBE-D3", "H2O_RPBE-D3_4G"};

BOOST_DATA_TEST_CASE_F(FixtureRepairDir,
                       AnalyticForces_SmallCell_MatchFiniteDifference,
                       bdata::make(smallCellModels),
                       name)
{
    BOOST_REQUIRE(copy_directory_recursively("../../examples/nnp-predict/"
                                             + name, "test"));
    bfs::current_path("test");
    {
        ofstream file("input.data");
        file << "begin\n"
             << "lattice 10.0 0.0 0.0\n"
             << "lattice 0.0 10.0 0.0\n"
             << "lattice 0.0 0.0 10.0\n"
             << "atom 5.0000000 5.0000000 5.0000000 O 0.0 0.0 0.0 0.0 0.0\n"
             << "atom 4.0625069 5.3600306 6.5729116 H 0.0 0.0 0.0 0.0 0.0\n"
             << "atom 6.7495085 5.0876455 5.5389499 H 0.0 0.0 0.0 0.0 0.0\n"
             << "energy 0.0\n"
             << "charge 0.0\n"
             << "end\n";
    }

    requireForcesMatchFiniteDifference(name + " (small cell)");

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()
