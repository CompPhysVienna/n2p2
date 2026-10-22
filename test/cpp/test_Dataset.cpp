// This file needs its own main() (calling MPI_Init/MPI_Finalize around
// Boost.Test's runner) because Dataset is fundamentally MPI-coupled:
// setupMPI() and most other methods dereference MPI_COMM_WORLD, which is
// only valid after MPI_Init(). No other file in this test suite does this
// -- every other MPI-touching regression test instead spawns the real
// application binary as a subprocess via mpirun rather than linking
// Dataset in-process. Run single-rank (no mpirun wrapper needed): a
// single-process MPI job is a fully valid MPI_COMM_WORLD of size 1. See
// test/cpp/makefile's "test_Dataset: CC=$(MPICC)" override, needed because
// linking Dataset.o in-process (rather than only spawning a subprocess
// binary that already has it linked in) pulls in real MPI symbol
// references the plain compiler cannot resolve.
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "Atom.h"
#include "Dataset.h"
#include "Structure.h"
#include "Vec3D.h"
#include <cmath>     // std::sqrt
#include <fstream>   // std::ifstream
#include <iterator>  // std::istreambuf_iterator
#include <limits>    // std::numeric_limits
#include <map>       // std::map
#include <mpi.h>
#include <stdexcept> // std::runtime_error
#include <string>    // std::string

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

double const accuracy = 1.0E-6;

// Reuses the real H2O_RPBE-D3 example (20 structures, already shipped in
// the repo and already used by test_Mode.cpp/test_Prediction.cpp) to get a
// fully set-up Dataset (MPI, RNG, elements, symmetry functions, NN
// topology) ready for distributeStructures().
struct FixtureH2ODataset : public FixtureRepairDir
{
    FixtureH2ODataset()
    {
        BOOST_REQUIRE(copy_directory_recursively(
            "../../examples/nnp-atomenv/H2O_RPBE-D3", "test"));
        bfs::current_path("test");

        dataset.log.writeToStdout = false;
        dataset.setupMPI();
        dataset.loadSettingsFile();
        dataset.setupRandomNumberGenerator();
        dataset.setupGeneric();
    }

    ~FixtureH2ODataset()
    {
        bfs::current_path("..");
        bfs::remove_all("test");
    }

    Dataset dataset;
};

// Builds a small, hand-constructed 2-atom structure (not read from a file)
// for the tests below that don't need a realistic dataset, just *some*
// valid Structure -- keeps calculateBufferSize/sendStructure/recvStructure/
// prepareNumericForces fast regardless of how large the real example
// structures are (the first H2O_RPBE-D3 structure alone has 192 atoms,
// which would make prepareNumericForces() allocate 1 + 6*192 copies).
static Structure makeSmallStructure(ElementMap const& em)
{
    Structure s;
    s.setElementMap(em);
    s.isPeriodic = false;
    s.index = 0;

    Atom a0;
    a0.r = Vec3D(0.0, 0.0, 0.0);
    s.addAtom(a0, "H");

    Atom a1;
    a1.r = Vec3D(1.0, 0.0, 0.0);
    s.addAtom(a1, "H");

    s.energy = -1.5;

    return s;
}

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_FIXTURE_TEST_CASE(SetupMPIAndRandomNumberGenerator_NoThrow,
                        FixtureH2ODataset)
{
    // Setup already ran in the fixture constructor without throwing; just
    // confirm the object is otherwise usable afterward.
    BOOST_REQUIRE_NO_THROW(dataset.setupRandomNumberGenerator());
}

BOOST_FIXTURE_TEST_CASE(GetNumStructures_CountsBeginBlocks, FixtureH2ODataset)
{
    ifstream dataFile("input.data");
    BOOST_REQUIRE_EQUAL(dataset.getNumStructures(dataFile), 4);
    dataFile.close();
}

BOOST_FIXTURE_TEST_CASE(
    DistributeStructures_SingleProcess_AllStructuresLocal,
    FixtureH2ODataset)
{
    dataset.distributeStructures(false);

    BOOST_REQUIRE_EQUAL(dataset.structures.size(), 4);
    BOOST_REQUIRE_EQUAL(dataset.structures.at(0).atoms.size(), 192);
}

BOOST_FIXTURE_TEST_CASE(
    ToNormalizedUnitsAndBack_EnergyRoundTrips,
    FixtureH2ODataset)
{
    dataset.distributeStructures(false);
    double const energyBefore = dataset.structures.at(0).energy;

    dataset.toNormalizedUnits();
    // Normalization is active for this example (mean_energy/conv_energy/
    // conv_length are all set), so the energy value must actually change.
    BOOST_REQUIRE(dataset.structures.at(0).energy != energyBefore);

    dataset.toPhysicalUnits();
    BOOST_REQUIRE_SMALL(dataset.structures.at(0).energy - energyBefore,
                        accuracy);
}

BOOST_FIXTURE_TEST_CASE(
    CalculateBufferSizeAndSendRecvStructure_SelfLoopbackRoundTrips,
    FixtureH2ODataset)
{
    Structure const original = makeSmallStructure(dataset.elementMap);

    int const bufferSize = dataset.calculateBufferSize(original);
    BOOST_REQUIRE(bufferSize > 0);

    // Single-process self loopback: rank 0 sends to itself, then
    // immediately receives from itself. Safe for small structures (well
    // under any eager-send threshold), since MPI_Send need not block
    // waiting for a matching receive already posted.
    int const bytesSent = dataset.sendStructure(original, 0);
    BOOST_REQUIRE_EQUAL(bytesSent, bufferSize);

    Structure received;
    int const bytesReceived = dataset.recvStructure(&received, 0);
    BOOST_REQUIRE_EQUAL(bytesReceived, bufferSize);

    BOOST_REQUIRE_EQUAL(received.atoms.size(), original.atoms.size());
    BOOST_REQUIRE_SMALL(received.energy - original.energy, accuracy);
    for (size_t i = 0; i < original.atoms.size(); ++i)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            BOOST_REQUIRE_SMALL(received.atoms.at(i).r[c]
                                - original.atoms.at(i).r[c], accuracy);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(
    PrepareNumericForces_ReturnsExpectedStructureCount,
    FixtureH2ODataset)
{
    // Must be independent of dataset.structures: prepareNumericForces()
    // clears that vector as its very first action, so a reference into it
    // would dangle immediately.
    Structure original = makeSmallStructure(dataset.elementMap);
    size_t const numAtoms = original.atoms.size();

    size_t const numDistributed =
        dataset.prepareNumericForces(original, 1.0E-4);

    // 1 (unperturbed) + 6 (+/- x,y,z) per atom.
    BOOST_REQUIRE_EQUAL(numDistributed, 1 + 6 * numAtoms);
}

BOOST_FIXTURE_TEST_CASE(WriteSymmetryFunctionScaling_CreatesFile,
                        FixtureH2ODataset)
{
    dataset.setupSymmetryFunctionScalingNone();

    // writeSymmetryFunctionScaling() reads per-symmetry-function min/max/
    // mean/sigma from each Element's statistics, which are only populated
    // by actually computing symmetry functions (with statistics collection
    // enabled) on real atoms first.
    dataset.distributeStructures(false);
    // getMaxCutoffRadius() is already in normalized units (setupGeneric()
    // rescaled symmetry-function cutoffs by convLength), but the structure
    // read by distributeStructures() is still in raw physical units -- must
    // normalize it too, or the cutoff looks ~3x the periodic box size and
    // the neighbor search explodes into many periodic images (minutes
    // instead of milliseconds; caught by this taking ~30x longer than
    // every other test in this file put together).
    dataset.toNormalizedUnits();
    dataset.setupSymmetryFunctionStatistics(true, false, false, false);
    Structure& s = dataset.structures.at(0);
    s.calculateNeighborList(dataset.getMaxCutoffRadius());
    dataset.calculateSymmetryFunctionGroups(s, false);

    BOOST_REQUIRE_NO_THROW(
        dataset.writeSymmetryFunctionScaling("scaling_out.data"));

    ifstream check("scaling_out.data");
    BOOST_REQUIRE(check.is_open());
    string const content((istreambuf_iterator<char>(check)),
                         istreambuf_iterator<char>());
    BOOST_REQUIRE(content.find("sf_min") != string::npos);
}

BOOST_FIXTURE_TEST_CASE(SortNeighborLists_NoThrowWithoutNeighborData,
                        FixtureH2ODataset)
{
    // With no structures loaded yet, this is a trivial (empty) no-op --
    // exercises the method's outer loop bounds handling.
    BOOST_REQUIRE_NO_THROW(dataset.sortNeighborLists());
}

BOOST_FIXTURE_TEST_CASE(CollectError_Energy_NormalizesCorrectly,
                        FixtureH2ODataset)
{
    // Single-process MPI_Allreduce is a no-op, so the "collected" sums are
    // exactly what's passed in -- this tests the normalization formula
    // (sqrt(sum/count) for RMSE-type metrics, sum/count for MAE-type).
    map<string, double> error;
    error["RMSEpa"] = 4.0;
    error["RMSE"] = 9.0;
    error["MAEpa"] = 2.0;
    error["MAE"] = 6.0;
    size_t count = 4;

    dataset.collectError("energy", error, count);

    BOOST_REQUIRE_SMALL(error.at("RMSEpa") - sqrt(4.0 / 4.0), accuracy);
    BOOST_REQUIRE_SMALL(error.at("RMSE") - sqrt(9.0 / 4.0), accuracy);
    BOOST_REQUIRE_SMALL(error.at("MAEpa") - 2.0 / 4.0, accuracy);
    BOOST_REQUIRE_SMALL(error.at("MAE") - 6.0 / 4.0, accuracy);
}

BOOST_FIXTURE_TEST_CASE(CollectError_Force_NormalizesCorrectly,
                        FixtureH2ODataset)
{
    map<string, double> error;
    error["RMSE"] = 16.0;
    error["MAE"] = 8.0;
    size_t count = 4;

    dataset.collectError("force", error, count);

    BOOST_REQUIRE_SMALL(error.at("RMSE") - sqrt(16.0 / 4.0), accuracy);
    BOOST_REQUIRE_SMALL(error.at("MAE") - 8.0 / 4.0, accuracy);
}

BOOST_FIXTURE_TEST_CASE(CollectError_UnknownProperty_Throws,
                        FixtureH2ODataset)
{
    map<string, double> error;
    size_t count = 1;
    BOOST_REQUIRE_THROW(dataset.collectError("bogus", error, count),
                        runtime_error);
}

BOOST_FIXTURE_TEST_CASE(CombineFiles_MergesPerRankFileAndRemovesIt,
                        FixtureH2ODataset)
{
    // Single process -> exactly one per-rank file, suffixed ".0000".
    ofstream perRank("combined_out.0000");
    perRank << "line one\nline two\n";
    perRank.close();

    dataset.combineFiles("combined_out");

    ifstream combined("combined_out");
    BOOST_REQUIRE(combined.is_open());
    string const content((istreambuf_iterator<char>(combined)),
                         istreambuf_iterator<char>());
    BOOST_REQUIRE_EQUAL(content, "line one\nline two\n");

    ifstream perRankCheck("combined_out.0000");
    BOOST_REQUIRE(!perRankCheck.is_open());
}

BOOST_FIXTURE_TEST_CASE(WriteNeighborHistogram_CreatesFilesAndCombinesThem,
                        FixtureH2ODataset)
{
    dataset.distributeStructures(false);
    dataset.toNormalizedUnits();
    for (auto& s : dataset.structures)
    {
        s.calculateNeighborList(dataset.getMaxCutoffRadius());
    }

    size_t const maxNeighbors = dataset.writeNeighborHistogram(
        "neighbors_out.histo", "neighbors_out.out");

    BOOST_REQUIRE(maxNeighbors > 0);

    ifstream histo("neighbors_out.histo");
    BOOST_REQUIRE(histo.is_open());

    // combineFiles() (called internally) should have merged the per-rank
    // file into the plain-named one and removed the per-rank suffix file.
    ifstream combined("neighbors_out.out");
    BOOST_REQUIRE(combined.is_open());
    ifstream perRank("neighbors_out.out.0000");
    BOOST_REQUIRE(!perRank.is_open());
}

BOOST_FIXTURE_TEST_CASE(WriteSymmetryFunctionHistograms_CreatesFiles,
                        FixtureH2ODataset)
{
    // setupSymmetryFunctionGroups() already ran inside setupGeneric()
    // (fixture constructor), so groups exist but their per-group
    // scalingFactors array is only populated once a scaling method
    // actually runs -- without this, calculateSymmetryFunctionGroups()
    // indexes into that still-empty array and crashes (same root cause
    // diagnosed earlier for test_Element.cpp's group/scaling ordering).
    dataset.setupSymmetryFunctionScalingNone();

    dataset.distributeStructures(false);
    dataset.toNormalizedUnits();
    dataset.setupSymmetryFunctionStatistics(true, false, false, false);
    Structure& s = dataset.structures.at(0);
    s.calculateNeighborList(dataset.getMaxCutoffRadius());
    dataset.calculateSymmetryFunctionGroups(s, false);

    BOOST_REQUIRE_NO_THROW(dataset.writeSymmetryFunctionHistograms(
        10, "sf_histo.%03zu.%04zu.histo"));

    // First symmetry function of element H (atomic number 1).
    ifstream check("sf_histo.001.0001.histo");
    BOOST_REQUIRE(check.is_open());
}

BOOST_AUTO_TEST_SUITE_END()

bool init_unit_test()
{
    return true;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int const result = boost::unit_test::unit_test_main(&init_unit_test, argc, argv);
    MPI_Finalize();
    return result;
}
