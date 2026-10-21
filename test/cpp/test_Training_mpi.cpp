// Dedicated multi-rank counterpart to test_Training.cpp. Every other test in
// this suite (including test_Training.cpp itself) is only ever exercised
// single-rank (MPI_COMM_WORLD size 1), which never touches the actual
// inter-process communication paths in Dataset::distributeStructures() /
// sendStructure() / recvStructure() -- exactly the code a rewrite of the
// training pipeline is most likely to change the behaviour of (different
// batching/placement of structures across ranks).
// `make test_Training_mpi` runs it single-rank and then on MPI_TEST_RANKS
// ranks (default 4, see test/cpp/makefile); single-rank it still passes,
// just with a weaker check (see the size < 2 guard below).
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "Training.h"
#include <mpi.h>
#include <string>  // std::string, std::to_string
#include <vector>  // std::vector

using namespace std;
using namespace nnp;

namespace bfs = boost::filesystem;

double const accuracyNumeric = 1.0E-8;

// Total number of structures in examples/nnp-train/H2O_RPBE-D3/input.data,
// counted independently of the code under test (grep -c "^ *begin *$").
size_t const numStructuresTotal = 20;

// Same setup sequence as FixtureH2OTraining in test_Training.cpp, stopping
// after distributeStructures() (the operation under test here). Each rank
// works in its own directory (test_rank<N>) since, unlike every other test
// in this suite, multiple copies of this fixture run concurrently in
// separate MPI processes that share the same launch directory on the shared
// filesystem -- reusing a single shared "test" directory name would race.
struct FixtureH2OTrainingMultiRank
{
    FixtureH2OTrainingMultiRank()
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        dirName = "test_rank" + to_string(rank);

        if (bfs::exists(dirName)) bfs::remove_all(dirName);
        BOOST_REQUIRE(copy_directory_recursively(
            "../../examples/nnp-train/H2O_RPBE-D3", dirName));
        bfs::current_path(dirName);

        training.log.writeToStdout = false;
        training.setupMPI();
        training.initialize();
        training.loadSettingsFile();
        training.setStage(0);
        training.setupGeneric(
            "", training.settingsKeywordExists("normalize_data_set"), false);
        training.setupSymmetryFunctionScaling();
        training.setupSymmetryFunctionStatistics(false, false, false, false);
        training.setupRandomNumberGenerator();
        training.distributeStructures(true);
    }

    ~FixtureH2OTrainingMultiRank()
    {
        bfs::current_path("..");
        bfs::remove_all(dirName);
    }

    int      rank;
    int      size;
    string   dirName;
    Training training;
};

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_FIXTURE_TEST_CASE(DistributeStructures_CoversEveryStructureExactlyOnce,
                        FixtureH2OTrainingMultiRank)
{
    // Local presence bitmap: localCount.at(i) == 1 iff this rank owns the
    // structure with that global index.
    vector<int> localCount(numStructuresTotal, 0);
    for (Structure const& s : training.structures)
    {
        BOOST_REQUIRE(s.index < numStructuresTotal);
        localCount.at(s.index)++;
    }

    vector<int> globalCount(numStructuresTotal, 0);
    MPI_Allreduce(localCount.data(), globalCount.data(),
                  (int)numStructuresTotal, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Every structure must be distributed to exactly one rank: not
    // duplicated (would corrupt training statistics/double-count it in
    // error metrics) and not dropped (would silently shrink the data set).
    for (size_t i = 0; i < numStructuresTotal; ++i)
    {
        BOOST_REQUIRE_EQUAL(globalCount.at(i), 1);
    }
}

BOOST_FIXTURE_TEST_CASE(SendRecvStructure_RoundTrip_PreservesContentAcrossRanks,
                        FixtureH2OTrainingMultiRank)
{
    if (size < 2) return;

    int const tag = 99;
    if (rank == 0)
    {
        BOOST_REQUIRE(training.structures.size() > 0);
        Structure const& s = training.structures.at(0);

        // Reference values sent through a plain MPI_Send, independent of
        // Dataset's own (de)serialization, so the check on rank 1 below is
        // not just comparing the serialization pipeline against itself.
        BOOST_REQUIRE(s.numAtoms > 0);
        double refValues[4] = { (double)s.index, s.energy,
                                (double)s.numAtoms, s.atoms.at(0).r[0] };
        MPI_Send(refValues, 4, MPI_DOUBLE, 1, tag, MPI_COMM_WORLD);

        training.sendStructure(s, 1);
    }
    else if (rank == 1)
    {
        double refValues[4];
        MPI_Recv(refValues, 4, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        Structure received;
        training.recvStructure(&received, 0);

        BOOST_REQUIRE_EQUAL((double)received.index, refValues[0]);
        BOOST_REQUIRE_SMALL(received.energy - refValues[1], accuracyNumeric);
        BOOST_REQUIRE_EQUAL((double)received.numAtoms, refValues[2]);

        // Spot-check the first atom's position survived the pack/send/recv/
        // unpack round trip over a real MPI channel (not just the metadata).
        BOOST_REQUIRE(received.numAtoms > 0);
        BOOST_REQUIRE_SMALL(received.atoms.at(0).r[0] - refValues[3],
                            accuracyNumeric);
    }
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
