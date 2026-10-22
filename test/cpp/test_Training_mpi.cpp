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
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "Training.h"
#include "utility.h"
#include <cmath>   // std::fabs
#include <fstream> // std::ifstream, std::ofstream
#include <map>     // std::map
#include <mpi.h>
#include <sstream> // std::ostringstream
#include <string>  // std::string, std::to_string
#include <vector>  // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
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

// Sets keyword to value in the input.nn of the current directory (the
// whole line is replaced, whatever its spacing or comment).
void setKeyword(string const& keyword, string const& value)
{
    ifstream in("input.nn");
    ostringstream out;
    string line;
    bool found = false;
    while (getline(in, line))
    {
        vector<string> c = split(reduce(line));
        if (!c.empty() && c.at(0) == keyword)
        {
            out << keyword << " " << value << "\n";
            found = true;
        }
        else out << line << "\n";
    }
    in.close();
    BOOST_REQUIRE_MESSAGE(found, "keyword " + keyword + " not in input.nn");
    ofstream f("input.nn", ios::trunc);
    f << out.str();
}

// Runs the nnp-train sequence (as in src/application/nnp-train.cpp) for one
// epoch of the H2O_RPBE-D3 example with the given keyword settings. All
// ranks work in one shared directory, as nnp-train does (rank 0 combines
// per-rank output files from it). Returns the learning-curve rows of epochs
// 0 and 1 (energy train/test RMSE, force train/test RMSE), read on rank 0
// and broadcast to all ranks.
vector<double> trainOneEpoch(map<string, string> const& keywords,
                             string const&              tag)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    string const dir = "test_" + tag;
    if (rank == 0)
    {
        if (bfs::exists(dir)) bfs::remove_all(dir);
        BOOST_REQUIRE(copy_directory_recursively(
            "../../examples/nnp-train/H2O_RPBE-D3", dir));
        bfs::current_path(dir);
        setKeyword("epochs", "1");
        for (auto const& k : keywords) setKeyword(k.first, k.second);
        bfs::current_path("..");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    bfs::current_path(dir);

    {
        Training training;
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
        training.selectSets();
        training.initializeWeights();
        if (training.settingsKeywordExists("normalize_data_set"))
        {
            training.dataSetNormalization();
        }
        if (training.useNormalization()) training.toNormalizedUnits();
        training.setupTraining();
        training.calculateNeighborLists();
        training.loop();
    }

    vector<double> rows(8, 0.0);
    if (rank == 0)
    {
        ifstream file("learning-curve.out");
        string line;
        while (getline(file, line))
        {
            vector<string> c = split(reduce(line));
            if (c.empty() || c.at(0).at(0) == '#') continue;
            size_t const e = stoul(c.at(0));
            if (e > 1) continue;
            rows.at(4 * e + 0) = stod(c.at(1));
            rows.at(4 * e + 1) = stod(c.at(2));
            rows.at(4 * e + 2) = stod(c.at(9));
            rows.at(4 * e + 3) = stod(c.at(10));
        }
    }
    MPI_Bcast(rows.data(), 8, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    bfs::current_path("..");
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) bfs::remove_all(dir);
    return rows;
}

// Training configurations whose update() branches the examples do not
// reach: {jacobian_mode, update_strategy}. Per-task summed Jacobian with
// combined updates is the example's own setting.
vector<pair<string, string>> const configurations = {{"1", "0"},
                                                     {"2", "0"},
                                                     {"1", "1"}};

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

// parallel_mode 0 (update on rank 0, weights broadcast) and 1 (every rank
// updates) must train identically: only who computes the update differs.
// On several ranks this exercises the MPI_Reduce/MPI_Gather(v) branches of
// Training::update() for PM_TRAIN_RK0 and their MPI_Allreduce/
// MPI_Allgatherv counterparts for PM_TRAIN_ALL, for the full Jacobian
// (JM_FULL) and per-element updates (US_ELEMENT) as well, the parts a
// rewrite of the update or its communication is most likely to break.
BOOST_DATA_TEST_CASE(ParallelModes_TrainIdentically,
                     bdata::make({0, 1, 2}),
                     k)
{
    map<string, string> keywords = {
        {"jacobian_mode", configurations.at(k).first},
        {"update_strategy", configurations.at(k).second}};
    keywords["parallel_mode"] = "0";
    vector<double> const rk0 = trainOneEpoch(keywords, "pm0");
    keywords["parallel_mode"] = "1";
    vector<double> const all = trainOneEpoch(keywords, "pm1");

    for (size_t i = 0; i < rk0.size(); ++i)
    {
        BOOST_TEST_INFO("configuration " << k << ", learning-curve value "
                        << i << ": parallel_mode 0 " << rk0.at(i)
                        << ", parallel_mode 1 " << all.at(i));
        BOOST_REQUIRE(rk0.at(i) > 0.0);
        BOOST_REQUIRE_SMALL(rk0.at(i) - all.at(i), 1.0E-10 * rk0.at(i));
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
