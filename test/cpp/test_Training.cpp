// This file needs its own main() (calling MPI_Init/MPI_Finalize around
// Boost.Test's runner) for the same reason as test_Dataset.cpp: Training
// inherits from Dataset, which is fundamentally MPI-coupled. Run single-
// rank (no mpirun wrapper needed). See test/cpp/makefile's
// "test_Training: CC=$(MPICC)" override.
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "Training.h"
#include <cmath>     // std::fabs
#include <fstream>   // std::ifstream
#include <iterator>  // std::istreambuf_iterator
#include <mpi.h>
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

double const accuracyNumeric = 1.0E-4;

// Mirrors the setup sequence in src/application/nnp-train.cpp exactly (up
// through setupTraining() + calculateNeighborLists(), stopping short of the
// full multi-epoch loop() -- reusing the real H2O_RPBE-D3 example already
// shipped in the repo for nnp-train (epochs=2, update_strategy=combined,
// small enough for a fast unit test).
struct FixtureH2OTraining : public FixtureRepairDir
{
    FixtureH2OTraining()
    {
        BOOST_REQUIRE(copy_directory_recursively(
            "../../examples/nnp-train/H2O_RPBE-D3", "test"));
        bfs::current_path("test");

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
    }

    ~FixtureH2OTraining()
    {
        bfs::current_path("..");
        bfs::remove_all("test");
    }

    Training training;
};

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_FIXTURE_TEST_CASE(SetupPipeline_PopulatesExpectedState,
                        FixtureH2OTraining)
{
    BOOST_REQUIRE_EQUAL(training.getNumElements(), 2);
    BOOST_REQUIRE(training.structures.size() > 0);
    BOOST_REQUIRE(training.getNumConnections() > 0);
}

BOOST_FIXTURE_TEST_CASE(GetNumConnections_ConsistentWithPerElementBreakdown,
                        FixtureH2OTraining)
{
    vector<size_t> const perElement = training.getNumConnectionsPerElement();
    vector<size_t> const offsets = training.getConnectionOffsets();

    BOOST_REQUIRE_EQUAL(perElement.size(), 2);
    BOOST_REQUIRE_EQUAL(offsets.size(), 2);

    size_t sum = 0;
    for (size_t i = 0; i < perElement.size(); ++i)
    {
        BOOST_REQUIRE_EQUAL(offsets.at(i), sum);
        sum += perElement.at(i);
    }
    BOOST_REQUIRE_EQUAL(training.getNumConnections(), sum);
}

BOOST_FIXTURE_TEST_CASE(SelectSetsAndWriteSetsToFiles_CreatesTrainAndTestFiles,
                        FixtureH2OTraining)
{
    BOOST_REQUIRE_NO_THROW(training.writeSetsToFiles());

    ifstream trainFile("train.data");
    BOOST_REQUIRE(trainFile.is_open());
    string const trainContent((istreambuf_iterator<char>(trainFile)),
                              istreambuf_iterator<char>());
    BOOST_REQUIRE(trainContent.find("begin") != string::npos);

    ifstream testFile("test.data");
    BOOST_REQUIRE(testFile.is_open());
}

BOOST_FIXTURE_TEST_CASE(GetSetSingleWeight_RoundTrip, FixtureH2OTraining)
{
    // getSingleWeight()/setSingleWeight() are only meaningful (per their
    // own docstrings) with per-element weight layout.
    training.initializeWeightsMemory(Training::US_ELEMENT);

    double const original = training.getSingleWeight(0, 0);
    double const newValue = original + 0.1234;

    training.setSingleWeight(0, 0, newValue);

    BOOST_REQUIRE_SMALL(training.getSingleWeight(0, 0) - newValue,
                        accuracyNumeric);
}

BOOST_FIXTURE_TEST_CASE(
    CalculateWeightDerivatives_Energy_MatchesFiniteDifference,
    FixtureH2OTraining)
{
    training.initializeWeightsMemory(Training::US_ELEMENT);

    Structure& s = training.structures.at(0);

    vector<vector<double>> const dEdc =
        training.calculateWeightDerivatives(&s);

    // Perturb weight 0 of element 0 and finite-difference the total energy.
    double const original = training.getSingleWeight(0, 0);
    double const h = 1.0E-4;

    training.setSingleWeight(0, 0, original + h);
    training.calculateAtomicNeuralNetworks(s, false);
    training.calculateEnergy(s);
    double const eHi = s.energy;

    training.setSingleWeight(0, 0, original - h);
    training.calculateAtomicNeuralNetworks(s, false);
    training.calculateEnergy(s);
    double const eLo = s.energy;

    training.setSingleWeight(0, 0, original);

    double const numDeriv = (eHi - eLo) / (2.0 * h);

    BOOST_REQUIRE_SMALL(dEdc.at(0).at(0) - numDeriv, accuracyNumeric);
}

BOOST_FIXTURE_TEST_CASE(WriteWeights_CreatesOneFilePerElement,
                        FixtureH2OTraining)
{
    BOOST_REQUIRE_NO_THROW(
        training.writeWeights("short", "weights_out.%03zu.data"));

    // Element atomic numbers for H, O are 1, 8.
    ifstream fileH("weights_out.001.data");
    BOOST_REQUIRE(fileH.is_open());
    ifstream fileO("weights_out.008.data");
    BOOST_REQUIRE(fileO.is_open());
}

BOOST_FIXTURE_TEST_CASE(CalculateErrorEpochAndWriteLearningCurve_CreatesFile,
                        FixtureH2OTraining)
{
    BOOST_REQUIRE_NO_THROW(training.calculateErrorEpoch());
    BOOST_REQUIRE_NO_THROW(training.writeLearningCurve(false));

    ifstream check("learning-curve.out");
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
