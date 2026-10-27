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
#include "utility.h"
#include <cmath>     // std::fabs
#include <cstring>   // std::memset
#include <fstream>   // std::ifstream, std::ofstream
#include <iterator>  // std::istreambuf_iterator
#include <mpi.h>
#include <new>       // placement new
#include <stdexcept> // std::runtime_error
#include <string>    // std::string
#include <type_traits> // std::aligned_storage
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

// Identical setup to FixtureH2OTraining, except the copied input.nn is
// patched to select gradient descent (updater_type 0) instead of the
// example's default Kalman filter (updater_type 1) -- Training::update()
// dispatches to fundamentally different code (GradientDescent vs
// KalmanFilter) depending on this setting, and the KF-only Loop test above
// never exercises the GD branch of that dispatch end to end.
struct FixtureH2OTrainingGD : public FixtureRepairDir
{
    FixtureH2OTrainingGD()
    {
        BOOST_REQUIRE(copy_directory_recursively(
            "../../examples/nnp-train/H2O_RPBE-D3", "test"));
        bfs::current_path("test");

        BOOST_REQUIRE(replaceInFile("input.nn",
                                    "updater_type                    1",
                                    "updater_type                    0"));
        // Gradient descent only supports jacobian_mode JM_SUM (0); the
        // example's default (1, "per-task summed gradient") is only valid
        // together with the Kalman filter updater.
        BOOST_REQUIRE(replaceInFile("input.nn",
                                    "jacobian_mode                   1",
                                    "jacobian_mode                   0"));

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

    ~FixtureH2OTrainingGD()
    {
        bfs::current_path("..");
        bfs::remove_all("test");
    }

    Training training;
};

// Runs the given number of epochs of 4G stage-2 (energy and force) training
// on the H2O_RPBE-D3_4G example, with or without memorize_symfunc_results
// and with the given short_force_fraction, and returns the stage-2
// learning-curve values {energy RMSE train, test, force RMSE train, test} of
// epochs 0 to epochs. Same setup sequence as src/application/nnp-train.cpp
// for stage 2.
vector<double> trainStage2Epochs4G(bool          memorize,
                                   size_t        epochs,
                                   string const& forceFraction)
{
    if (bfs::exists("test")) bfs::remove_all("test");
    BOOST_REQUIRE(copy_directory_recursively(
        "../../examples/nnp-train/H2O_RPBE-D3_4G", "test"));
    bfs::current_path("test");
    BOOST_REQUIRE(replaceInFile("input.nn",
                                "epochs                          10",
                                "epochs                          "
                                + to_string(epochs)));
    BOOST_REQUIRE(replaceInFile("input.nn",
                                "short_force_fraction            0.025",
                                "short_force_fraction            "
                                + forceFraction));
    if (memorize)
    {
        BOOST_REQUIRE(replaceInFile("input.nn",
                                    "#memorize_symfunc_results",
                                    "memorize_symfunc_results"));
    }

    {
        Training training;
        training.log.writeToStdout = false;
        training.setupMPI();
        training.initialize();
        training.loadSettingsFile();
        training.setStage(2);
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

    vector<double> rows(4 * (epochs + 1), 0.0);
    ifstream file("learning-curve.out.stage-2");
    BOOST_REQUIRE(file.is_open());
    string line;
    while (getline(file, line))
    {
        vector<string> c = split(reduce(line));
        if (c.empty() || c.at(0).at(0) == '#') continue;
        size_t const e = stoul(c.at(0));
        BOOST_REQUIRE(e <= epochs);
        rows.at(4 * e + 0) = stod(c.at(1));
        rows.at(4 * e + 1) = stod(c.at(2));
        rows.at(4 * e + 2) = stod(c.at(9));
        rows.at(4 * e + 3) = stod(c.at(10));
    }
    file.close();
    bfs::current_path("..");
    bfs::remove_all("test");
    return rows;
}

// Copy of the H2O_RPBE-D3_4G example for 4G stage-1 tests, which may modify
// the files before running the setup.
struct FixtureH2O4GStage1 : public FixtureRepairDir
{
    FixtureH2O4GStage1()
    {
        BOOST_REQUIRE(copy_directory_recursively(
            "../../examples/nnp-train/H2O_RPBE-D3_4G", "test"));
        bfs::current_path("test");
    }

    ~FixtureH2O4GStage1()
    {
        bfs::current_path("..");
        bfs::remove_all("test");
    }

    // Same setup sequence as src/application/nnp-train.cpp for stage 1, up
    // to the weight initialization.
    void setupStage1()
    {
        training.log.writeToStdout = false;
        training.setupMPI();
        training.initialize();
        training.loadSettingsFile();
        training.setStage(1);
        training.setupGeneric(
            "", training.settingsKeywordExists("normalize_data_set"), true);
        training.setupSymmetryFunctionScaling();
        training.setupSymmetryFunctionStatistics(false, false, false, false);
        training.setupRandomNumberGenerator();
        training.distributeStructures(true);
        training.selectSets();
        training.initializeWeights();
    }

    // Atomic hardness of H and O (physical units) currently in the training.
    vector<double> hardness()
    {
        training.writeHardness("hardness.%03zu.check");
        return {readColumnsFromFile("hardness.001.check", {0}).at(0).at(0),
                readColumnsFromFile("hardness.008.check", {0}).at(0).at(0)};
    }

    Training training;
};

void writeHardnessFile(string const& fileName, double hardness)
{
    ofstream file(fileName.c_str());
    file << hardness << "\n";
    file.close();
}

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

// Energy and force Jacobians with respect to the weights, for the first,
// middle and last weight of every element (first hidden layer, a hidden
// layer, output bias), against central finite differences. These are the
// quantities Training::update() feeds to the updater, so a layout or
// indexing error in any layer shows up here even if it cancels for
// weight 0.
double const hWeight = 1.0E-4;
double const toleranceJacobianAbsolute = 1.0E-6;
double const toleranceJacobianRelative = 1.0E-5;

BOOST_FIXTURE_TEST_CASE(
    CalculateWeightDerivatives_EnergySampledWeights_MatchFiniteDifference,
    FixtureH2OTraining)
{
    training.initializeWeightsMemory(Training::US_ELEMENT);
    Structure& s = training.structures.at(0);
    vector<vector<double>> const dEdc =
        training.calculateWeightDerivatives(&s);

    auto energy = [&]()
    {
        training.calculateAtomicNeuralNetworks(s, false);
        training.calculateEnergy(s);
        return s.energy;
    };
    for (size_t e = 0; e < dEdc.size(); ++e)
    {
        size_t const n = dEdc.at(e).size();
        for (size_t w : {size_t(0), n / 2, n - 1})
        {
            double const original = training.getSingleWeight(e, w);
            training.setSingleWeight(e, w, original + hWeight);
            double const eHi = energy();
            training.setSingleWeight(e, w, original - hWeight);
            double const eLo = energy();
            training.setSingleWeight(e, w, original);
            double const fd = (eHi - eLo) / (2.0 * hWeight);
            BOOST_TEST_INFO("element " << e << ", weight " << w
                            << ": analytic " << dEdc.at(e).at(w)
                            << ", finite difference " << fd);
            BOOST_REQUIRE_LE(fabs(dEdc.at(e).at(w) - fd),
                             toleranceJacobianAbsolute
                             + toleranceJacobianRelative * fabs(fd));
        }
    }
}

BOOST_FIXTURE_TEST_CASE(
    CalculateWeightDerivatives_Force_MatchesFiniteDifference,
    FixtureH2OTraining)
{
    training.initializeWeightsMemory(Training::US_ELEMENT);
    Structure& s = training.structures.at(0);
    size_t const atom = 0;
    size_t const component = 1;
    vector<vector<double>> const dFdc =
        training.calculateWeightDerivatives(&s, atom, component);

    auto force = [&]()
    {
        training.calculateAtomicNeuralNetworks(s, true);
        training.calculateForces(s);
        return s.atoms.at(atom).f[component];
    };
    for (size_t e = 0; e < dFdc.size(); ++e)
    {
        size_t const n = dFdc.at(e).size();
        for (size_t w : {size_t(0), n / 2, n - 1})
        {
            double const original = training.getSingleWeight(e, w);
            training.setSingleWeight(e, w, original + hWeight);
            double const fHi = force();
            training.setSingleWeight(e, w, original - hWeight);
            double const fLo = force();
            training.setSingleWeight(e, w, original);
            double const fd = (fHi - fLo) / (2.0 * hWeight);
            BOOST_TEST_INFO("element " << e << ", weight " << w
                            << ": analytic " << dFdc.at(e).at(w)
                            << ", finite difference " << fd);
            BOOST_REQUIRE_LE(fabs(dFdc.at(e).at(w) - fd),
                             toleranceJacobianAbsolute
                             + toleranceJacobianRelative * fabs(fd));
        }
    }
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

BOOST_FIXTURE_TEST_CASE(Loop_RunsConfiguredEpochsAndWritesLearningCurve,
                        FixtureH2OTraining)
{
    // The example's input.nn sets epochs=2 -- exercises the full training
    // loop (selection-mode/schedule bookkeeping, per-property update()
    // dispatch, weight/hardness/learning-curve/updater-status/neuron-stats
    // writing) end to end, the same sequence src/application/nnp-train.cpp
    // itself runs, but in-process so it is actually captured by coverage
    // instrumentation (test_nnp-train.cpp only spawns the bin/nnp-train
    // subprocess, whose coverage isn't measured this way).
    BOOST_REQUIRE_NO_THROW(training.loop());

    ifstream check("learning-curve.out");
    BOOST_REQUIRE(check.is_open());
    string line;
    size_t numDataLines = 0;
    while (getline(check, line))
    {
        if (!line.empty() && line.at(0) != '#') numDataLines++;
    }
    // One line for the initial (pre-loop) error evaluation plus one per
    // training epoch (epochs=2 in this example's input.nn).
    BOOST_REQUIRE_EQUAL(numDataLines, 3);
}

BOOST_FIXTURE_TEST_CASE(
    Loop_GradientDescent_RunsConfiguredEpochsAndWritesLearningCurve,
    FixtureH2OTrainingGD)
{
    // Same end-to-end loop() check as the Kalman-filter test above, but with
    // updater_type patched to 0 -- covers the UT_GD branch of
    // Training::update()'s dispatch, which the example's own default
    // (Kalman filter) never exercises.
    BOOST_REQUIRE_NO_THROW(training.loop());

    ifstream check("learning-curve.out");
    BOOST_REQUIRE(check.is_open());
    string line;
    size_t numDataLines = 0;
    while (getline(check, line))
    {
        if (!line.empty() && line.at(0) != '#') numDataLines++;
    }
    BOOST_REQUIRE_EQUAL(numDataLines, 3);
}

// The update counter in the training log (column "count") starts at 1. It
// used to be left uninitialized; constructing the Training object in memory
// filled with ones makes such a member visible deterministically.
BOOST_FIXTURE_TEST_CASE(Loop_TrainingLogCountsUpdatesFromOne,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(copy_directory_recursively(
        "../../examples/nnp-train/H2O_RPBE-D3", "test"));
    bfs::current_path("test");

    typedef aligned_storage<sizeof(Training), alignof(Training)>::type
        Storage;
    Storage* storage = new Storage;
    memset(storage, 0xff, sizeof(Storage));
    Training* t = new (storage) Training();
    Training& training = *t;
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
    if (training.useNormalization()) training.toNormalizedUnits();
    training.setupTraining();
    training.calculateNeighborLists();
    BOOST_REQUIRE_NO_THROW(training.loop());
    t->~Training();
    delete storage;

    // One line per update candidate, the count increases by one per update
    // (several lines with the same count only for multiple streams).
    ifstream log("train-log.out");
    BOOST_REQUIRE(log.is_open());
    string line;
    size_t previous = 0;
    size_t numLines = 0;
    while (getline(log, line))
    {
        vector<string> const columns = split(reduce(line));
        if (columns.empty() || columns.at(0).at(0) == '#') continue;
        size_t const count = stoul(columns.at(2));
        BOOST_TEST_INFO("training log line: " << line);
        if (numLines == 0) BOOST_REQUIRE_EQUAL(count, 1);
        else BOOST_REQUIRE(count == previous || count == previous + 1);
        previous = count;
        numLines++;
    }
    BOOST_REQUIRE(numLines > 0);

    bfs::current_path("..");
    bfs::remove_all("test");
}

// Data set normalization is not implemented for 4G-HDNNPs, also not in
// stage 2: the statistics would be computed without charges and the
// electrostatics setup would stay in physical units.
BOOST_FIXTURE_TEST_CASE(Stage2_4G_NormalizeDataSetThrows, FixtureRepairDir)
{
    BOOST_REQUIRE(copy_directory_recursively(
        "../../examples/nnp-train/H2O_RPBE-D3_4G", "test"));
    bfs::current_path("test");
    {
        ofstream file("input.nn", ios::app);
        file << "normalize_data_set force\n";
    }

    {
        Training training;
        training.log.writeToStdout = false;
        training.setupMPI();
        training.initialize();
        training.loadSettingsFile();
        training.setStage(2);
        training.setupGeneric(
            "", training.settingsKeywordExists("normalize_data_set"), false);
        training.setupSymmetryFunctionScaling();
        training.setupSymmetryFunctionStatistics(false, false, false, false);
        training.setupRandomNumberGenerator();
        training.distributeStructures(true);
        training.selectSets();
        training.initializeWeights();
        BOOST_CHECK_THROW(training.dataSetNormalization(), runtime_error);
    }

    bfs::current_path("..");
    bfs::remove_all("test");
}

// Keeping symmetry function results in memory is a speed/memory trade-off
// and must not change what is trained. Without memorize_symfunc_results the
// per-atom derivative arrays are freed after each use; in 4G force updates
// the charge equilibration stays cached across trials, so anything freed
// with the atoms (dChidG) has to be recomputed, otherwise the forces and
// dQ/dr used for the update silently miss terms.
BOOST_FIXTURE_TEST_CASE(Stage2_4G_MemorizeSymfuncResults_DoesNotChangeTraining,
                        FixtureRepairDir)
{
    vector<double> const memo = trainStage2Epochs4G(true, 1, "0.025");
    vector<double> const freed = trainStage2Epochs4G(false, 1, "0.025");

    for (size_t i = 0; i < memo.size(); ++i)
    {
        BOOST_TEST_INFO("learning-curve value " << i << ": memorized "
                        << memo.at(i) << ", freed " << freed.at(i));
        BOOST_REQUIRE(memo.at(i) > 0.0);
        BOOST_REQUIRE_SMALL(memo.at(i) - freed.at(i), 1.0E-10 * memo.at(i));
    }
}

// Same for the errors of the learning curve. With short_force_fraction 0.1
// an epoch ends while the force updates of a structure are still in
// progress: its charge equilibration is still cached when the errors are
// calculated, but its dChidG was freed with the atoms. Two epochs, because
// the training errors of one epoch set the update thresholds of the next.
BOOST_FIXTURE_TEST_CASE(
    Stage2_4G_MemorizeSymfuncResults_DoesNotChangeTrainingErrors,
    FixtureRepairDir)
{
    vector<double> const memo = trainStage2Epochs4G(true, 2, "0.1");
    vector<double> const freed = trainStage2Epochs4G(false, 2, "0.1");

    for (size_t i = 0; i < memo.size(); ++i)
    {
        BOOST_TEST_INFO("learning-curve value " << i << ": memorized "
                        << memo.at(i) << ", freed " << freed.at(i));
        BOOST_REQUIRE(memo.at(i) > 0.0);
        BOOST_REQUIRE_SMALL(memo.at(i) - freed.at(i), 1.0E-10 * memo.at(i));
    }
}

// A 4G training from scratch starts from "initial_hardness" in input.nn and
// needs no hardness files.
BOOST_FIXTURE_TEST_CASE(Stage1_4G_FromScratch_UsesInitialHardness,
                        FixtureH2O4GStage1)
{
    bfs::remove("hardness.001.data");
    bfs::remove("hardness.008.data");

    setupStage1();

    vector<double> const h = hardness();
    BOOST_REQUIRE_CLOSE(h.at(0), 10.0, 1.0E-10);
    BOOST_REQUIRE_CLOSE(h.at(1), 10.0, 1.0E-10);
}

// Without an "initial_hardness" entry the hardness of the element would stay
// at zero and could not be trained.
BOOST_FIXTURE_TEST_CASE(Stage1_4G_MissingInitialHardnessThrows,
                        FixtureH2O4GStage1)
{
    BOOST_REQUIRE(replaceInFile("input.nn",
                                "initial_hardness        O",
                                "#initial_hardness        O"));

    BOOST_REQUIRE_THROW(setupStage1(), runtime_error);
}

// The atomic hardness is trained in stage 1 together with the
// electronegativity weights, a restart with "use_old_weights_charge" has to
// continue from the hardness files, not from "initial_hardness".
BOOST_FIXTURE_TEST_CASE(Stage1_4G_UseOldWeightsCharge_ReadsHardnessFiles,
                        FixtureH2O4GStage1)
{
    BOOST_REQUIRE(replaceInFile("input.nn",
                                "#use_old_weights_charge",
                                "use_old_weights_charge"));
    writeHardnessFile("hardness.001.data", 7.5);
    writeHardnessFile("hardness.008.data", 12.25);

    setupStage1();

    vector<double> const h = hardness();
    BOOST_REQUIRE_CLOSE(h.at(0), 7.5, 1.0E-10);
    BOOST_REQUIRE_CLOSE(h.at(1), 12.25, 1.0E-10);
}

BOOST_FIXTURE_TEST_CASE(Stage1_4G_UseOldWeightsCharge_MissingHardnessFileThrows,
                        FixtureH2O4GStage1)
{
    BOOST_REQUIRE(replaceInFile("input.nn",
                                "#use_old_weights_charge",
                                "use_old_weights_charge"));
    bfs::remove("hardness.008.data");

    BOOST_REQUIRE_THROW(setupStage1(), runtime_error);
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
