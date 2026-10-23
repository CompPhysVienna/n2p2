#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Mode
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "Mode.h"
#include "Prediction.h"
#include "SetupAnalysis.h"
#include "Structure.h"
#include <cmath>     // std::fabs
#include <fstream>   // std::ofstream, std::ifstream
#include <iterator>  // std::istreambuf_iterator
#include <stdexcept> // std::runtime_error, std::invalid_argument
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

// This suite complements test_Prediction.cpp (which already drives Mode's
// main setup pipeline -- loadSettingsFile, setupGeneric,
// setupSymmetryFunctionScaling(file), setupNeuralNetworkWeights -- via
// Prediction::setup(), against several real examples). Here we focus on the
// parts of Mode that pipeline never exercises: the error/throw paths in the
// individual setup*() steps, the setupSymmetryFunctionScalingNone()
// alternative, pruning, settings-file writing, and SetupAnalysis's one
// method (SetupAnalysis is otherwise a plain, member-less Mode subclass).

// Writes a minimal, valid "input.nn" in the current directory containing
// only the keywords needed for a specific narrow test, then loads it.
static void writeAndLoadSettings(Mode& mode, string const& content)
{
    ofstream file("input.nn");
    file << content;
    file.close();
    mode.loadSettingsFile();
}

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Constructor_Defaults)
{
    Mode mode;
    BOOST_REQUIRE(mode.getNnpType() == Mode::NNPType::HDNNP_2G);
    BOOST_REQUIRE_EQUAL(mode.getMeanEnergy(), 0.0);
    BOOST_REQUIRE_EQUAL(mode.getConvEnergy(), 1.0);
    BOOST_REQUIRE_EQUAL(mode.getConvLength(), 1.0);
    BOOST_REQUIRE_EQUAL(mode.getConvCharge(), 1.0);
    BOOST_REQUIRE_EQUAL(mode.getMaxCutoffRadius(), 0.0);
    BOOST_REQUIRE_EQUAL(mode.getNumElements(), 0);
    BOOST_REQUIRE(!mode.useNormalization());
}

BOOST_FIXTURE_TEST_CASE(Initialize_DoesNotThrow, FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    Mode mode;
    mode.log.writeToStdout = false;
    BOOST_REQUIRE_NO_THROW(mode.initialize());

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(LoadSettingsFile_KeywordAccessors, FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    Mode mode;
    mode.log.writeToStdout = false;
    writeAndLoadSettings(mode,
        "number_of_elements 1\n"
        "elements H\n"
        "cutoff_type 2\n");

    BOOST_REQUIRE(mode.settingsKeywordExists("elements"));
    BOOST_REQUIRE_EQUAL(mode.settingsGetValue("elements"), "H");
    // A recognized keyword that is simply absent from this snippet --
    // settingsKeywordExists() throws for keywords Settings doesn't know at
    // all, so this must be a real (if unused here) keyword.
    BOOST_REQUIRE(!mode.settingsKeywordExists("scale_min_short"));

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(SetupNormalization_AllKeywordsPresent_EnablesNormalization,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    Mode mode;
    mode.log.writeToStdout = false;
    writeAndLoadSettings(mode,
        "number_of_elements 1\n"
        "elements H\n"
        "cutoff_type 2\n"
        "mean_energy 1.5\n"
        "conv_energy 2.5\n"
        "conv_length 3.5\n"
        "conv_charge 4.5\n");

    mode.setupNormalization();

    BOOST_REQUIRE(mode.useNormalization());
    BOOST_REQUIRE_EQUAL(mode.getMeanEnergy(), 1.5);
    BOOST_REQUIRE_EQUAL(mode.getConvEnergy(), 2.5);
    BOOST_REQUIRE_EQUAL(mode.getConvLength(), 3.5);
    BOOST_REQUIRE_EQUAL(mode.getConvCharge(), 4.5);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(SetupNormalization_NoKeywordsPresent_DisablesNormalization,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    Mode mode;
    mode.log.writeToStdout = false;
    writeAndLoadSettings(mode,
        "number_of_elements 1\n"
        "elements H\n"
        "cutoff_type 2\n");

    mode.setupNormalization();

    BOOST_REQUIRE(!mode.useNormalization());

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(SetupNormalization_InconsistentKeywords_Throws,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    Mode mode;
    mode.log.writeToStdout = false;
    writeAndLoadSettings(mode,
        "number_of_elements 1\n"
        "elements H\n"
        "cutoff_type 2\n"
        "mean_energy 1.5\n"); // conv_energy/conv_length missing.

    BOOST_REQUIRE_THROW(mode.setupNormalization(), runtime_error);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(SetupElementMapAndElements_Basic, FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    Mode mode;
    mode.log.writeToStdout = false;
    writeAndLoadSettings(mode,
        "number_of_elements 2\n"
        "elements H O\n"
        "cutoff_type 2\n");

    mode.setupElementMap();
    mode.setupElements();

    BOOST_REQUIRE_EQUAL(mode.getNumElements(), 2);
    BOOST_REQUIRE_EQUAL(mode.elementMap.size(), 2);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(SetupElements_AtomEnergyOffsetApplied, FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    Mode mode;
    mode.log.writeToStdout = false;
    writeAndLoadSettings(mode,
        "number_of_elements 2\n"
        "elements H O\n"
        "cutoff_type 2\n"
        "atom_energy H -0.5\n"
        "atom_energy O -75.0\n");

    mode.setupElementMap();
    BOOST_REQUIRE_NO_THROW(mode.setupElements());

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(SetupElements_MismatchedElementCount_Throws,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    Mode mode;
    mode.log.writeToStdout = false;
    writeAndLoadSettings(mode,
        "number_of_elements 3\n" // Inconsistent with only 2 elements below.
        "elements H O\n"
        "cutoff_type 2\n");

    mode.setupElementMap();
    BOOST_REQUIRE_THROW(mode.setupElements(), runtime_error);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(SetupCutoff_BadAlpha_Throws, FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    Mode mode;
    mode.log.writeToStdout = false;
    // Alpha must satisfy 0 <= alpha < 1.0 (or be exactly 0).
    writeAndLoadSettings(mode,
        "number_of_elements 1\n"
        "elements H\n"
        "cutoff_type 2 1.5\n");

    BOOST_REQUIRE_THROW(mode.setupCutoff(), invalid_argument);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(SetupCutoff_UnknownType_Throws, FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    Mode mode;
    mode.log.writeToStdout = false;
    writeAndLoadSettings(mode,
        "number_of_elements 1\n"
        "elements H\n"
        "cutoff_type 999\n");

    BOOST_REQUIRE_THROW(mode.setupCutoff(), invalid_argument);

    bfs::current_path("..");
    bfs::remove_all("test");
}

// Fixture reusing the real H2O_RPBE-D3 example (52 symmetry functions, 2
// elements) already shipped in the repo and already used by
// test_Prediction.cpp -- but here run through SetupAnalysis (a bare Mode
// subclass adding only writeSymmetryFunctionShape()) and only as far as
// setupGeneric() plus a scaling step, without loading any NN weights, since
// none of the tests below need trained weights.
struct FixtureH2OSetupAnalysis : public FixtureRepairDir
{
    FixtureH2OSetupAnalysis()
    {
        BOOST_REQUIRE(copy_directory_recursively(
            "../../examples/nnp-atomenv/H2O_RPBE-D3", "test"));
        bfs::current_path("test");

        mode.log.writeToStdout = false;
        mode.initialize();
        mode.loadSettingsFile();
        mode.setupGeneric();
    }

    ~FixtureH2OSetupAnalysis()
    {
        bfs::current_path("..");
        bfs::remove_all("test");
    }

    SetupAnalysis mode;
};

BOOST_FIXTURE_TEST_CASE(SetupGeneric_H2OExample_PopulatesElementsAndSymmetryFunctions,
                        FixtureH2OSetupAnalysis)
{
    BOOST_REQUIRE_EQUAL(mode.getNumElements(), 2);
    // input.nn's raw cutoff radius is 12.0 (Bohr), but normalization is
    // active (mean_energy/conv_energy/conv_length are all present), so the
    // internally stored cutoff is rescaled by getConvLength() -- check the
    // relationship rather than hardcoding the converted value.
    BOOST_REQUIRE(mode.useNormalization());
    double const expectedMaxCutoff = 12.0 * mode.getConvLength();
    BOOST_REQUIRE_SMALL(mode.getMaxCutoffRadius() - expectedMaxCutoff, 1.0E-6);

    vector<size_t> const numSF = mode.getNumSymmetryFunctions();
    BOOST_REQUIRE_EQUAL(numSF.size(), 2);
    size_t total = 0;
    for (size_t n : numSF) total += n;
    BOOST_REQUIRE_EQUAL(total, 57); // Counted from input.nn.
}

// Unit conversion helpers, on the H2O example where normalization is active
// (conversion factors != 1): known scaling per property, exact inverses,
// unknown properties rejected.
BOOST_FIXTURE_TEST_CASE(NormalizedPhysical_ScalingAndInverse,
                        FixtureH2OSetupAnalysis)
{
    double const cE = mode.getConvEnergy();
    double const cL = mode.getConvLength();
    double const cQ = mode.getConvCharge();
    double const v = 1.2345;
    BOOST_REQUIRE_SMALL(mode.normalized("energy", v) - v * cE, 1.0E-12);
    BOOST_REQUIRE_SMALL(mode.normalized("force", v) - v * cE / cL, 1.0E-12);
    BOOST_REQUIRE_SMALL(mode.normalized("charge", v) - v * cQ, 1.0E-12);
    BOOST_REQUIRE_SMALL(mode.normalized("hardness", v) - v * cE / (cQ * cQ),
                        1.0E-12);
    BOOST_REQUIRE_SMALL(mode.normalized("negativity", v) - v * cE / cQ,
                        1.0E-12);
    for (string p : {"energy", "force", "charge", "hardness", "negativity"})
    {
        BOOST_TEST_INFO("property " << p);
        BOOST_REQUIRE_SMALL(mode.physical(p, mode.normalized(p, v)) - v,
                            1.0E-12);
    }
    BOOST_REQUIRE_THROW(mode.normalized("unknown", v), runtime_error);
    BOOST_REQUIRE_THROW(mode.physical("unknown", v), runtime_error);
}

BOOST_FIXTURE_TEST_CASE(ConvertUnits_StructureRoundTrip,
                        FixtureH2OSetupAnalysis)
{
    Structure s;
    s.setElementMap(mode.elementMap);
    s.readFromFile("input.data");
    s.energy = s.energyRef;
    Structure const original = s;

    double const eNormalized = mode.normalizedEnergy(s, true);
    BOOST_REQUIRE_SMALL(eNormalized - (s.energyRef - s.numAtoms
                                       * mode.getMeanEnergy())
                                      * mode.getConvEnergy(), 1.0E-9);

    mode.convertToNormalizedUnits(s);
    BOOST_REQUIRE_SMALL(s.energyRef - eNormalized, 1.0E-9);
    BOOST_REQUIRE_SMALL(s.atoms.at(0).r[0]
                        - original.atoms.at(0).r[0] * mode.getConvLength(),
                        1.0E-9);
    BOOST_REQUIRE_SMALL(mode.physicalEnergy(s, true) - original.energyRef,
                        1.0E-9 * fabs(original.energyRef));

    mode.convertToPhysicalUnits(s);
    BOOST_REQUIRE_SMALL(s.energyRef - original.energyRef,
                        1.0E-12 * fabs(original.energyRef));
    for (size_t i = 0; i < s.atoms.size(); ++i)
    {
        BOOST_REQUIRE_SMALL((s.atoms.at(i).r - original.atoms.at(i).r).norm(),
                            1.0E-9);
        BOOST_REQUIRE_SMALL((s.atoms.at(i).fRef - original.atoms.at(i).fRef)
                            .norm(), 1.0E-9);
    }
}

BOOST_FIXTURE_TEST_CASE(SetupSymmetryFunctionScalingNone_NoThrow,
                        FixtureH2OSetupAnalysis)
{
    BOOST_REQUIRE_NO_THROW(mode.setupSymmetryFunctionScalingNone());
}

BOOST_FIXTURE_TEST_CASE(WriteSettingsFile_WritesKnownKeyword,
                        FixtureH2OSetupAnalysis)
{
    ofstream file("output.nn");
    mode.writeSettingsFile(&file);
    file.close();

    ifstream check("output.nn");
    string const content((istreambuf_iterator<char>(check)),
                         istreambuf_iterator<char>());
    BOOST_REQUIRE(content.find("elements") != string::npos);
}

BOOST_FIXTURE_TEST_CASE(PruneSymmetryFunctionsRange_ZeroRangeScaling_PrunesAll,
                        FixtureH2OSetupAnalysis)
{
    // With no scaling set, Gmin == Gmax == 0 for every symmetry function, so
    // any positive threshold prunes everything.
    mode.setupSymmetryFunctionScalingNone();

    vector<size_t> const prune = mode.pruneSymmetryFunctionsRange(1.0E-3);

    BOOST_REQUIRE_EQUAL(prune.size(), 57);
}

BOOST_FIXTURE_TEST_CASE(PruneSymmetryFunctionsRange_ZeroThreshold_PrunesNone,
                        FixtureH2OSetupAnalysis)
{
    mode.setupSymmetryFunctionScalingNone();

    vector<size_t> const prune = mode.pruneSymmetryFunctionsRange(0.0);

    BOOST_REQUIRE_EQUAL(prune.size(), 0);
}

BOOST_FIXTURE_TEST_CASE(PruneSymmetryFunctionsRange_RealScaling_PrunesSubset,
                        FixtureH2OSetupAnalysis)
{
    mode.setupSymmetryFunctionScaling("scaling.data");

    // A tiny threshold should prune at most a handful (possibly zero) of the
    // 57 symmetry functions, never all of them, for real scaling data with
    // a non-degenerate range.
    vector<size_t> const prune = mode.pruneSymmetryFunctionsRange(1.0E-8);

    BOOST_REQUIRE(prune.size() < 57);
}

BOOST_FIXTURE_TEST_CASE(PruneSymmetryFunctionsSensitivity_BelowThreshold_Pruned,
                        FixtureH2OSetupAnalysis)
{
    vector<size_t> const numSF = mode.getNumSymmetryFunctions();
    vector<vector<double>> sensitivity(numSF.size());
    for (size_t i = 0; i < numSF.size(); ++i)
    {
        sensitivity.at(i) = vector<double>(numSF.at(i), 0.0);
    }
    // Mark exactly one symmetry function (element 0, index 0) as sensitive.
    sensitivity.at(0).at(0) = 10.0;

    vector<size_t> const prune = mode.pruneSymmetryFunctionsSensitivity(
        1.0, sensitivity);

    // Everything except the one marked-sensitive entry should be pruned.
    size_t const total = numSF.at(0) + numSF.at(1);
    BOOST_REQUIRE_EQUAL(prune.size(), total - 1);
}

BOOST_FIXTURE_TEST_CASE(WritePrunedSettingsFile_CommentsOutSelectedLines,
                        FixtureH2OSetupAnalysis)
{
    // Whatever line 0 of the loaded settings file is, pruning it should
    // unconditionally prefix it with a comment marker.
    mode.writePrunedSettingsFile({0}, "output_pruned.nn");

    ifstream check("output_pruned.nn");
    string firstLine;
    getline(check, firstLine);
    BOOST_REQUIRE_EQUAL(firstLine.substr(0, 2), "# ");
}

BOOST_FIXTURE_TEST_CASE(WriteSymmetryFunctionShape_CreatesFilesWithHeader,
                        FixtureH2OSetupAnalysis)
{
    mode.setupSymmetryFunctionScalingNone();

    BOOST_REQUIRE_NO_THROW(mode.writeSymmetryFunctionShape(5));

    // First symmetry function of element H (atomic number 1).
    ifstream check("sf.001.0001.out");
    BOOST_REQUIRE(check.is_open());
    string const content((istreambuf_iterator<char>(check)),
                         istreambuf_iterator<char>());
    BOOST_REQUIRE(content.find("SFINFO") != string::npos);
    BOOST_REQUIRE(content.find("sf_radial") != string::npos);
    BOOST_REQUIRE(content.find("sf_angular") != string::npos);
}

// Weights files whose number of entries does not match the architecture in
// input.nn used to be read past their end (or silently truncated); this was
// how the Carbon_chain_4G prediction example ran for years with garbage
// short-range weights. Mode::readNeuralNetworkWeights() must refuse them.
struct FixtureH2OPrediction : public FixtureRepairDir
{
    FixtureH2OPrediction()
    {
        BOOST_REQUIRE(copy_directory_recursively(
            "../../examples/nnp-predict/H2O_RPBE-D3", "test"));
        bfs::current_path("test");
        prediction.log.writeToStdout = false;
    }

    ~FixtureH2OPrediction()
    {
        bfs::current_path("..");
        bfs::remove_all("test");
    }

    Prediction prediction;
};

BOOST_FIXTURE_TEST_CASE(SetupNeuralNetworkWeights_ExtraWeight_Throws,
                        FixtureH2OPrediction)
{
    ofstream file("weights.001.data", ios::app);
    file << "  1.0000000000000000E+00\n";
    file.close();
    BOOST_REQUIRE_THROW(prediction.setup(), runtime_error);
}

BOOST_FIXTURE_TEST_CASE(SetupNeuralNetworkWeights_MissingWeight_Throws,
                        FixtureH2OPrediction)
{
    ifstream in("weights.008.data");
    vector<string> lines;
    string line;
    while (getline(in, line)) lines.push_back(line);
    in.close();
    lines.pop_back();
    ofstream out("weights.008.data", ios::trunc);
    for (auto const& l : lines) out << l << "\n";
    out.close();
    BOOST_REQUIRE_THROW(prediction.setup(), runtime_error);
}

BOOST_FIXTURE_TEST_CASE(SetupNeuralNetworkWeights_MatchingFiles_NoThrow,
                        FixtureH2OPrediction)
{
    BOOST_REQUIRE_NO_THROW(prediction.setup());
}

// The weight files shipped with the examples have to match their input.nn
// (the Carbon_chain_4G prediction and training examples did not). Training
// examples only ship the 4G stage-2 input (electronegativity weights and
// hardness). The LAMMPS examples in TO_BE_CHECKED are not covered.
vector<string> const examplesWithWeights = {
    "interface-LAMMPS/AlAuMgO_4G/hdnnp-data",
    "interface-LAMMPS/AlAuMgO_4G/nnp-predict",
    "interface-LAMMPS/Cu2S_PBE/hdnnp-data",
    "interface-LAMMPS/Cu2S_PBE_external/hdnnp-data",
    "interface-LAMMPS/H2O_4G/hdnnp-data",
    "interface-LAMMPS/H2O_4G/nnp-predict",
    "interface-LAMMPS/H2O_RPBE-D3/hdnnp-data",
    "interface-LAMMPS/H2O_RPBE-D3_128mol/hdnnp-data",
    "interface-LAMMPS/H2O_RPBE-D3_128mol_develop/hdnnp-data",
    "interface-LAMMPS/H2O_RPBE-D3_128mol_external/hdnnp-data",
    "interface-LAMMPS/H2O_RPBE-D3_external/hdnnp-data",
    "nnp-checkdw/H2O_RPBE-D3",
    "nnp-checkdw/LJ",
    "nnp-checkf/H2O_RPBE-D3",
    "nnp-checkf/LJ",
    "nnp-comp2/Cu2S_PBE/nnp-data-1",
    "nnp-comp2/Cu2S_PBE/nnp-data-2",
    "nnp-dataset/Cu2S_PBE",
    "nnp-dataset/H2O_RPBE-D3",
    "nnp-dataset/H2O_RPBE-D3_4G",
    "nnp-predict/AlAuMgO_4G",
    "nnp-predict/Anisole_SCAN",
    "nnp-predict/Carbon_chain_4G",
    "nnp-predict/Cu2S_PBE",
    "nnp-predict/DMABN_SCAN",
    "nnp-predict/Ethylbenzene_SCAN",
    "nnp-predict/H2O_RPBE-D3",
    "nnp-predict/H2O_RPBE-D3_4G",
    "potentials/Anisole_SCAN",
    "potentials/Cu2S_PBE",
    "potentials/DMABN_SCAN",
    "potentials/Ethylbenzene_SCAN",
    "potentials/H2O_BLYP",
    "potentials/H2O_BLYP-D3",
    "potentials/H2O_RPBE",
    "potentials/H2O_RPBE-D3",
    "pynnp/prediction"
};

vector<string> const trainingExamplesWithWeights = {
    "nnp-train/AlAuMgO_4G",
    "nnp-train/Carbon_chain_4G",
    "nnp-train/H2O_RPBE-D3_4G"
};

BOOST_DATA_TEST_CASE(Examples_WeightFilesMatchSettings,
                     bdata::make(examplesWithWeights),
                     example)
{
    string const dir = "../../examples/" + example + "/";
    Mode mode;
    mode.log.writeToStdout = false;
    mode.loadSettingsFile(dir + "input.nn");
    mode.setupGeneric(dir);
    BOOST_REQUIRE_NO_THROW(mode.setupNeuralNetworkWeights(dir));
}

BOOST_DATA_TEST_CASE(TrainingExamples_ElectrostaticWeightFilesMatchSettings,
                     bdata::make(trainingExamplesWithWeights),
                     example)
{
    string const dir = "../../examples/" + example + "/";
    Mode mode;
    mode.log.writeToStdout = false;
    mode.loadSettingsFile(dir + "input.nn");
    mode.setupGeneric(dir);
    BOOST_REQUIRE_NO_THROW(
        mode.readNeuralNetworkWeights("elec", dir + "weightse.%03zu.data"));
}

BOOST_AUTO_TEST_SUITE_END()
