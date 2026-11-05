#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Prediction
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"
#include "ExamplePrediction.h"
#include "tolerance.h"

#include "Atom.h"
#include "Structure.h"
#include "Prediction.h"
#include "utility.h"
#include <algorithm> // std::max
#include <cmath>   // std::fabs
#include <cstddef> // std::size_t
#include <fstream> // std::ifstream, std::ofstream
#include <limits> // std::numeric_limits
#include <string> // std::string, std::string::find_last_of
#include <vector> // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

Tolerance const tolerance =
    Tolerance(1000.0 * numeric_limits<double>::epsilon());

BoostDataContainer<ExamplePrediction> container;

BOOST_AUTO_TEST_SUITE(RegressionTests)

BOOST_DATA_TEST_CASE_F(FixtureRepairDir,
                       PredictStructure_CorrectEnergiesForces,
                       bdata::make(container.examples),
                       example)
{
    BOOST_REQUIRE(copy_directory_recursively(example.pathData, "test"));
    bfs::current_path("test");

    Prediction p;
    p.log.writeToStdout = false;
    p.setup();
    p.readStructureFromFile();
    p.predict();
    BOOST_TEST_INFO(example.name + " Potential energy");
    REQUIRE_CLOSE_TO_REFERENCE(example.energy, p.structure.energy, tolerance);

    for (size_t i = 0; i < p.structure.atoms.size(); ++i)
    {
        Vec3D const& f = p.structure.atoms.at(i).f;
        Vec3D const& fRef = example.forces.at(i);
        BOOST_TEST_INFO(example.name + " Atom " << i << " fx");
        REQUIRE_CLOSE_TO_REFERENCE(fRef[0], f[0], tolerance);
        BOOST_TEST_INFO(example.name + " Atom " << i << " fy");
        REQUIRE_CLOSE_TO_REFERENCE(fRef[1], f[1], tolerance);
        BOOST_TEST_INFO(example.name + " Atom " << i << " fz");
        REQUIRE_CLOSE_TO_REFERENCE(fRef[2], f[2], tolerance);
    }

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()

// A model trained with normalized units (nnp-norm) and its exact copy in the
// units of the training data must give the same results. The copy drops the
// normalization keywords, moves the mean energy per atom into the atomic
// energy offsets and divides the output layer of the short-range networks
// (linear activation) by conv_energy; charge widths, hardness and
// electronegativity networks are already stored in the units of the data.

// Write the copy of the model in directory "physical".
void writePhysicalCopy()
{
    vector<string> settings;
    double meanEnergy = 0.0;
    double convEnergy = 1.0;
    {
        ifstream in("physical/input.nn");
        string line;
        while (getline(in, line))
        {
            vector<string> t = split(reduce(line.substr(0, line.find('#'))));
            if (!t.empty() && t.at(0) == "mean_energy")
            {
                meanEnergy = stod(t.at(1));
                continue;
            }
            if (!t.empty() && t.at(0) == "conv_energy")
            {
                convEnergy = stod(t.at(1));
                continue;
            }
            if (!t.empty() && (t.at(0) == "conv_length" ||
                               t.at(0) == "conv_charge")) continue;
            settings.push_back(line);
            if (!t.empty() && t.at(0) == "elements")
            {
                for (size_t i = 1; i < t.size(); ++i)
                {
                    settings.push_back(strpr("atom_energy %s %24.16E",
                                             t.at(i).c_str(),
                                             meanEnergy));
                }
            }
        }
    }
    BOOST_REQUIRE(convEnergy != 1.0);
    {
        ofstream out("physical/input.nn");
        for (auto const& line : settings) out << line << "\n";
    }
    for (bfs::directory_iterator it("physical");
         it != bfs::directory_iterator(); ++it)
    {
        string const name = it->path().filename().string();
        if (name.find("weights.") != 0) continue;
        vector<vector<string>> rows;
        vector<string> lines;
        size_t outputLayer = 0;
        {
            ifstream in(it->path().string());
            string line;
            while (getline(in, line))
            {
                lines.push_back(line);
                vector<string> t = split(reduce(line));
                if (t.size() == 7 && t.at(1) == "a")
                {
                    outputLayer = max(outputLayer, (size_t)stoi(t.at(5)));
                }
            }
        }
        BOOST_REQUIRE(outputLayer > 0);
        ofstream out(it->path().string());
        for (auto const& line : lines)
        {
            vector<string> t = split(reduce(line));
            bool const output = line.find("#") != 0 && t.size() >= 4 &&
                ((t.at(1) == "a" && t.size() == 7 &&
                  (size_t)stoi(t.at(5)) == outputLayer) ||
                 (t.at(1) == "b" && (size_t)stoi(t.at(3)) == outputLayer));
            if (!output)
            {
                out << line << "\n";
                continue;
            }
            out << strpr("%24.16E", stod(t.at(0)) / convEnergy);
            for (size_t i = 1; i < t.size(); ++i) out << " " << t.at(i);
            out << "\n";
        }
    }
}

struct PredictionResult
{
    double         energy;
    vector<double> charges;
    vector<double> forces;
};

PredictionResult predictInDirectory(string const& directory)
{
    bfs::current_path(directory);
    Prediction p;
    p.log.writeToStdout = false;
    p.setup();
    p.readStructureFromFile();
    p.predict();
    bfs::current_path("..");
    PredictionResult r;
    r.energy = p.structure.energy;
    for (auto const& a : p.structure.atoms)
    {
        r.charges.push_back(a.charge);
        for (size_t c = 0; c < 3; ++c) r.forces.push_back(a.f[c]);
    }
    return r;
}

BOOST_AUTO_TEST_SUITE(NormalizationTests)

BOOST_FIXTURE_TEST_CASE(NormalizedModel_MatchesCopyInUnitsOfData,
                        FixtureRepairDir)
{
    string const example = "../../examples/nnp-predict/H2O_SPCFw_4G";
    BOOST_REQUIRE(copy_directory_recursively(example, "normalized"));
    BOOST_REQUIRE(copy_directory_recursively(example, "physical"));
    writePhysicalCopy();

    PredictionResult const n = predictInDirectory("normalized");
    PredictionResult const p = predictInDirectory("physical");

    Tolerance const t(1.0E-10 * fabs(p.energy));
    REQUIRE_CLOSE_TO_REFERENCE(p.energy, n.energy, t);
    BOOST_REQUIRE_EQUAL(p.charges.size(), n.charges.size());
    for (size_t i = 0; i < p.charges.size(); ++i)
    {
        BOOST_TEST_INFO("Atom " << i << " charge");
        REQUIRE_CLOSE_TO_REFERENCE(p.charges.at(i),
                                   n.charges.at(i),
                                   Tolerance(1.0E-10));
    }
    for (size_t i = 0; i < p.forces.size(); ++i)
    {
        BOOST_TEST_INFO("Force component " << i);
        REQUIRE_CLOSE_TO_REFERENCE(p.forces.at(i),
                                   n.forces.at(i),
                                   Tolerance(1.0E-10));
    }

    bfs::remove_all("normalized");
    bfs::remove_all("physical");
}

BOOST_FIXTURE_TEST_CASE(ModelInOtherUnits_MatchesNormalizedModel,
                        FixtureRepairDir)
{
    // The same model converted to eV and Angstrom without normalization
    // (four_pi_epsilon != 1, see
    // examples/interface-LAMMPS/H2O_SPCFw_4G_metal), with the structure
    // converted to Angstrom.
    double const cflength = 1.8897261328; // Angstrom -> Bohr
    double const cfenergy = 0.0367493254; // eV -> Hartree
    BOOST_REQUIRE(copy_directory_recursively(
        "../../examples/nnp-predict/H2O_SPCFw_4G", "reference"));
    BOOST_REQUIRE(copy_directory_recursively(
        "../../examples/interface-LAMMPS/H2O_SPCFw_4G_metal/hdnnp-data",
        "metal"));
    {
        ifstream in("reference/input.data");
        ofstream out("metal/input.data");
        string line;
        while (getline(in, line))
        {
            vector<string> t = split(reduce(line));
            if (!t.empty() && (t.at(0) == "lattice" || t.at(0) == "atom"))
            {
                for (size_t i = 1; i < 4; ++i)
                {
                    t.at(i) = strpr("%.16E", stod(t.at(i)) / cflength);
                }
                line = "";
                for (auto const& w : t) line += w + " ";
            }
            out << line << "\n";
        }
    }

    PredictionResult const n = predictInDirectory("reference");
    PredictionResult const m = predictInDirectory("metal");

    // Remaining differences: the Ewald sums are truncated in different
    // units.
    REQUIRE_CLOSE_TO_REFERENCE(n.energy / cfenergy,
                               m.energy,
                               Tolerance(1.0E-6 * fabs(m.energy)));
    BOOST_REQUIRE_EQUAL(n.charges.size(), m.charges.size());
    for (size_t i = 0; i < n.charges.size(); ++i)
    {
        BOOST_TEST_INFO("Atom " << i << " charge");
        REQUIRE_CLOSE_TO_REFERENCE(n.charges.at(i),
                                   m.charges.at(i),
                                   Tolerance(1.0E-6));
    }
    for (size_t i = 0; i < n.forces.size(); ++i)
    {
        BOOST_TEST_INFO("Force component " << i);
        REQUIRE_CLOSE_TO_REFERENCE(n.forces.at(i) * cflength / cfenergy,
                                   m.forces.at(i),
                                   Tolerance(1.0E-5));
    }

    bfs::remove_all("reference");
    bfs::remove_all("metal");
}

BOOST_AUTO_TEST_SUITE_END()

// Committee of 2G-HDNNPs (examples/nnp-predict/H2O_RPBE-D3_Committee, 8
// members in prediction mode): compared with each member evaluated alone,
// as an ordinary 2G-HDNNP.

// Copy of the settings file with the committee keywords removed
// (mode == "") or with committee_mode set to the given mode.
void writeCommitteeSettings(string const& fileName, string const& mode)
{
    vector<string> lines;
    {
        ifstream in("input.nn");
        string line;
        while (getline(in, line))
        {
            vector<string> t = split(reduce(line.substr(0, line.find('#'))));
            if (!t.empty() && t.at(0).find("committee_") == 0)
            {
                if (mode == "") continue;
                if (t.at(0) == "committee_mode")
                {
                    line = "committee_mode " + mode;
                }
            }
            lines.push_back(line);
        }
    }
    ofstream out(fileName);
    for (auto const& line : lines) out << line << "\n";
}

struct CommitteeMemberResult
{
    double        energy;
    vector<Vec3D> forces;
};

CommitteeMemberResult predictCommitteeMember(size_t const c)
{
    Prediction p;
    p.log.writeToStdout = false;
    p.fileNameSettings = "input-single.nn";
    if (c > 0)
    {
        p.formatWeightsFilesShort = strpr("nnp-data-%zu/weights.%%03zu.data",
                                          c);
    }
    p.setup();
    p.readStructureFromFile();
    p.predict();
    CommitteeMemberResult r;
    r.energy = p.structure.energy;
    for (auto const& a : p.structure.atoms) r.forces.push_back(a.f);
    return r;
}

BOOST_AUTO_TEST_SUITE(CommitteeTests)

BOOST_FIXTURE_TEST_CASE(CommitteePrediction_MatchesMembersEvaluatedAlone,
                        FixtureRepairDir)
{
    string const example = "../../examples/nnp-predict/H2O_RPBE-D3_Committee";
    BOOST_REQUIRE(copy_directory_recursively(example, "test"));
    bfs::current_path("test");
    writeCommitteeSettings("input-single.nn", "");
    writeCommitteeSettings("input-validation.nn", "validation");

    size_t const n = 8;
    vector<CommitteeMemberResult> members;
    for (size_t c = 0; c < n; ++c) members.push_back(predictCommitteeMember(c));

    Prediction p;
    p.log.writeToStdout = false;
    p.setup();
    p.readStructureFromFile();
    p.predict();
    Structure const& s = p.structure;
    BOOST_REQUIRE_EQUAL(s.energyCom.size(), n);

    Tolerance const te(0.0, 1.0E-13);
    Tolerance const tf(1.0E-12, 1.0E-12);
    double energy = 0.0;
    for (size_t c = 0; c < n; ++c)
    {
        BOOST_TEST_INFO("Member " << c << " energy");
        REQUIRE_CLOSE_TO_REFERENCE(members.at(c).energy, s.energyCom.at(c), te);
        energy += members.at(c).energy / n;
    }
    BOOST_TEST_INFO("Committee energy");
    REQUIRE_CLOSE_TO_REFERENCE(energy, s.energy, te);
    double sum = 0.0;
    for (size_t c = 0; c < n; ++c)
    {
        sum += (energy - members.at(c).energy) * (energy - members.at(c).energy);
    }
    BOOST_TEST_INFO("Committee energy disagreement");
    REQUIRE_CLOSE_TO_REFERENCE(sqrt(sum / n),
                               s.committeeDisagreement,
                               Tolerance(1.0E-10 * fabs(energy)));

    for (size_t i = 0; i < s.atoms.size(); ++i)
    {
        Atom const& a = s.atoms.at(i);
        BOOST_REQUIRE_EQUAL(a.fCom.size(), n);
        for (size_t k = 0; k < 3; ++k)
        {
            double f = 0.0;
            for (size_t c = 0; c < n; ++c)
            {
                BOOST_TEST_INFO("Atom " << i << " member " << c
                                << " force component " << k);
                REQUIRE_CLOSE_TO_REFERENCE(members.at(c).forces.at(i)[k],
                                           a.fCom.at(c)[k],
                                           tf);
                f += members.at(c).forces.at(i)[k] / n;
            }
            BOOST_TEST_INFO("Atom " << i << " force component " << k);
            REQUIRE_CLOSE_TO_REFERENCE(f, a.f[k], tf);
            double d = 0.0;
            for (size_t c = 0; c < n; ++c)
            {
                double const diff = f - members.at(c).forces.at(i)[k];
                d += diff * diff;
            }
            BOOST_TEST_INFO("Atom " << i << " force disagreement " << k);
            REQUIRE_CLOSE_TO_REFERENCE(sqrt(d / n),
                                       a.committeeDisagreement[k],
                                       tf);
        }
    }

    // Validation mode: member 0 is used, the others give the disagreement.
    Prediction v;
    v.log.writeToStdout = false;
    v.fileNameSettings = "input-validation.nn";
    v.setup();
    v.readStructureFromFile();
    v.predict();
    BOOST_TEST_INFO("Validation energy");
    REQUIRE_CLOSE_TO_REFERENCE(members.at(0).energy, v.structure.energy, te);
    for (size_t i = 0; i < v.structure.atoms.size(); ++i)
    {
        for (size_t k = 0; k < 3; ++k)
        {
            BOOST_TEST_INFO("Validation atom " << i << " force " << k);
            REQUIRE_CLOSE_TO_REFERENCE(members.at(0).forces.at(i)[k],
                                       v.structure.atoms.at(i).f[k],
                                       tf);
        }
    }

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(CommitteeSettings_MissingDataKeyword_Throws,
                        FixtureRepairDir)
{
    string const example = "../../examples/nnp-predict/H2O_RPBE-D3_Committee";
    BOOST_REQUIRE(copy_directory_recursively(example, "test"));
    bfs::current_path("test");
    {
        ifstream in("input.nn");
        ofstream out("input-nodata.nn");
        string line;
        while (getline(in, line))
        {
            if (line.find("committee_data") == 0) continue;
            out << line << "\n";
        }
    }
    Prediction p;
    p.log.writeToStdout = false;
    p.fileNameSettings = "input-nodata.nn";
    BOOST_REQUIRE_THROW(p.setup(), runtime_error);

    bfs::current_path("..");
    bfs::remove_all("test");
}

// Members with their own input.nn or scaling.data must agree with the main
// settings in normalization, atomic energy offsets and scaling.

// Copy of the committee example in "test" (current directory afterwards).
void enterCommitteeCopy()
{
    string const example = "../../examples/nnp-predict/H2O_RPBE-D3_Committee";
    BOOST_REQUIRE(copy_directory_recursively(example, "test"));
    bfs::current_path("test");
}

// Replace the first line of a file starting with the given keyword (or
// append the line if keyword is not found and append is true).
void replaceLine(string const& fileName,
                 string const& keyword,
                 string const& newLine,
                 bool const    append = false)
{
    vector<string> lines;
    bool found = false;
    {
        ifstream in(fileName);
        string line;
        while (getline(in, line))
        {
            if (!found && line.find(keyword) == 0)
            {
                line = newLine;
                found = true;
            }
            lines.push_back(line);
        }
    }
    if (!found && append) lines.push_back(newLine);
    BOOST_REQUIRE(found || append);
    ofstream out(fileName);
    for (auto const& line : lines) out << line << "\n";
}

void requireCommitteeSetupThrows()
{
    Prediction p;
    p.log.writeToStdout = false;
    BOOST_REQUIRE_THROW(p.setup(), runtime_error);
    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(CommitteeMembers_DifferentNormalization_Throws,
                        FixtureRepairDir)
{
    enterCommitteeCopy();
    replaceLine("nnp-data-3/input.nn",
                "conv_energy",
                "conv_energy   2.5E+02");
    requireCommitteeSetupThrows();
}

BOOST_FIXTURE_TEST_CASE(CommitteeMembers_DifferentEnergyOffsets_Throws,
                        FixtureRepairDir)
{
    enterCommitteeCopy();
    replaceLine("nnp-data-5/input.nn",
                "atom_energy",
                "atom_energy H -0.45890771",
                true);
    requireCommitteeSetupThrows();
}

BOOST_FIXTURE_TEST_CASE(CommitteeMembers_DifferentScaling_Throws,
                        FixtureRepairDir)
{
    enterCommitteeCopy();
    // Change the maximum of the first symmetry function of member 2.
    vector<string> lines;
    {
        ifstream in("nnp-data-2/scaling.data");
        string line;
        bool changed = false;
        while (getline(in, line))
        {
            vector<string> t = split(reduce(line));
            if (!changed && !t.empty() && t.at(0).at(0) != '#')
            {
                t.at(3) = strpr("%24.16E", 1.01 * stod(t.at(3)) + 1.0E-3);
                line = "";
                for (auto const& c : t) line += c + " ";
                changed = true;
            }
            lines.push_back(line);
        }
        BOOST_REQUIRE(changed);
    }
    {
        ofstream out("nnp-data-2/scaling.data");
        for (auto const& line : lines) out << line << "\n";
    }
    requireCommitteeSetupThrows();
}

BOOST_FIXTURE_TEST_CASE(CommitteeMembers_OnlyWeights_Accepted,
                        FixtureRepairDir)
{
    enterCommitteeCopy();
    for (size_t c = 1; c < 8; ++c)
    {
        bfs::remove(strpr("nnp-data-%zu/input.nn", c));
        bfs::remove(strpr("nnp-data-%zu/scaling.data", c));
    }
    Prediction p;
    p.log.writeToStdout = false;
    BOOST_REQUIRE_NO_THROW(p.setup());
    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()
