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

BOOST_AUTO_TEST_SUITE_END()
