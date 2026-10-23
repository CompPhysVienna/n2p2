#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-dataset
#include "Example_nnp_dataset.h"
#include "nnp_test.h"

#include "Prediction.h"
#include "utility.h"
#include <cmath>   // std::fabs
#include <cstddef> // std::size_t
#include <cstdlib> // std::system
#include <fstream> // std::ifstream
#include <map>     // std::map
#include <string>  // std::string
#include <vector>  // std::vector

using namespace std;
using namespace nnp;

BoostDataContainer<Example_nnp_dataset> container;

// Lines of a file without the comment (header) lines.
vector<string> readDataLines(string const& fileName)
{
    vector<string> lines;
    ifstream file(fileName);
    BOOST_REQUIRE_MESSAGE(file.is_open(), "cannot open " + fileName);
    string line;
    while (getline(file, line))
    {
        if (!line.empty() && line.at(0) != '#') lines.push_back(line);
    }
    return lines;
}

// Runs a command in the current directory, discarding its output.
int run(string const& command)
{
    return std::system((command + " > /dev/null").c_str());
}

// Copies an example directory to dir (removing an old copy first).
void copyExample(Example_nnp_dataset const& example, string const& dir)
{
    if (bfs::exists(dir)) bfs::remove_all(dir);
    BOOST_REQUIRE(copy_directory_recursively(example.pathData, dir));
}

// charges.comp prints charges with 9 significant digits.
double const accuracyCharges = 1.0E-8;

// 4G-HDNNP: one charges.comp row per atom, predicted charges of each
// structure add up to its total charge (zero here), and the charges of the
// first structure agree with an in-process evaluation by Prediction.
void checkCharges4G()
{
    vector<string> const lines = readDataLines("charges.comp");
    // 20 structures of 16 water molecules each (grep -c "^atom" input.data).
    BOOST_REQUIRE_EQUAL(lines.size(), size_t(960));

    map<size_t, double> sum;
    map<size_t, vector<double>> charges;
    for (auto const& line : lines)
    {
        vector<string> const c = split(reduce(line));
        size_t const s = stoul(c.at(0));
        sum[s] += stod(c.at(3));
        charges[s].push_back(stod(c.at(3)));
    }
    BOOST_REQUIRE_EQUAL(sum.size(), size_t(20));
    for (auto const& s : sum)
    {
        BOOST_TEST_INFO("structure " << s.first);
        BOOST_REQUIRE_SMALL(s.second, 48 * accuracyCharges);
    }

    Prediction p;
    p.log.writeToStdout = false;
    p.setup();
    p.readStructureFromFile("input.data");
    p.predict();
    vector<double> const& first = charges.begin()->second;
    BOOST_REQUIRE_EQUAL(first.size(), p.structure.atoms.size());
    for (size_t i = 0; i < first.size(); ++i)
    {
        BOOST_TEST_INFO("atom " << i);
        BOOST_REQUIRE_SMALL(first.at(i) - p.structure.atoms.at(i).charge,
                            accuracyCharges);
    }
}

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_dataset const example)
{
    BOOST_REQUIRE(bfs::exists("nnp-dataset.log.0000"));
    BOOST_REQUIRE(bfs::exists("energy.comp"));
    BOOST_REQUIRE(bfs::exists("forces.comp"));
    BOOST_REQUIRE(bfs::exists("output.data"));
    if (example.name == "H2O_RPBE-D3_4G") checkCharges4G();
    else BOOST_REQUIRE(!bfs::exists("charges.comp"));

    return;
}

BOOST_AUTO_TEST_SUITE(OptionalArguments)

// <data_file>: reading the same structures from another file name gives
// the same results as the default input.data.
BOOST_AUTO_TEST_CASE(DataFile_SameResultsAsDefaultInputData)
{
    Example_nnp_dataset const example("H2O_RPBE-D3");

    copyExample(example, "test_default");
    bfs::current_path("test_default");
    BOOST_REQUIRE_EQUAL(run(example.command + " 0"), 0);
    bfs::current_path("..");

    copyExample(example, "test_datafile");
    bfs::current_path("test_datafile");
    bfs::rename("input.data", "other.data");
    BOOST_REQUIRE_EQUAL(run(example.command + " 0 other.data"), 0);
    bfs::current_path("..");

    for (string const f : {"energy.comp", "forces.comp", "output.data"})
    {
        BOOST_TEST_INFO(f);
        BOOST_REQUIRE(readDataLines("test_default/" + f)
                      == readDataLines("test_datafile/" + f));
    }
    bfs::remove_all("test_default");
    bfs::remove_all("test_datafile");
}

// <charges_only>: without any short-range weights files, the 4G charges
// are the same as in a full run, and no energy, forces or output.data are
// written.
BOOST_AUTO_TEST_CASE(ChargesOnly_SameChargesWithoutShortRangeWeights)
{
    Example_nnp_dataset const example("H2O_RPBE-D3_4G");

    copyExample(example, "test_full");
    bfs::current_path("test_full");
    BOOST_REQUIRE_EQUAL(run(example.command + " 0"), 0);
    bfs::current_path("..");

    copyExample(example, "test_charges");
    bfs::current_path("test_charges");
    bfs::remove("weights.001.data");
    bfs::remove("weights.008.data");
    BOOST_REQUIRE_EQUAL(run(example.command + " 0 input.data 1"), 0);
    BOOST_REQUIRE(!bfs::exists("energy.comp"));
    BOOST_REQUIRE(!bfs::exists("forces.comp"));
    BOOST_REQUIRE(!bfs::exists("output.data"));
    bfs::current_path("..");

    BOOST_REQUIRE(readDataLines("test_full/charges.comp")
                  == readDataLines("test_charges/charges.comp"));
    bfs::remove_all("test_full");
    bfs::remove_all("test_charges");
}

BOOST_AUTO_TEST_SUITE_END()
