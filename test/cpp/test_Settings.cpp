#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Settings
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include <cstdlib>
#include <fstream>
#include "Settings.h"

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

BOOST_AUTO_TEST_SUITE(RegressionTests)

BOOST_FIXTURE_TEST_CASE(ReadRecommendedSettings_NoCriticalErrors,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::copy_file("../../examples/input.nn.recommended", "test/input.nn");
    bfs::current_path("test");

    Settings s;
    size_t numCriticalProblems = s.loadFile();
    BOOST_REQUIRE_EQUAL(numCriticalProblems, 0);

    size_t numProblems = 0;
    for (auto line : s.info())
    {
        if (line.find("WARNING") != line.npos) numProblems++;
    }
    BOOST_REQUIRE_EQUAL(numProblems, 0);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(UnknownKeyword_NoCriticalErrors,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::copy_file("../../examples/input.nn.recommended", "test/input.nn");
    bfs::current_path("test");

    ofstream f;
    f.open("input.nn", ios_base::app);
    f << "abcdefghijk";
    f.flush();
    f.close();

    Settings s;
    size_t numCriticalProblems = s.loadFile();
    BOOST_REQUIRE_EQUAL(numCriticalProblems, 0);

    size_t numProblems = 0;
    for (auto line : s.info())
    {
        if (line.find("WARNING") != line.npos) numProblems++;
    }
    BOOST_REQUIRE_EQUAL(numProblems, 2);

    BOOST_TEST_INFO("Forbidden to parse file for unknown keywords.\n");
    BOOST_REQUIRE_THROW(s.keywordExists("abcdefghijk"), runtime_error);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(DuplicatedKeyword_CriticalError,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::copy_file("../../examples/input.nn.recommended", "test/input.nn");
    bfs::current_path("test");

    ofstream f;
    f.open("input.nn", ios_base::app);
    f << "use_short_forces";
    f.flush();
    f.close();

    Settings s;
    size_t numCriticalProblems = s.loadFile();
    BOOST_REQUIRE_EQUAL(numCriticalProblems, 1);

    size_t numProblems = 0;
    for (auto line : s.info())
    {
        if (line.find("WARNING") != line.npos) numProblems++;
    }
    BOOST_REQUIRE_EQUAL(numProblems, 2);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(DuplicatedAlternativeKeyword_CriticalError,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::copy_file("../../examples/input.nn.recommended", "test/input.nn");
    bfs::current_path("test");

    ofstream f;
    f.open("input.nn", ios_base::app);
    f << "rmse_threshold_energy 0.5";
    f.flush();
    f.close();

    Settings s;
    size_t numCriticalProblems = s.loadFile();
    BOOST_REQUIRE_EQUAL(numCriticalProblems, 2);

    size_t numProblems = 0;
    for (auto line : s.info())
    {
        if (line.find("WARNING") != line.npos) numProblems++;
    }
    BOOST_REQUIRE_EQUAL(numProblems, 3);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_FIXTURE_TEST_CASE(UseAlternativeKeyword_CorrectValue,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::copy_file("../../examples/input.nn.recommended", "test/input.nn");
    bfs::current_path("test");

    Settings s;
    s.loadFile();

    double value = atof(s["short_force_error_threshold"].c_str());
    BOOST_REQUIRE_EQUAL(value, 1.0);
    value = atof(s["rmse_threshold_force"].c_str());
    BOOST_REQUIRE_EQUAL(value, 1.0);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()
