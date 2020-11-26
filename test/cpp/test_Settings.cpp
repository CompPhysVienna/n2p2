#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Settings
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include <cstdlib> 
#include "Settings.h"

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

BOOST_AUTO_TEST_SUITE(RegressionTests)

BOOST_FIXTURE_TEST_CASE(ReadRecommendedSettings_NoErrors,
                        FixtureRepairDir)
{
    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::copy_file("../../examples/input.nn.recommended", "test/input.nn");
    bfs::current_path("test");

    Settings s;
    size_t numCriticalProblems = s.loadFile();

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()
