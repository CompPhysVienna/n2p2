#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-sfclust
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/process.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "Atom.h"
#include "Structure.h"
#include "Prediction.h"
#include <cstddef> // std::size_t
#include <limits> // std::numeric_limits
#include <string> // std::string
#include <vector> // std::vector

using namespace std;
using namespace nnp;
using namespace boost::filesystem;

namespace bdata = boost::unit_test::data;
namespace bprocess = boost::process;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

vector<string> sampleDirectories = {"../../examples/nnp-sfclust/H2O_RPBE-D3",
                                    "../../examples/nnp-sfclust/Cu2S_PBE"};

vector<string> args = {"100 6 2 4 4 ",
                       "100 12 6 3 6 "};

BOOST_AUTO_TEST_SUITE(RegressionTests)

BOOST_DATA_TEST_CASE(Execute_CorrectOutputFiles,
                     bdata::make(sampleDirectories) ^ bdata::make(args),
                     sampleDirectory,
                     args)
{
    BOOST_REQUIRE(copy_directory_recursively(sampleDirectory, "test"));
    current_path("test");

    bprocess::ipstream outStream;
    int result = bprocess::system("../../../bin/nnp-sfclust " + args,
                                  bprocess::std_out > outStream);

    BOOST_REQUIRE_EQUAL(result, 0);

    current_path("..");
    remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()
