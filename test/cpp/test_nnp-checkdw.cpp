#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-checkdw
#include "Example_nnp_checkdw.h"
#include "nnp_test.h"

#include <limits> // std::numeric_limits

using namespace std;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_nnp_checkdw> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_checkdw const /*example*/)
{
    BOOST_REQUIRE(bfs::exists("nnp-checkdw.log.0000"));
    BOOST_REQUIRE(bfs::exists("checkdw-summary.out"));
    BOOST_REQUIRE(bfs::exists("checkdw-weights.energy.out"));
    BOOST_REQUIRE(bfs::exists("checkdw-weights.force.out"));

    return;
}
