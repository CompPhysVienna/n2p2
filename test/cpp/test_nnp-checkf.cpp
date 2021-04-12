#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-checkf
#include "Example_nnp_checkf.h"
#include "nnp_test.h"

#include <limits> // std::numeric_limits

using namespace std;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_nnp_checkf> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_checkf const /*example*/)
{
    BOOST_REQUIRE(bfs::exists("nnp-checkf.log.0000"));
    BOOST_REQUIRE(bfs::exists("checkf-forces.out"));
    BOOST_REQUIRE(bfs::exists("checkf-summary.out"));

    return;
}
