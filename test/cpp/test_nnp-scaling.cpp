#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-scaling
#include "Example_nnp_scaling.h"
#include "nnp_test.h"

#include <limits> // std::numeric_limits

using namespace std;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_nnp_scaling> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_scaling const /*example*/)
{
    BOOST_REQUIRE(bfs::exists("nnp-scaling.log.0000"));
    BOOST_REQUIRE(bfs::exists("scaling.data"));
    BOOST_REQUIRE(bfs::exists("neighbors.histo"));

    return;
}
