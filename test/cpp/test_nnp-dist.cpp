#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-dist
#include "Example_nnp_dist.h"
#include "nnp_test.h"

#include <limits> // std::numeric_limits

using namespace std;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_nnp_dist> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_dist const example)
{
    BOOST_REQUIRE(bfs::exists("nnp-dist.log"));
    for (auto f : example.createdFiles)
    {
        BOOST_REQUIRE(bfs::exists(f));
    }

    return;
}
