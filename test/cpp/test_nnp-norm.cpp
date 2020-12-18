#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-norm
#include "Example_nnp_norm.h"
#include "nnp_test.h"

#include <limits> // std::numeric_limits

using namespace std;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_nnp_norm> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_norm const example)
{
    BOOST_REQUIRE(bfs::exists("nnp-norm.log.0000"));
    for (auto f : example.createdFiles)
    {
        BOOST_REQUIRE(bfs::exists(f));
    }

    return;
}
