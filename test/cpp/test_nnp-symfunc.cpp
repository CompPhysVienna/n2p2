#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-symfunc
#include "Example_nnp_symfunc.h"
#include "nnp_test.h"

#include <limits> // std::numeric_limits

using namespace std;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_nnp_symfunc> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_symfunc const example)
{
    BOOST_REQUIRE(bfs::exists("nnp-symfunc.log"));
    for (auto f : example.createdFiles)
    {
        BOOST_REQUIRE(bfs::exists(f));
    }

    return;
}
