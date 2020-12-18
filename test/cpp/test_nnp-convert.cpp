#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-convert
#include "Example_nnp_convert.h"
#include "nnp_test.h"

#include <limits> // std::numeric_limits

using namespace std;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_nnp_convert> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_convert const example)
{
    BOOST_REQUIRE(bfs::exists("nnp-convert.log"));
    for (auto f : example.createdFiles)
    {
        BOOST_REQUIRE(bfs::exists(f));
    }

    return;
}
