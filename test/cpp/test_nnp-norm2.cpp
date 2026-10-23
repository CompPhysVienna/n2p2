#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-norm2
#include "Example_nnp_norm2.h"
#include "nnp_test.h"

using namespace std;

BoostDataContainer<Example_nnp_norm2> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_norm2 const example)
{
    BOOST_REQUIRE(bfs::exists(example.logFile));
    for (auto f : example.createdFiles)
    {
        BOOST_REQUIRE(bfs::exists(f));
    }
    for (auto f : example.notCreatedFiles)
    {
        BOOST_REQUIRE(!bfs::exists(f));
    }

    return;
}
