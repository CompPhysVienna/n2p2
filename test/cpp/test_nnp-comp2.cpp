#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-comp2
#include "Example_nnp_comp2.h"
#include "nnp_test.h"

using namespace std;

BoostDataContainer<Example_nnp_comp2> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_comp2 const /*example*/)
{
    // Per-NNP logs: "nnp-comp2.log.<nnpId>.<rankInNnpComm>".
    BOOST_REQUIRE(bfs::exists("nnp-comp2.log.1.0000"));
    BOOST_REQUIRE(bfs::exists("nnp-comp2.log.2.0000"));
    // Combined comparison output.
    BOOST_REQUIRE(bfs::exists("comp.data"));
    BOOST_REQUIRE(bfs::exists("diff.out"));

    return;
}
