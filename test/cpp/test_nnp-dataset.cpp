#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-dataset
#include "Example_nnp_dataset.h"
#include "nnp_test.h"

#include <limits> // std::numeric_limits

using namespace std;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_nnp_dataset> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_dataset const /*example*/)
{
    BOOST_REQUIRE(bfs::exists("nnp-dataset.log.0000"));
    BOOST_REQUIRE(bfs::exists("energy.comp"));
    BOOST_REQUIRE(bfs::exists("forces.comp"));
    BOOST_REQUIRE(bfs::exists("output.data"));

    return;
}
