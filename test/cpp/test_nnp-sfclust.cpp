#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-sfclust
#include "Example_nnp_sfclust.h"
#include "nnp_test.h"

#include <limits> // std::numeric_limits

using namespace std;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_nnp_sfclust> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_sfclust const example)
{

    return;
}
