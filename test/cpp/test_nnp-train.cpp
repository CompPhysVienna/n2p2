#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nnp-train
#include "Example_nnp_train.h"
#include "nnp_test.h"

#include <limits> // std::numeric_limits

using namespace std;

double const accuracy = 10.0 * numeric_limits<double>::epsilon();

BoostDataContainer<Example_nnp_train> container;

NNP_TOOL_TEST_CASE()

void nnpToolTestBody(Example_nnp_train const example)
{

    return;
}
