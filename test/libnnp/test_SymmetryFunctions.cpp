#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SymmetryFunctions
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "SymmetryFunction.h"
#include <limits> // std::numeric_limits
#include <vector> // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 10.0 * numeric_limits<double>::epsilon(); 

BOOST_AUTO_TEST_SUITE(IntegrationTests)


BOOST_AUTO_TEST_SUITE_END()
