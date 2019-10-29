#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE CutoffFunction
#include <boost/test/unit_test.hpp>

#include "CutoffFunction.h"
#include <limits> // std::numeric_limits

using namespace std;
using namespace nnp;

double accuracy = 10.0 * numeric_limits<double>::epsilon(); 

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(CheckCutoffTypeNumbers_CorrectNumbering)
{
    BOOST_REQUIRE_EQUAL(CutoffFunction::CT_HARD , 0);
    BOOST_REQUIRE_EQUAL(CutoffFunction::CT_COS  , 1);
    BOOST_REQUIRE_EQUAL(CutoffFunction::CT_TANHU, 2);
    BOOST_REQUIRE_EQUAL(CutoffFunction::CT_TANH , 3);
    BOOST_REQUIRE_EQUAL(CutoffFunction::CT_EXP  , 4);
    BOOST_REQUIRE_EQUAL(CutoffFunction::CT_POLY1, 5);
    BOOST_REQUIRE_EQUAL(CutoffFunction::CT_POLY2, 6);
    BOOST_REQUIRE_EQUAL(CutoffFunction::CT_POLY3, 7);
    BOOST_REQUIRE_EQUAL(CutoffFunction::CT_POLY4, 8);
}

BOOST_AUTO_TEST_CASE(CheckDefaultInitialization_AllZeroOutput)
{
    CutoffFunction fc;

    BOOST_REQUIRE_EQUAL(fc.getCutoffType(), CutoffFunction::CT_HARD);
    BOOST_REQUIRE_SMALL(fc.getCutoffRadius(), accuracy);
    BOOST_REQUIRE_SMALL(fc.getCutoffParameter(), accuracy);

}

BOOST_AUTO_TEST_SUITE_END()
