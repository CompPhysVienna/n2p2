#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE CutoffFunction
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "CutoffFunction.h"
#include <limits> // std::numeric_limits
#include <vector> // std::vector
#include <stdexcept> // std::invalid_argument

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 10.0 * numeric_limits<double>::epsilon(); 

double const cutoffRadius = 3.0;
double const cutoffParameter = 0.2;
double const testRadius = 0.8 * cutoffRadius;
double const innerTestRadius = 0.1 * cutoffRadius;
vector<CutoffFunction::CutoffType> const types = {CutoffFunction::CT_HARD ,
                                                  CutoffFunction::CT_COS  ,
                                                  CutoffFunction::CT_TANHU,
                                                  CutoffFunction::CT_TANH ,
                                                  CutoffFunction::CT_EXP  ,
                                                  CutoffFunction::CT_POLY1,
                                                  CutoffFunction::CT_POLY2,
                                                  CutoffFunction::CT_POLY3,
                                                  CutoffFunction::CT_POLY4};
vector<double> const f = {1.0000000000000000E+00,
                          1.4644660940672610E-01,
                          7.6891537120697701E-03,
                          1.7406350897769030E-02,
                          2.7645304662956416E-01,
                          1.5624999999999989E-01,
                          1.0351562500000003E-01,
                          7.0556640624999140E-02,
                          4.8927307128909442E-02};
vector<double> const df = { 0.0000000000000000E+00,
                           -4.6280030605816297E-01,
                           -3.7439367857705103E-02,
                           -8.4753511078717925E-02,
                           -9.0270382572918983E-01,
                           -4.6874999999999989E-01,
                           -4.3945312499999917E-01,
                           -3.8452148437500033E-01,
                           -3.2444000244141674E-01};

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

BOOST_AUTO_TEST_CASE(CheckUnknownCutoffType_ThrowsError)
{
    CutoffFunction fc;

    BOOST_REQUIRE_THROW(fc.setCutoffType(
                            static_cast<CutoffFunction::CutoffType>(
                                types.size())),
                        invalid_argument);
}

BOOST_AUTO_TEST_CASE(CheckDefaultInitialization_AllZeroOutput)
{
    CutoffFunction fc;

    BOOST_REQUIRE_EQUAL(fc.getCutoffType(), CutoffFunction::CT_HARD);
    BOOST_REQUIRE_SMALL(fc.getCutoffRadius(), accuracy);
    BOOST_REQUIRE_SMALL(fc.getCutoffParameter(), accuracy);
}

BOOST_AUTO_TEST_CASE(CheckInvalidCutoffParameter_ThrowsError)
{
    CutoffFunction fc;
    fc.setCutoffType(CutoffFunction::CT_COS);

    BOOST_REQUIRE_THROW(fc.setCutoffParameter(-0.001), invalid_argument);
    BOOST_REQUIRE_THROW(fc.setCutoffParameter(1.0), invalid_argument);
    BOOST_REQUIRE_THROW(fc.setCutoffParameter(1.1), invalid_argument);
}

BOOST_DATA_TEST_CASE(CalculateFunctions_CorrectResults,
                     bdata::make(types) ^ bdata::make(f),
                     type,
                     f)
{
    CutoffFunction fc;
    fc.setCutoffType(type);
    fc.setCutoffRadius(cutoffRadius);
    fc.setCutoffParameter(cutoffParameter);
    BOOST_REQUIRE_SMALL(fc.f(testRadius) - f, accuracy);
}

BOOST_DATA_TEST_CASE(CalculateDerivatives_CorrectResults,
                     bdata::make(types) ^ bdata::make(df),
                     type,
                     df)
{
    CutoffFunction fc;
    fc.setCutoffType(type);
    fc.setCutoffRadius(cutoffRadius);
    fc.setCutoffParameter(cutoffParameter);
    BOOST_REQUIRE_SMALL(fc.df(testRadius) - df, accuracy);
}

BOOST_DATA_TEST_CASE(CalculateFunctionsAndDerivatives_CorrectResults,
                     bdata::make(types) ^ bdata::make(f) ^ bdata::make(df),
                     type,
                     f,
                     df)
{
    CutoffFunction fc;
    fc.setCutoffType(type);
    fc.setCutoffRadius(cutoffRadius);
    fc.setCutoffParameter(cutoffParameter);
    double resf;
    double resdf;
    fc.fdf(testRadius, resf, resdf);
    BOOST_REQUIRE_SMALL(resf - f, accuracy);
    BOOST_REQUIRE_SMALL(resdf - df, accuracy);
}

BOOST_DATA_TEST_CASE(CalculateFunctionsOutOfCutoff_ZeroResult,
                     bdata::make(types),
                     type)
{
    CutoffFunction fc;
    fc.setCutoffType(type);
    fc.setCutoffRadius(cutoffRadius);
    fc.setCutoffParameter(cutoffParameter);
    BOOST_REQUIRE_SMALL(fc.f(1.5 * cutoffRadius), accuracy);
}

BOOST_DATA_TEST_CASE(CalculateDerivativesOutOfCutoff_ZeroResult,
                     bdata::make(types),
                     type)
{
    CutoffFunction fc;
    fc.setCutoffType(type);
    fc.setCutoffRadius(cutoffRadius);
    fc.setCutoffParameter(cutoffParameter);
    BOOST_REQUIRE_SMALL(fc.df(1.5 * cutoffRadius), accuracy);
}

BOOST_DATA_TEST_CASE(CalculateFunctionsAndDerivativesOutOfCutoff_ZeroResult,
                     bdata::make(types),
                     type)
{
    CutoffFunction fc;
    fc.setCutoffType(type);
    fc.setCutoffRadius(cutoffRadius);
    fc.setCutoffParameter(cutoffParameter);
    double resf;
    double resdf;
    fc.fdf(1.5 * cutoffRadius, resf, resdf);
    BOOST_REQUIRE_SMALL(resf, accuracy);
    BOOST_REQUIRE_SMALL(resdf, accuracy);
}

BOOST_DATA_TEST_CASE(CalculateFunctionsInsideInnerCutoff_OneResult,
                     bdata::make(types),
                     type)
{
    CutoffFunction fc;
    fc.setCutoffType(type);
    fc.setCutoffRadius(cutoffRadius);
    fc.setCutoffParameter(cutoffParameter);
    double resf = fc.f(innerTestRadius);
    if (type == CutoffFunction::CT_TANHU)
    {
        BOOST_REQUIRE_SMALL(resf - 3.6752000144553820E-01, accuracy);
    }
    else if (type == CutoffFunction::CT_TANH)
    {
        BOOST_REQUIRE_SMALL(resf - 8.3197479809356334E-01, accuracy);
    }
    else
    {
        BOOST_REQUIRE_SMALL(resf - 1.0, accuracy);
    }
}

BOOST_DATA_TEST_CASE(CalculateDerivativesInsideInnerCutoff_ZeroResult,
                     bdata::make(types),
                     type)
{
    CutoffFunction fc;
    fc.setCutoffType(type);
    fc.setCutoffRadius(cutoffRadius);
    fc.setCutoffParameter(cutoffParameter);
    double resdf = fc.df(innerTestRadius);
    if (type == CutoffFunction::CT_TANHU)
    {
        BOOST_REQUIRE_SMALL(resdf - -2.4982884456067705E-01, accuracy);
    }
    else if (type == CutoffFunction::CT_TANH)
    {
        BOOST_REQUIRE_SMALL(resdf - -5.6555099503099682E-01, accuracy);
    }
    else
    {
        BOOST_REQUIRE_SMALL(resdf, accuracy);
    }
}

BOOST_DATA_TEST_CASE(
              CalculateFunctionsAndDerivativesInsideInnerCutoff_OneZeroResults,
              bdata::make(types),
              type)
{
    CutoffFunction fc;
    fc.setCutoffType(type);
    fc.setCutoffRadius(cutoffRadius);
    fc.setCutoffParameter(cutoffParameter);
    double resf;
    double resdf;
    fc.fdf(innerTestRadius, resf, resdf);
    if (type == CutoffFunction::CT_TANHU)
    {
        BOOST_REQUIRE_SMALL(resf - 3.6752000144553820E-01, accuracy);
        BOOST_REQUIRE_SMALL(resdf - -2.4982884456067705E-01, accuracy);
    }
    else if (type == CutoffFunction::CT_TANH)
    {
        BOOST_REQUIRE_SMALL(resf - 8.3197479809356334E-01, accuracy);
        BOOST_REQUIRE_SMALL(resdf - -5.6555099503099682E-01, accuracy);
    }
    else
    {
        BOOST_REQUIRE_SMALL(resf - 1.0, accuracy);
        BOOST_REQUIRE_SMALL(resdf, accuracy);
    }
}

BOOST_AUTO_TEST_SUITE_END()
