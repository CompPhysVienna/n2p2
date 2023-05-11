#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ScreeningFunction
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include "ExampleScreeningFunction.h"

#include "ScreeningFunction.h"
#include "utility.h"
//#include <iostream>
//#include <fstream>
#include <limits> // std::numeric_limits
#include <vector> // std::vector
#include <stdexcept> // std::invalid_argument
#include <string> // std::string

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 100.0 * numeric_limits<double>::epsilon(); 

BoostDataContainer<ExampleScreeningFunction> container;

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(CheckUnknownCoreFunctionType_ThrowsError)
{
    ScreeningFunction fs;

    BOOST_REQUIRE_THROW(fs.setCoreFunction("bla"),
                        invalid_argument);
}

BOOST_AUTO_TEST_CASE(CheckDefaultInitialization_AllZeroOutput)
{
    ScreeningFunction fs;

    BOOST_REQUIRE_SMALL(fs.getInner(), accuracy);
    BOOST_REQUIRE_SMALL(fs.getOuter(), accuracy);
}

BOOST_AUTO_TEST_CASE(CheckInvalidCutoffParameter_ThrowsError)
{
    ScreeningFunction fs;

    BOOST_REQUIRE_THROW(fs.setInnerOuter(10.0, 5.0), invalid_argument);
}

BOOST_DATA_TEST_CASE(CombinedAndSeparateCall_SameResults,
                     bdata::make(container.examples),
                     e)
{
    double const rmin = 0.9 * e.inner;
    double const rmax = 1.1 * e.outer;
    size_t const n = 100;
    double const dr = (rmax - rmin) / n;

    BOOST_TEST_INFO(e.name + " \"" + e.type + "\"");
    //ofstream file;
    //file.open("fs." + e.type);
    //double f;
    //double df;
    //e.fs.fdf(e.rt, f, df);
    //file << strpr("# %f %24.16E %24.16E\n", e.rt, f, df);
    for (size_t i = 0; i < n; ++i)
    {
        double const r = rmin + i * dr;
        double f;
        double df;
        e.fs.fdf(r, f, df);
        //file << r << " " << f << " " << df << endl;
        BOOST_REQUIRE_SMALL(f - e.fs.f(r), accuracy);
        BOOST_REQUIRE_SMALL(df - e.fs.df(r), accuracy);
    }
    //file.close();

}

BOOST_DATA_TEST_CASE(CompareAnalyticNumericDerivatives_ComparableResults,
                     bdata::make(container.examples),
                     e)
{
    double const accuracyNumeric = 1.0E-7; 
    double const delta = 1.0E-6;
    double const rmin = e.inner + 10 * delta;
    double const rmax = e.outer - 10 * delta;
    size_t const n = 100;
    double const dr = (rmax - rmin) / n;

    BOOST_TEST_INFO(e.name + " \"" + e.type + "\"");
    for (size_t i = 0; i < n; ++i)
    {
        double const r = rmin + i * dr;
        double df = e.fs.df(r);
        double flow = e.fs.f(r - delta);
        double fhigh = e.fs.f(r + delta);
        //cout << r << " " << (fhigh - flow) / (2.0 * delta) << " " << df << endl;
        BOOST_REQUIRE_SMALL((fhigh - flow) / (2.0 * delta) - df,
                            accuracyNumeric);
    }

}

BOOST_DATA_TEST_CASE(CalculateFunctions_CorrectResults,
                     bdata::make(container.examples),
                     e)
{
    BOOST_TEST_INFO(e.name + " \"" + e.type + "\"");
    BOOST_REQUIRE_SMALL(e.fs.f(e.rt) - e.f, accuracy);
    BOOST_REQUIRE_SMALL(e.fs.f(e.rtbelow), accuracy);
    BOOST_REQUIRE_SMALL(e.fs.f(e.rtabove) - 1.0, accuracy);
}

BOOST_DATA_TEST_CASE(CalculateDerivatives_CorrectResults,
                     bdata::make(container.examples),
                     e)
{
    BOOST_TEST_INFO(e.name + " \"" + e.type + "\"");
    BOOST_REQUIRE_SMALL(e.fs.df(e.rt) - e.df, accuracy);
    BOOST_REQUIRE_SMALL(e.fs.df(e.rtbelow), accuracy);
    BOOST_REQUIRE_SMALL(e.fs.df(e.rtabove), accuracy);
}

BOOST_DATA_TEST_CASE(CalculateFunctionsAndDerivatives_CorrectResults,
                     bdata::make(container.examples),
                     e)
{
    BOOST_TEST_INFO(e.name + " \"" + e.type + "\"");
    double f;
    double df;
    e.fs.fdf(e.rt, f, df);
    BOOST_REQUIRE_SMALL(f - e.f, accuracy);
    BOOST_REQUIRE_SMALL(df - e.df, accuracy);

    e.fs.fdf(e.rtbelow, f, df);
    BOOST_REQUIRE_SMALL(f, accuracy);
    BOOST_REQUIRE_SMALL(df, accuracy);

    e.fs.fdf(e.rtabove, f, df);
    BOOST_REQUIRE_SMALL(f - 1.0, accuracy);
    BOOST_REQUIRE_SMALL(df, accuracy);
}

}
