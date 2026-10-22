#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GradientDescent
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "GradientDescent.h"
#include <cstddef>   // std::size_t
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Constructor_ZeroSizeState_Throws)
{
    BOOST_REQUIRE_THROW(GradientDescent(0, GradientDescent::DT_FIXED),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(Fixed_Update_CorrectValues)
{
    size_t const n = 3;
    GradientDescent gd(n, GradientDescent::DT_FIXED);
    gd.setParametersFixed(0.1);

    vector<double> state = {1.0, 2.0, 3.0};
    double const error = 2.0;
    vector<double> const gradient = {0.5, -0.5, 1.0};

    gd.setState(state.data());
    gd.setError(&error);
    gd.setJacobian(gradient.data());
    gd.update();

    vector<double> const expected = {1.1, 1.9, 3.2};
    for (size_t i = 0; i < n; ++i)
    {
        BOOST_REQUIRE_SMALL(state.at(i) - expected.at(i), accuracy);
    }
}

BOOST_AUTO_TEST_CASE(Adam_Update_CorrectValues)
{
    size_t const n = 2;
    GradientDescent gd(n, GradientDescent::DT_ADAM);
    gd.setParametersAdam(0.001, 0.9, 0.999, 1.0E-8);

    vector<double> state = {0.0, 0.0};
    double const error = 1.0;
    vector<double> const gradient = {1.0, -1.0};

    gd.setState(state.data());
    gd.setError(&error);
    gd.setJacobian(gradient.data());

    // Reference values obtained from an independent Python re-implementation
    // of the exact update formula (see GradientDescent::update(), DT_ADAM
    // branch) for two consecutive calls.
    gd.update();
    vector<double> const expected1 = { 9.999996837723340E-04,
                                       -9.999996837723340E-04};
    for (size_t i = 0; i < n; ++i)
    {
        BOOST_REQUIRE_SMALL(state.at(i) - expected1.at(i), accuracy);
    }

    gd.update();
    vector<double> const expected2 = { 1.999999460109657E-03,
                                       -1.999999460109657E-03};
    for (size_t i = 0; i < n; ++i)
    {
        BOOST_REQUIRE_SMALL(state.at(i) - expected2.at(i), accuracy);
    }
}

BOOST_AUTO_TEST_CASE(Fixed_InfoAndStatus_ReportCorrectType)
{
    GradientDescent gd(1, GradientDescent::DT_FIXED);
    gd.setParametersFixed(0.1);

    vector<string> const info = gd.info();
    BOOST_REQUIRE(!info.empty());
    BOOST_REQUIRE(info.at(0).find("DT_FIXED") != string::npos);

    string const s = gd.status(5);
    BOOST_REQUIRE(s.find("5") != string::npos);

    vector<string> const header = gd.statusHeader();
    BOOST_REQUIRE(!header.empty());
}

BOOST_AUTO_TEST_CASE(Adam_InfoAndStatus_ReportCorrectType)
{
    GradientDescent gd(2, GradientDescent::DT_ADAM);
    gd.setParametersAdam(0.001, 0.9, 0.999, 1.0E-8);

    vector<double> state = {0.0, 0.0};
    double const error = 1.0;
    vector<double> const gradient = {1.0, -1.0};
    gd.setState(state.data());
    gd.setError(&error);
    gd.setJacobian(gradient.data());
    gd.update();

    vector<string> const info = gd.info();
    BOOST_REQUIRE(!info.empty());
    BOOST_REQUIRE(info.at(0).find("DT_ADAM") != string::npos);

    // Status line for Adam additionally reports beta1t, beta2t and the
    // moment magnitudes.
    string const s = gd.status(1);
    BOOST_REQUIRE(s.find("1") != string::npos);

    vector<string> const header = gd.statusHeader();
    BOOST_REQUIRE(!header.empty());
}

BOOST_AUTO_TEST_SUITE_END()
