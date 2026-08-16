#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE KalmanFilter
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "KalmanFilter.h"
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
    BOOST_REQUIRE_THROW(KalmanFilter(0, KalmanFilter::KT_STANDARD),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(Constructor_UnknownType_Throws)
{
    BOOST_REQUIRE_THROW(
        KalmanFilter(1, static_cast<KalmanFilter::KalmanType>(2)),
        runtime_error);
}

BOOST_AUTO_TEST_CASE(Standard_Update_CorrectValues)
{
    // Scalar case (sizeState = sizeObservation = 1): reference values
    // obtained from an independent Python re-implementation of the exact
    // update formula (see KalmanFilter::update(), KT_STANDARD branch),
    // replicating the same operation order/precision.
    KalmanFilter kf(1, KalmanFilter::KT_STANDARD);
    kf.setParametersStandard(2.0, 0.01, 0.1, 0.001, 1.0, 0.1, 10.0);

    vector<double> state = {1.0};
    double const error = 0.4;
    vector<double> const jacobian = {0.5};

    kf.setState(state.data());
    kf.setError(&error);
    kf.setJacobian(jacobian.data());

    kf.update();
    BOOST_REQUIRE_SMALL(state.at(0) - 1.0971027059695633, accuracy);

    // Second consecutive update: also exercises the learning-rate increase
    // and process-noise annealing recurrences between calls.
    kf.update();
    BOOST_REQUIRE_SMALL(state.at(0) - 1.193618839027742, accuracy);

    BOOST_REQUIRE_EQUAL(kf.getNumUpdates(), 2);
}

BOOST_AUTO_TEST_CASE(FadingMemory_Update_CorrectValues)
{
    // Same scalar-case approach as Standard_Update_CorrectValues, for the
    // KT_FADINGMEMORY branch.
    KalmanFilter kf(1, KalmanFilter::KT_FADINGMEMORY);
    kf.setParametersFadingMemory(2.0, 0.01, 0.1, 0.001, 0.99, 0.9);

    vector<double> state = {1.0};
    double const error = 0.4;
    vector<double> const jacobian = {0.5};

    kf.setState(state.data());
    kf.setError(&error);
    kf.setJacobian(jacobian.data());

    kf.update();

    BOOST_REQUIRE_SMALL(state.at(0) - 1.0896860986547086, accuracy);
    BOOST_REQUIRE_SMALL(kf.getLambda() - 0.991, accuracy);
    BOOST_REQUIRE_SMALL(kf.getGamma() - 0.5022601707684581, accuracy);
}

BOOST_AUTO_TEST_CASE(Standard_InfoAndStatus_ReportCorrectType)
{
    KalmanFilter kf(2, KalmanFilter::KT_STANDARD);
    kf.setParametersStandard(1.0, 0.01, 0.1, 0.001, 1.0, 0.1, 10.0);

    vector<double> state = {0.0, 0.0};
    double const error = 1.0;
    vector<double> const jacobian = {0.3, -0.2};
    kf.setState(state.data());
    kf.setError(&error);
    kf.setJacobian(jacobian.data());
    kf.update();

    vector<string> const info = kf.info();
    BOOST_REQUIRE(!info.empty());
    BOOST_REQUIRE(info.at(0).find("KT_STANDARD") != string::npos);

    string const s = kf.status(3);
    BOOST_REQUIRE(s.find("3") != string::npos);

    vector<string> const header = kf.statusHeader();
    BOOST_REQUIRE(!header.empty());
}

BOOST_AUTO_TEST_CASE(FadingMemory_InfoAndStatus_ReportCorrectType)
{
    KalmanFilter kf(2, KalmanFilter::KT_FADINGMEMORY);
    kf.setParametersFadingMemory(1.0, 0.01, 0.1, 0.001, 0.99, 0.9);

    vector<double> state = {0.0, 0.0};
    double const error = 1.0;
    vector<double> const jacobian = {0.3, -0.2};
    kf.setState(state.data());
    kf.setError(&error);
    kf.setJacobian(jacobian.data());
    kf.update();

    vector<string> const info = kf.info();
    BOOST_REQUIRE(!info.empty());
    BOOST_REQUIRE(info.at(0).find("KT_FADINGMEMORY") != string::npos);

    string const s = kf.status(1);
    BOOST_REQUIRE(s.find("1") != string::npos);

    vector<string> const header = kf.statusHeader();
    BOOST_REQUIRE(!header.empty());
}

BOOST_AUTO_TEST_CASE(Getters_ReflectSetParameters)
{
    KalmanFilter kf(2, KalmanFilter::KT_STANDARD);
    kf.setParametersStandard(2.0, 0.02, 0.05, 0.001, 0.5, 0.2, 5.0);

    BOOST_REQUIRE(kf.getType() == KalmanFilter::KT_STANDARD);
    BOOST_REQUIRE_SMALL(kf.getEpsilon() - 2.0, accuracy);
    BOOST_REQUIRE_SMALL(kf.getQ0() - 0.02, accuracy);
    BOOST_REQUIRE_SMALL(kf.getQtau() - 0.05, accuracy);
    BOOST_REQUIRE_SMALL(kf.getQmin() - 0.001, accuracy);
    BOOST_REQUIRE_SMALL(kf.getEta() - 0.5, accuracy);
    BOOST_REQUIRE_EQUAL(kf.getNumUpdates(), 0);

    kf.setSizeObservation(3);
    BOOST_REQUIRE_EQUAL(kf.getSizeObservation(), 3);
}

BOOST_AUTO_TEST_SUITE_END()
