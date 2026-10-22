#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE EwaldTruncKolafaFixR
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "EwaldTruncKolafaFixR.h"
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1.0E7 * numeric_limits<double>::epsilon();

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(CalculateParameters_CorrectValues)
{
    EwaldTruncKolafaFixR t;
    EwaldGlobalSettings settings;
    settings.precision = 1.0E-6;
    settings.maxCharge = 1.0;
    settings.fourPiEps = 1.0;
    EwaldStructureData sData(1000.0, 100);
    EwaldParameters params;
    params.rCut = 8.0; // Fixed real-space cutoff (input, not computed).

    t.calculateParameters(settings, sData, params);

    // Reference values obtained from an independent Python re-implementation
    // of calculateC()/calculateEta()/calculateKCut() (including the
    // Newton-style iteration in calculateKCut()).
    double const etaRef  = 1.589592481847531E+00;
    double const kCutRef = 3.166054502866813E+00;
    BOOST_REQUIRE_SMALL(params.eta  - etaRef , accuracy);
    BOOST_REQUIRE_SMALL(params.kCut - kCutRef, accuracy);
    // rCut is a fixed input parameter and must remain unchanged.
    BOOST_REQUIRE_EQUAL(params.rCut, 8.0);
}

BOOST_AUTO_TEST_CASE(CalculateParameters_ZeroRCut_Throws)
{
    EwaldTruncKolafaFixR t;
    EwaldGlobalSettings settings;
    EwaldStructureData sData(1000.0, 100);
    EwaldParameters params;
    params.rCut = 0.0;

    BOOST_REQUIRE_THROW(t.calculateParameters(settings, sData, params),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(CalculateParameters_ZeroMaxCharge_ThrowsBadPrecision)
{
    EwaldTruncKolafaFixR t;
    EwaldGlobalSettings settings;
    settings.maxCharge = 0.0; // Forces C = 0, so x = ... / C diverges.
    EwaldStructureData sData(1000.0, 100);
    EwaldParameters params;
    params.rCut = 8.0;

    BOOST_REQUIRE_THROW(t.calculateParameters(settings, sData, params),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(IsEstimateReliable_ComparesEtaToMaxQSigma)
{
    EwaldTruncKolafaFixR t;
    EwaldGlobalSettings settings;
    EwaldParameters params;
    params.eta = 1.589592481847531E+00;

    settings.maxQSigma = 1.0;
    BOOST_REQUIRE_EQUAL(t.isEstimateReliable(settings, params), true);

    settings.maxQSigma = 5.0;
    BOOST_REQUIRE_EQUAL(t.isEstimateReliable(settings, params), false);
}

BOOST_AUTO_TEST_CASE(PublishedNewCutoffs_TriggeredByVolumeOrAtomCountChange)
{
    EwaldTruncKolafaFixR t;
    EwaldGlobalSettings settings;
    EwaldParameters params;
    params.rCut = 8.0;

    EwaldStructureData sData1(1000.0, 100);
    t.calculateParameters(settings, sData1, params);
    BOOST_REQUIRE_EQUAL(t.publishedNewCutoffs(), false);
    BOOST_REQUIRE_EQUAL(t.publishedNewCutoffs(), true);

    // Same volume and atom count: cached, no new cutoffs.
    t.calculateParameters(settings, sData1, params);
    BOOST_REQUIRE_EQUAL(t.publishedNewCutoffs(), true);

    // Same volume but different atom count: must recalculate.
    EwaldStructureData sData2(1000.0, 200);
    t.calculateParameters(settings, sData2, params);
    BOOST_REQUIRE_EQUAL(t.publishedNewCutoffs(), false);
}

BOOST_AUTO_TEST_SUITE_END()
