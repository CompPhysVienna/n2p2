#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE EwaldTruncKolafaOptEta
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "EwaldTruncKolafaOptEta.h"
#include <limits> // std::numeric_limits

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1.0E7 * numeric_limits<double>::epsilon();

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(CalculateParameters_CorrectValues)
{
    EwaldTruncKolafaOptEta t;
    EwaldGlobalSettings settings;
    settings.precision = 1.0E-6;
    settings.maxCharge = 1.0;
    settings.fourPiEps = 1.0;
    settings.maxQSigma = 0.0; // Below the computed eta, so no override.
    EwaldStructureData sData(1000.0, 100);
    EwaldParameters params;

    t.calculateParameters(settings, sData, params);

    // Reference values obtained from an independent Python re-implementation
    // of calculateC()/calculateEta()/calculateS()/calculateRCut()/
    // calculateKCut() (self-consistent eta/s iteration included).
    double const etaRef  = 3.232668889886725E+00;
    double const rCutRef = 1.604411556991580E+01;
    double const kCutRef = 1.535300297861868E+00;
    BOOST_REQUIRE_SMALL(params.eta  - etaRef , accuracy);
    BOOST_REQUIRE_SMALL(params.rCut - rCutRef, accuracy);
    BOOST_REQUIRE_SMALL(params.kCut - kCutRef, accuracy);
}

BOOST_AUTO_TEST_CASE(PublishedNewCutoffs_TriggeredByVolumeOrAtomCountChange)
{
    EwaldTruncKolafaOptEta t;
    EwaldGlobalSettings settings;
    EwaldParameters params;

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

BOOST_AUTO_TEST_CASE(IsEstimateReliable_AlwaysTrue)
{
    EwaldTruncKolafaOptEta t;
    EwaldGlobalSettings settings;
    EwaldParameters params;
    BOOST_REQUIRE_EQUAL(t.isEstimateReliable(settings, params), true);
}

BOOST_AUTO_TEST_SUITE_END()
