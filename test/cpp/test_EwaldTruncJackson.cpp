#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE EwaldTruncJackson
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "EwaldTruncJackson.h"
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1.0E5 * numeric_limits<double>::epsilon();

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(CalculateParameters_CorrectValues)
{
    EwaldTruncJackson t;
    EwaldGlobalSettings settings;
    settings.precision = 1.0E-6;
    EwaldStructureData sData(1000.0, 10);
    EwaldParameters params;

    t.calculateParameters(settings, sData, params);

    // Reference values obtained from an independent Python re-implementation
    // of calculateEta()/calculateRCut()/calculateKCut().
    double const etaRef  = 3.989422804014326E+00;
    double const rCutRef = 2.097048781806605E+01;
    double const kCutRef = 1.317614609428611E+00;
    BOOST_REQUIRE_SMALL(params.eta  - etaRef , accuracy);
    BOOST_REQUIRE_SMALL(params.rCut - rCutRef, accuracy);
    BOOST_REQUIRE_SMALL(params.kCut - kCutRef, accuracy);
}

BOOST_AUTO_TEST_CASE(CalculateParameters_PrecisionTooLarge_Throws)
{
    EwaldTruncJackson t;
    EwaldGlobalSettings settings;
    settings.precision = 1.0; // Method requires precision < 1.0.
    EwaldStructureData sData(1000.0, 10);
    EwaldParameters params;

    BOOST_REQUIRE_THROW(t.calculateParameters(settings, sData, params),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(CalculateParameters_ZeroVolume_Throws)
{
    EwaldTruncJackson t;
    EwaldGlobalSettings settings;
    EwaldParameters params;

    // Move the cached volume away from its default (0.0) first, so that the
    // subsequent call with V=0.0 is recognized as a genuine change and
    // triggers recalculation (eta = 0 for V = 0, which must be rejected).
    EwaldStructureData sDataInit(1000.0, 10);
    t.calculateParameters(settings, sDataInit, params);

    EwaldStructureData sDataZero(0.0, 10);
    BOOST_REQUIRE_THROW(t.calculateParameters(settings, sDataZero, params),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(PublishedNewCutoffs_ReflectsRecalculation)
{
    EwaldTruncJackson t;
    EwaldGlobalSettings settings;
    EwaldParameters params;

    // First calculation: cutoffs are new but not yet reported as published.
    EwaldStructureData sData1(1000.0, 10);
    t.calculateParameters(settings, sData1, params);
    BOOST_REQUIRE_EQUAL(t.publishedNewCutoffs(), false);
    // Querying again without a new calculation: already published.
    BOOST_REQUIRE_EQUAL(t.publishedNewCutoffs(), true);

    // Same volume again: cached, publish flag stays true.
    t.calculateParameters(settings, sData1, params);
    BOOST_REQUIRE_EQUAL(t.publishedNewCutoffs(), true);

    // Different volume: fresh cutoffs, not yet published.
    EwaldStructureData sData2(2000.0, 10);
    t.calculateParameters(settings, sData2, params);
    BOOST_REQUIRE_EQUAL(t.publishedNewCutoffs(), false);
}

BOOST_AUTO_TEST_CASE(IsEstimateReliable_AlwaysTrue)
{
    EwaldTruncJackson t;
    EwaldGlobalSettings settings;
    EwaldParameters params;
    BOOST_REQUIRE_EQUAL(t.isEstimateReliable(settings, params), true);
}

BOOST_AUTO_TEST_SUITE_END()
