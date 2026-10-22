#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE EwaldSetup
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "EwaldSetup.h"
#include "Log.h"
#include <fstream>   // std::ofstream
#include <iterator>  // std::istreambuf_iterator
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

double const accuracy = 1.0E5 * numeric_limits<double>::epsilon();

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Constructor_Defaults)
{
    EwaldSetup es;
    BOOST_REQUIRE(es.getTruncMethod() == EWALDTruncMethod::JACKSON_CATLOW);
    BOOST_REQUIRE_SMALL(es.params.eta, accuracy);
    BOOST_REQUIRE_SMALL(es.params.rCut, accuracy);
    BOOST_REQUIRE_SMALL(es.params.kCut, accuracy);
}

BOOST_AUTO_TEST_CASE(SetGetTruncMethod_RoundTrip)
{
    EwaldSetup es;
    es.setTruncMethod(EWALDTruncMethod::KOLAFA_PERRAM);
    BOOST_REQUIRE(es.getTruncMethod() == EWALDTruncMethod::KOLAFA_PERRAM);
}

BOOST_AUTO_TEST_CASE(ReadFromArgs_Jackson_WrongArgCount_Throws)
{
    EwaldSetup es; // Default truncMethod is JACKSON_CATLOW.
    BOOST_REQUIRE_THROW(es.readFromArgs({"1.0E-6", "1.0"}), runtime_error);
    BOOST_REQUIRE_THROW(es.readFromArgs({}), runtime_error);
}

BOOST_AUTO_TEST_CASE(ReadFromArgs_NonPositivePrecision_Throws)
{
    EwaldSetup es;
    BOOST_REQUIRE_THROW(es.readFromArgs({"0.0"}), runtime_error);
    BOOST_REQUIRE_THROW(es.readFromArgs({"-1.0E-6"}), runtime_error);
}

BOOST_AUTO_TEST_CASE(ReadFromArgs_KolafaPerram_OneArg_LeavesTruncImplUnset)
{
    // KolafaPerram's arg-count check allows 1-3 args, but only the 2- and
    // 3-arg branches actually construct a truncImpl. With exactly 1 arg,
    // readFromArgs() succeeds but leaves the truncation strategy unset, so
    // the next calculateParameters() call throws.
    EwaldSetup es;
    es.setTruncMethod(EWALDTruncMethod::KOLAFA_PERRAM);
    BOOST_REQUIRE_NO_THROW(es.readFromArgs({"1.0E-6"}));

    BOOST_REQUIRE_THROW(es.calculateParameters(1000.0, 100), runtime_error);
}

BOOST_AUTO_TEST_CASE(CalculateParameters_Jackson_MatchesDirectImplementation)
{
    EwaldSetup es; // JACKSON_CATLOW is the default.
    es.readFromArgs({"1.0E-6"});
    es.calculateParameters(1000.0, 10);

    // Reference values match test_EwaldTruncJackson.cpp's independently
    // derived (Python) reference for the same volume/precision.
    double const etaRef  = 3.989422804014326E+00;
    double const rCutRef = 2.097048781806605E+01;
    double const kCutRef = 1.317614609428611E+00;
    BOOST_REQUIRE_SMALL(es.params.eta  - etaRef , accuracy);
    BOOST_REQUIRE_SMALL(es.params.rCut - rCutRef, accuracy);
    BOOST_REQUIRE_SMALL(es.params.kCut - kCutRef, accuracy);
}

BOOST_AUTO_TEST_CASE(CalculateParameters_KolafaOptEta_TwoArgs_Succeeds)
{
    EwaldSetup es;
    es.setTruncMethod(EWALDTruncMethod::KOLAFA_PERRAM);
    es.readFromArgs({"1.0E-6", "1.0"}); // precision, maxCharge -> OptEta.
    es.calculateParameters(1000.0, 100);

    // Reference values match test_EwaldTruncKolafaOptEta.cpp.
    double const etaRef  = 3.232668889886725E+00;
    double const rCutRef = 1.604411556991580E+01;
    double const kCutRef = 1.535300297861868E+00;
    BOOST_REQUIRE_SMALL(es.params.eta  - etaRef , accuracy);
    BOOST_REQUIRE_SMALL(es.params.rCut - rCutRef, accuracy);
    BOOST_REQUIRE_SMALL(es.params.kCut - kCutRef, accuracy);
}

BOOST_AUTO_TEST_CASE(CalculateParameters_KolafaFixR_ThreeArgs_UsesProvidedRCut)
{
    EwaldSetup es;
    es.setTruncMethod(EWALDTruncMethod::KOLAFA_PERRAM);
    // precision, maxCharge, rCut -> FixR.
    es.readFromArgs({"1.0E-6", "1.0", "8.0"});
    es.calculateParameters(1000.0, 100);

    // Reference values match test_EwaldTruncKolafaFixR.cpp.
    double const etaRef  = 1.589592481847531E+00;
    double const kCutRef = 3.166054502866813E+00;
    BOOST_REQUIRE_SMALL(es.params.eta  - etaRef , accuracy);
    BOOST_REQUIRE_SMALL(es.params.kCut - kCutRef, accuracy);
    BOOST_REQUIRE_EQUAL(es.params.rCut, 8.0);
}

BOOST_AUTO_TEST_CASE(CalculateParameters_NoTruncImpl_Throws)
{
    EwaldSetup es;
    BOOST_REQUIRE_THROW(es.calculateParameters(1000.0, 10), runtime_error);
}

BOOST_AUTO_TEST_CASE(ToNormalizedUnits_JacksonMethod_NoOp)
{
    EwaldSetup es; // JACKSON_CATLOW.
    es.readFromArgs({"1.0E-6"});
    es.calculateParameters(1000.0, 10);
    EwaldParameters const before = es.params;
    double const precisionBefore = es.getPrecision();

    es.toNormalizedUnits(2.0, 3.0);

    BOOST_REQUIRE_SMALL(es.params.eta  - before.eta , accuracy);
    BOOST_REQUIRE_SMALL(es.params.rCut - before.rCut, accuracy);
    BOOST_REQUIRE_SMALL(es.params.kCut - before.kCut, accuracy);
    BOOST_REQUIRE_SMALL(es.getPrecision() - precisionBefore, accuracy);
}

BOOST_AUTO_TEST_CASE(ToNormalizedUnits_KolafaMethod_ScalesParameters)
{
    EwaldSetup es;
    es.setTruncMethod(EWALDTruncMethod::KOLAFA_PERRAM);
    es.readFromArgs({"1.0E-6", "1.0", "8.0"});
    es.calculateParameters(1000.0, 100);

    EwaldParameters const before = es.params;
    double const precisionBefore = es.getPrecision();

    double const convEnergy = 2.0;
    double const convLength = 3.0;
    es.toNormalizedUnits(convEnergy, convLength);

    BOOST_REQUIRE_SMALL(es.getPrecision()
                        - precisionBefore * convEnergy / convLength,
                        accuracy);
    BOOST_REQUIRE_SMALL(es.params.eta  - before.eta  * convLength, accuracy);
    BOOST_REQUIRE_SMALL(es.params.rCut - before.rCut * convLength, accuracy);
    BOOST_REQUIRE_SMALL(es.params.kCut - before.kCut / convLength, accuracy);
}

// publishedNewCutoffs()/isEstimateReliable() are private (only used
// internally by logEwaldCutoffs()) -- exercised indirectly by
// LogEwaldCutoffs_WritesOnFirstCallOnly below rather than called directly.

BOOST_AUTO_TEST_CASE(GetMaxChargeMaxQSigma_ReflectSettings)
{
    EwaldSetup es;
    es.setTruncMethod(EWALDTruncMethod::KOLAFA_PERRAM);
    es.readFromArgs({"1.0E-6", "2.5", "8.0"});

    BOOST_REQUIRE_SMALL(es.getMaxCharge() - 2.5, accuracy);
    BOOST_REQUIRE_SMALL(es.getPrecision() - 1.0E-6, accuracy);

    es.setMaxQSigma(0.5);
    BOOST_REQUIRE_SMALL(es.getMaxQSigma() - 0.5, accuracy);
}

BOOST_FIXTURE_TEST_CASE(LogEwaldCutoffs_WritesOnFirstCallOnly, FixtureRepairDir)
{
    EwaldSetup es;
    es.readFromArgs({"1.0E-6"});
    es.calculateParameters(1000.0, 10);

    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    ofstream logFile("ewald.log");
    Log log;
    log.writeToStdout = false;
    log.registerStreamPointer(&logFile);

    // First call: cutoffs not yet published -> logs details.
    es.logEwaldCutoffs(log, 1.0);
    logFile.flush();

    ifstream check("ewald.log");
    string content((istreambuf_iterator<char>(check)),
                   istreambuf_iterator<char>());
    BOOST_REQUIRE(content.find("Real space cutoff") != string::npos);

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_SUITE_END()
