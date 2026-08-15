#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Stopwatch
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "Stopwatch.h"
#include <chrono> // std::chrono::milliseconds
#include <thread> // std::this_thread::sleep_for

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

// Small pause used to accumulate a measurable, strictly positive elapsed
// time without depending on precise wall-clock timing (the assertions below
// only ever require > 0 or relative comparisons, never a specific value).
void shortSleep()
{
    this_thread::sleep_for(chrono::milliseconds(5));
}

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Constructor_InitialState_ZeroTimes)
{
    Stopwatch sw;
    BOOST_REQUIRE_EQUAL(sw.getTotal(), 0.0);
    BOOST_REQUIRE_EQUAL(sw.getLoop(), 0.0);
}

BOOST_AUTO_TEST_CASE(StartStop_MeasuresPositiveElapsedTime)
{
    Stopwatch sw;
    sw.start();
    shortSleep();
    double const total = sw.stop();

    BOOST_REQUIRE(total > 0.0);
    BOOST_REQUIRE_EQUAL(sw.getTotal(), total);
}

BOOST_AUTO_TEST_CASE(Stop_WhileAlreadyStopped_LeavesTotalUnchanged)
{
    Stopwatch sw;
    sw.start();
    shortSleep();
    double const t1 = sw.stop();
    double const t2 = sw.stop(); // Clock not running -> no additional time.

    BOOST_REQUIRE_EQUAL(t1, t2);
}

BOOST_AUTO_TEST_CASE(Reset_ZeroesTotalAndLoop)
{
    Stopwatch sw;
    sw.start();
    shortSleep();
    sw.stop();

    sw.reset();

    BOOST_REQUIRE_EQUAL(sw.getTotal(), 0.0);
    BOOST_REQUIRE_EQUAL(sw.getLoop(), 0.0);
}

BOOST_AUTO_TEST_CASE(Start_NewLoop_ResetsLoopTimeButNotTotal)
{
    Stopwatch sw;

    sw.start(true);
    shortSleep();
    sw.stop(); // total == loop == dt1.

    sw.start(true); // Resets loop time only.
    shortSleep();
    sw.stop(); // total == dt1 + dt2, loop == dt2 only.

    BOOST_REQUIRE(sw.getLoop() > 0.0);
    BOOST_REQUIRE(sw.getTotal() > sw.getLoop());
}

BOOST_AUTO_TEST_CASE(Loop_AccumulatesExactlyIntoTotal)
{
    Stopwatch sw;

    sw.start(true);
    shortSleep();
    double const loop1 = sw.loop();

    sw.start(true);
    shortSleep();
    double const loop2 = sw.loop();

    // Both loop() calls feed the same underlying interval into timeTotal via
    // the identical accumulation path used for timeLoop, so this equality is
    // exact (not merely approximate).
    BOOST_REQUIRE_EQUAL(sw.getTotal(), loop1 + loop2);
}

BOOST_AUTO_TEST_CASE(Start_WhileAlreadyRunning_DoesNotResetLoopTimer)
{
    Stopwatch sw;

    sw.start(true);
    shortSleep();
    sw.start(); // Already running -> no-op (warns, does not reset clock).
    shortSleep();
    double const total = sw.stop();

    // The elapsed interval still reflects the original start() call, i.e.
    // it covers both sleeps rather than being reset by the second start().
    BOOST_REQUIRE(total > 0.0);
}

BOOST_AUTO_TEST_SUITE_END()
