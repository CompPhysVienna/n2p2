#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SymFncStatistics
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "SymFncStatistics.h"
#include <limits>    // std::numeric_limits
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Container_DefaultConstructed_ExpectedDefaults)
{
    SymFncStatistics::Container c;
    BOOST_REQUIRE_EQUAL(c.count, 0);
    BOOST_REQUIRE_EQUAL(c.countEW, 0);
    BOOST_REQUIRE_EQUAL(c.type, 0);
    BOOST_REQUIRE_EQUAL(c.min,  numeric_limits<double>::max());
    BOOST_REQUIRE_EQUAL(c.max, -numeric_limits<double>::max());
    BOOST_REQUIRE_SMALL(c.Gmin, accuracy);
    BOOST_REQUIRE_SMALL(c.Gmax, accuracy);
    BOOST_REQUIRE_SMALL(c.sum, accuracy);
    BOOST_REQUIRE_SMALL(c.sum2, accuracy);
    BOOST_REQUIRE_EQUAL(c.element, "");
    BOOST_REQUIRE(c.indexStructureEW.empty());
    BOOST_REQUIRE(c.indexAtomEW.empty());
    BOOST_REQUIRE(c.valueEW.empty());
}

BOOST_AUTO_TEST_CASE(Constructor_DefaultFlags_AllFalse)
{
    SymFncStatistics s;
    BOOST_REQUIRE_EQUAL(s.collectStatistics, false);
    BOOST_REQUIRE_EQUAL(s.collectExtrapolationWarnings, false);
    BOOST_REQUIRE_EQUAL(s.writeExtrapolationWarnings, false);
    BOOST_REQUIRE_EQUAL(s.stopOnExtrapolationWarnings, false);
    BOOST_REQUIRE(s.data.empty());
}

BOOST_AUTO_TEST_CASE(AddValue_AccumulatesCountSumMinMax)
{
    SymFncStatistics s;
    s.addValue(0, 2.0);
    s.addValue(0, -1.0);
    s.addValue(0, 5.0);

    SymFncStatistics::Container const& c = s.data.at(0);
    BOOST_REQUIRE_EQUAL(c.count, 3);
    BOOST_REQUIRE_SMALL(c.sum  - 6.0 , accuracy); // 2 - 1 + 5
    BOOST_REQUIRE_SMALL(c.sum2 - 30.0, accuracy); // 4 + 1 + 25
    BOOST_REQUIRE_SMALL(c.min  - (-1.0), accuracy);
    BOOST_REQUIRE_SMALL(c.max  - 5.0   , accuracy);
}

BOOST_AUTO_TEST_CASE(AddValue_SeparateIndicesTrackedIndependently)
{
    SymFncStatistics s;
    s.addValue(0, 1.0);
    s.addValue(1, -3.0);
    s.addValue(1, -7.0);

    BOOST_REQUIRE_EQUAL(s.data.size(), 2);
    BOOST_REQUIRE_EQUAL(s.data.at(0).count, 1);
    BOOST_REQUIRE_EQUAL(s.data.at(1).count, 2);
    BOOST_REQUIRE_SMALL(s.data.at(1).min - (-7.0), accuracy);
    BOOST_REQUIRE_SMALL(s.data.at(1).max - (-3.0), accuracy);
}

BOOST_AUTO_TEST_CASE(AddExtrapolationWarning_AccumulatesEntries)
{
    SymFncStatistics s;
    s.addExtrapolationWarning(0, 2, 1.5, -1.0, 1.0, "H", 10, 20);
    s.addExtrapolationWarning(0, 2, 2.5, -1.0, 1.0, "H", 11, 21);

    SymFncStatistics::Container const& c = s.data.at(0);
    BOOST_REQUIRE_EQUAL(c.countEW, 2);
    BOOST_REQUIRE_EQUAL(c.type, 2);
    BOOST_REQUIRE_EQUAL(c.element, "H");
    BOOST_REQUIRE_SMALL(c.Gmin - (-1.0), accuracy);
    BOOST_REQUIRE_SMALL(c.Gmax - 1.0   , accuracy);
    BOOST_REQUIRE_EQUAL(c.valueEW.size(), 2);
    BOOST_REQUIRE_SMALL(c.valueEW.at(0) - 1.5, accuracy);
    BOOST_REQUIRE_SMALL(c.valueEW.at(1) - 2.5, accuracy);
    BOOST_REQUIRE_EQUAL(c.indexStructureEW.at(0), 10);
    BOOST_REQUIRE_EQUAL(c.indexStructureEW.at(1), 11);
    BOOST_REQUIRE_EQUAL(c.indexAtomEW.at(0), 20);
    BOOST_REQUIRE_EQUAL(c.indexAtomEW.at(1), 21);
}

BOOST_AUTO_TEST_CASE(CountExtrapolationWarnings_SumsAcrossIndices)
{
    SymFncStatistics s;
    s.addExtrapolationWarning(0, 0, 1.0, 0.0, 1.0, "H", 0, 0);
    s.addExtrapolationWarning(0, 0, 1.0, 0.0, 1.0, "H", 0, 1);
    s.addExtrapolationWarning(1, 0, 1.0, 0.0, 1.0, "O", 0, 2);

    BOOST_REQUIRE_EQUAL(s.countExtrapolationWarnings(), 3);
}

BOOST_AUTO_TEST_CASE(GetExtrapolationWarningLines_OneLinePerWarning)
{
    SymFncStatistics s;
    s.addExtrapolationWarning(0, 3, 4.2, -1.0, 1.0, "Cu", 7, 8);
    s.addExtrapolationWarning(2, 1, 9.9, -2.0, 2.0, "S" , 5, 6);

    vector<string> const lines = s.getExtrapolationWarningLines();
    BOOST_REQUIRE_EQUAL(lines.size(), 2);

    // Symmetry function index is reported 1-based (it->first + 1).
    BOOST_REQUIRE(lines.at(0).find("SYMFUNC:    1") != string::npos);
    BOOST_REQUIRE(lines.at(0).find("ELEMENT: Cu") != string::npos);
    BOOST_REQUIRE(lines.at(0).find("STRUCTURE:      7") != string::npos);
    BOOST_REQUIRE(lines.at(0).find("ATOM:         8") != string::npos);

    BOOST_REQUIRE(lines.at(1).find("SYMFUNC:    3") != string::npos);
    BOOST_REQUIRE(lines.at(1).find("ELEMENT:  S") != string::npos);
}

BOOST_AUTO_TEST_CASE(ResetStatistics_ClearsStatsButKeepsWarnings)
{
    SymFncStatistics s;
    s.addValue(0, 3.0);
    s.addExtrapolationWarning(0, 1, 3.0, -1.0, 1.0, "H", 0, 0);

    s.resetStatistics();

    SymFncStatistics::Container const& c = s.data.at(0);
    BOOST_REQUIRE_EQUAL(c.count, 0);
    BOOST_REQUIRE_EQUAL(c.min,  numeric_limits<double>::max());
    BOOST_REQUIRE_EQUAL(c.max, -numeric_limits<double>::max());
    BOOST_REQUIRE_SMALL(c.sum, accuracy);
    // Extrapolation warning data must survive.
    BOOST_REQUIRE_EQUAL(c.countEW, 1);
    BOOST_REQUIRE_EQUAL(c.element, "H");
}

BOOST_AUTO_TEST_CASE(ResetExtrapolationWarnings_ClearsWarningsButKeepsStats)
{
    SymFncStatistics s;
    s.addValue(0, 3.0);
    s.addExtrapolationWarning(0, 1, 3.0, -1.0, 1.0, "H", 0, 0);

    s.resetExtrapolationWarnings();

    SymFncStatistics::Container const& c = s.data.at(0);
    BOOST_REQUIRE_EQUAL(c.countEW, 0);
    BOOST_REQUIRE_EQUAL(c.element, "");
    BOOST_REQUIRE(c.valueEW.empty());
    // Statistics data must survive.
    BOOST_REQUIRE_EQUAL(c.count, 1);
    BOOST_REQUIRE_SMALL(c.sum - 3.0, accuracy);
}

BOOST_AUTO_TEST_CASE(Clear_RemovesAllEntries)
{
    SymFncStatistics s;
    s.addValue(0, 1.0);
    s.addValue(1, 2.0);
    BOOST_REQUIRE_EQUAL(s.data.size(), 2);

    s.clear();
    BOOST_REQUIRE(s.data.empty());
}

BOOST_AUTO_TEST_SUITE_END()
