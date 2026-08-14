#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ElementMap
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "ElementMap.h"
#include <stdexcept> // std::runtime_error, std::range_error
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(RegisterElements_SortsByAtomicNumber)
{
    ElementMap em;
    // Input order deliberately not sorted; S (Z=16), Zn (Z=30), Cd (Z=48)
    // must end up ordered by ascending atomic number.
    size_t const n = em.registerElements("Cd S Zn");

    BOOST_REQUIRE_EQUAL(n, 3);
    BOOST_REQUIRE_EQUAL(em.size(), 3);
    BOOST_REQUIRE_EQUAL(em.symbol(0), "S");
    BOOST_REQUIRE_EQUAL(em.symbol(1), "Zn");
    BOOST_REQUIRE_EQUAL(em.symbol(2), "Cd");
    BOOST_REQUIRE_EQUAL(em.index("S") , 0);
    BOOST_REQUIRE_EQUAL(em.index("Zn"), 1);
    BOOST_REQUIRE_EQUAL(em.index("Cd"), 2);
}

BOOST_AUTO_TEST_CASE(OperatorBrackets_MatchIndexAndSymbolMethods)
{
    ElementMap em;
    em.registerElements("O H");

    BOOST_REQUIRE_EQUAL(em[size_t(0)], em.symbol(0));
    BOOST_REQUIRE_EQUAL(em[size_t(1)], em.symbol(1));
    BOOST_REQUIRE_EQUAL(em[string("H")], em.index("H"));
    BOOST_REQUIRE_EQUAL(em[string("O")], em.index("O"));
}

BOOST_AUTO_TEST_CASE(GetElementsString_SpaceSeparatedSortedOrder)
{
    ElementMap em;
    em.registerElements("Cd S Zn");
    BOOST_REQUIRE_EQUAL(em.getElementsString(), "S Zn Cd");

    ElementMap single;
    single.registerElements("He");
    BOOST_REQUIRE_EQUAL(single.getElementsString(), "He");

    ElementMap empty;
    BOOST_REQUIRE_EQUAL(empty.getElementsString(), "");
}

BOOST_AUTO_TEST_CASE(AtomicNumber_MemberMatchesStatic)
{
    ElementMap em;
    em.registerElements("Cd S Zn");

    // Reference atomic numbers, standard periodic table order.
    BOOST_REQUIRE_EQUAL(em.atomicNumber(0), 16); // S
    BOOST_REQUIRE_EQUAL(em.atomicNumber(1), 30); // Zn
    BOOST_REQUIRE_EQUAL(em.atomicNumber(2), 48); // Cd
}

BOOST_AUTO_TEST_CASE(StaticAtomicNumber_KnownElements_CorrectValues)
{
    BOOST_REQUIRE_EQUAL(ElementMap::atomicNumber("H") , 1);
    BOOST_REQUIRE_EQUAL(ElementMap::atomicNumber("C") , 6);
    BOOST_REQUIRE_EQUAL(ElementMap::atomicNumber("O") , 8);
    BOOST_REQUIRE_EQUAL(ElementMap::atomicNumber("Cu"), 29);
    BOOST_REQUIRE_EQUAL(ElementMap::atomicNumber("Zn"), 30);
    BOOST_REQUIRE_EQUAL(ElementMap::atomicNumber("No"), 102); // last known element.
}

BOOST_AUTO_TEST_CASE(StaticAtomicNumber_UnknownElement_Throws)
{
    BOOST_REQUIRE_THROW(ElementMap::atomicNumber("Xx"), runtime_error);
}

BOOST_AUTO_TEST_CASE(StaticSymbolFromAtomicNumber_RoundTrips)
{
    BOOST_REQUIRE_EQUAL(ElementMap::symbolFromAtomicNumber(1)  , "H" );
    BOOST_REQUIRE_EQUAL(ElementMap::symbolFromAtomicNumber(29) , "Cu");
    BOOST_REQUIRE_EQUAL(ElementMap::symbolFromAtomicNumber(102), "No"); // last known element.
}

BOOST_AUTO_TEST_CASE(StaticSymbolFromAtomicNumber_OutOfRange_Throws)
{
    BOOST_REQUIRE_THROW(ElementMap::symbolFromAtomicNumber(0)  , runtime_error);
    BOOST_REQUIRE_THROW(ElementMap::symbolFromAtomicNumber(103), runtime_error);
}

BOOST_AUTO_TEST_CASE(Index_UnknownSymbol_Throws)
{
    ElementMap em;
    em.registerElements("H");
    BOOST_REQUIRE_THROW(em.index("He"), range_error);
}

BOOST_AUTO_TEST_CASE(Symbol_OutOfRangeIndex_Throws)
{
    ElementMap em;
    em.registerElements("H");
    BOOST_REQUIRE_THROW(em.symbol(1), range_error);
}

BOOST_AUTO_TEST_CASE(RegisterElements_CalledTwice_Throws)
{
    ElementMap em;
    em.registerElements("H");
    BOOST_REQUIRE_THROW(em.registerElements("O"), runtime_error);
}

BOOST_AUTO_TEST_CASE(DeregisterElements_ClearsMapAndAllowsReRegistration)
{
    ElementMap em;
    em.registerElements("H He");
    BOOST_REQUIRE_EQUAL(em.size(), 2);

    em.deregisterElements();
    BOOST_REQUIRE_EQUAL(em.size(), 0);

    // Must not throw now that the map is empty again.
    size_t const n = em.registerElements("O");
    BOOST_REQUIRE_EQUAL(n, 1);
    BOOST_REQUIRE_EQUAL(em.symbol(0), "O");
}

BOOST_AUTO_TEST_CASE(Info_ContainsRegisteredElements)
{
    ElementMap em;
    em.registerElements("H O");
    vector<string> const info = em.info();
    BOOST_REQUIRE(!info.empty());

    bool foundH = false;
    bool foundO = false;
    for (auto const& line : info)
    {
        if (line.find("H ") != string::npos) foundH = true;
        if (line.find("O ") != string::npos) foundO = true;
    }
    BOOST_REQUIRE(foundH);
    BOOST_REQUIRE(foundO);
}

BOOST_AUTO_TEST_SUITE_END()
