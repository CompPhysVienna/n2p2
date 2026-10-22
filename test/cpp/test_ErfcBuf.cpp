#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ErfcBuf
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "Atom.h"
#include "ErfcBuf.h"
#include <cmath>  // std::erfc
#include <limits> // std::numeric_limits
#include <vector> // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(DefaultConstructed_EmptyAndDefaultValuesPerPair)
{
    ErfcBuf buf;
    BOOST_REQUIRE(buf.f.empty());
    BOOST_REQUIRE_EQUAL(buf.numValuesPerPair, 1);
}

BOOST_AUTO_TEST_CASE(Reset_ResizesPerAtomAndInitializesToSentinel)
{
    vector<Atom> atoms(2);
    atoms.at(0).numNeighbors = 3;
    atoms.at(1).numNeighbors = 0;

    ErfcBuf buf;
    buf.reset(atoms, 2);

    BOOST_REQUIRE_EQUAL(buf.numValuesPerPair, 2);
    BOOST_REQUIRE_EQUAL(buf.f.size(), 2);
    BOOST_REQUIRE_EQUAL(buf.f.at(0).size(), 3 * 2);
    BOOST_REQUIRE_EQUAL(buf.f.at(1).size(), 0);
    for (double v : buf.f.at(0))
    {
        BOOST_REQUIRE_SMALL(v - (-1.0), accuracy);
    }
}

BOOST_AUTO_TEST_CASE(Getf_FirstCallComputesErfc)
{
    vector<Atom> atoms(1);
    atoms.at(0).numNeighbors = 1;

    ErfcBuf buf;
    buf.reset(atoms, 1);

    double const x = 0.75;
    double const result = buf.getf(0, 0, 0, x);
    BOOST_REQUIRE_SMALL(result - erfc(x), accuracy);
}

BOOST_AUTO_TEST_CASE(Getf_SecondCallReturnsCachedValueIgnoringNewX)
{
    vector<Atom> atoms(1);
    atoms.at(0).numNeighbors = 1;

    ErfcBuf buf;
    buf.reset(atoms, 1);

    double const xFirst = 0.75;
    double const first = buf.getf(0, 0, 0, xFirst);

    // A different x passed on the second call must be ignored: the cached
    // value from the first call is returned instead.
    double const xSecond = 3.0;
    double const second = buf.getf(0, 0, 0, xSecond);

    BOOST_REQUIRE_SMALL(second - first, accuracy);
    BOOST_REQUIRE_SMALL(second - erfc(xFirst), accuracy);
    BOOST_REQUIRE(fabs(second - erfc(xSecond)) > accuracy);
}

BOOST_AUTO_TEST_CASE(Getf_DifferentValIndexAndNeighIndexDoNotCollide)
{
    vector<Atom> atoms(1);
    atoms.at(0).numNeighbors = 2;

    ErfcBuf buf;
    buf.reset(atoms, 2);

    // (neighIndex=0, valIndex=1) -> slot 1; (neighIndex=1, valIndex=0) ->
    // slot 2. Different slots, must not share a cached value.
    double const a = buf.getf(0, 0, 1, 0.5);
    double const b = buf.getf(0, 1, 0, 2.0);

    BOOST_REQUIRE_SMALL(a - erfc(0.5), accuracy);
    BOOST_REQUIRE_SMALL(b - erfc(2.0), accuracy);
}

BOOST_AUTO_TEST_CASE(Getf_DifferentAtomsIndependentCaches)
{
    vector<Atom> atoms(2);
    atoms.at(0).numNeighbors = 1;
    atoms.at(1).numNeighbors = 1;

    ErfcBuf buf;
    buf.reset(atoms, 1);

    double const a = buf.getf(0, 0, 0, 1.0);
    double const b = buf.getf(1, 0, 0, 2.0);

    BOOST_REQUIRE_SMALL(a - erfc(1.0), accuracy);
    BOOST_REQUIRE_SMALL(b - erfc(2.0), accuracy);
}

BOOST_AUTO_TEST_SUITE_END()
