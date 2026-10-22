#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SymFncs
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include "fixtures.h"

#include "Atom.h"
#include "ElementMap.h"
#include "Structure.h"
#include "SymFnc.h"
#include "SymFncBaseCutoff.h"
#include "SymFncExpRad.h"
#include "SymFncCompRad.h"
#include "SymFncExpAngn.h"
#include "SymFncExpAngw.h"
#include "SymFncCompAngw.h"
#include "SymFncCompAngn.h"
#include "SymFncExpRadWeighted.h"
#include "SymFncExpAngnWeighted.h"
#include "SymFncCompRadWeighted.h"
#include "SymFncCompAngnWeighted.h"
#include "SymFncCompAngwWeighted.h"
#include "utility.h"
#include <cstddef> // std::size_t
//#include <iostream> // std::cerr
#include <limits> // std::numeric_limits
#include <string> // std::string
#include <vector> // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();
double const accuracyNumeric = 1E-6;
size_t const maxElements = 3;
size_t const maxCacheSize = 2;

SymFnc* setupSymmetryFunction(ElementMap   em,
                              size_t const type,
                              string const setupLine)
{
    SymFnc* sf;
    if      (type ==  2)  sf = new SymFncExpRad(em);
    else if (type ==  3)  sf = new SymFncExpAngn(em);
    else if (type ==  9)  sf = new SymFncExpAngw(em);
    else if (type == 12)  sf = new SymFncExpRadWeighted(em);
    else if (type == 13)  sf = new SymFncExpAngnWeighted(em);
    else if (type == 20)  sf = new SymFncCompRad(em);
    else if (type == 21)  sf = new SymFncCompAngn(em);
    else if (type == 22)  sf = new SymFncCompAngw(em);
    else if (type == 23)  sf = new SymFncCompRadWeighted(em);
    else if (type == 24)  sf = new SymFncCompAngnWeighted(em);
    else if (type == 25)  sf = new SymFncCompAngwWeighted(em);
    else
    {
        throw runtime_error("ERROR: Unknown symmetry function type.\n");
    }

    sf->setIndex(0);
    for (size_t i = 0; i < em.size(); ++i)
    {
        sf->setIndexPerElement(i, 0);
    }
    sf->setParameters(setupLine);
    SymFncBaseCutoff* sfcb = dynamic_cast<SymFncBaseCutoff*>(sf);
    if (sfcb != nullptr)
    {
        sfcb->setCutoffFunction(CutoffFunction::CT_TANHU, 0.0);
    }
    string scalingLine = "1 1 0.0 0.0 0.0 0.0";
    sf->setScalingType(SymFnc::ST_NONE, scalingLine, 0.0, 0.0);
#ifndef NOSFCACHE
    auto ci = sf->getCacheIdentifiers();
    vector<size_t> count(em.size());
    for (size_t i = 0; i < ci.size(); ++i)
    {
        size_t ne = atoi(split(ci.at(i))[0].c_str());
        sf->addCacheIndex(ne, count.at(ne), ci.at(i));
        count.at(ne)++;
    }
#endif

    return sf;
}

void recalculateSymmetryFunction(Structure&    s,
                                 Atom&         a,
                                 SymFnc const& sf)
{
    s.clearNeighborList();
    s.calculateNeighborList(10.0);
    a.allocate(true);
    sf.calculate(a, true);

    return;
}

void compareAnalyticNumericDeriv(Structure&   s,
                                 ElementMap&  em,
                                 size_t const type,
                                 string const setupLine)
{
    SymFnc* sf = setupSymmetryFunction(em, type, setupLine);

    // Allocate symmetry function arrays.
    s.atoms.at(0).numSymmetryFunctions = 1;
    s.atoms.at(0).numSymmetryFunctionDerivatives = vector<size_t>(em.size(), 1);
#ifndef NOSFCACHE
    s.atoms.at(0).cacheSizePerElement = vector<size_t>(maxElements,
                                                       maxCacheSize);
#endif
    s.atoms.at(0).allocate(true);

    double h = 1.0E-7;

    vector<vector<double> > results;
    for (size_t ia = 0; ia < s.numAtoms; ++ia)
    {
        results.push_back(vector<double>());
        for (size_t ic = 0; ic < 3; ++ic)
        {
            s.atoms.at(ia).r[ic] += h;
            recalculateSymmetryFunction(s, s.atoms.at(0), *sf);
            double Gh = s.atoms.at(0).G.at(0);
            s.atoms.at(ia).r[ic] -= 2 * h;
            recalculateSymmetryFunction(s, s.atoms.at(0), *sf);
            double Gl = s.atoms.at(0).G.at(0);
            s.atoms.at(ia).r[ic] += h;
            results.back().push_back((Gh - Gl)/(2 * h));
        }
    }

    // Calculate symmetry function for atom 0.
    recalculateSymmetryFunction(s, s.atoms.at(0), *sf);

    BOOST_TEST_INFO(string("Symmetry function derivatives, type ")
                    << type << "\n");
    for (size_t ic = 0; ic < 3; ++ic)
    {
        BOOST_REQUIRE_SMALL(s.atoms.at(0).dGdr.at(0)[ic]
                            - results.at(0).at(ic),
                            accuracyNumeric);
    }
    for (size_t ia = 1; ia < s.numAtoms; ++ia)
    {
        for (size_t ic = 0; ic < 3; ++ic)
        {
            BOOST_REQUIRE_SMALL(s.atoms.at(0).neighbors.at(ia-1).dGdr.at(0)[ic]
                                - results.at(ia).at(ic),
                                accuracyNumeric);
        }
    }

    delete sf;

    return;
}

void checkAbsoluteValue(Structure&   s,
                        ElementMap&  em,
                        size_t const type,
                        string const setupLine,
                        double const value)
{
    SymFnc* sf = setupSymmetryFunction(em, type, setupLine);

    // Allocate symmetry function arrays.
    s.atoms.at(0).numSymmetryFunctions = 1;
    s.atoms.at(0).numSymmetryFunctionDerivatives = vector<size_t>(em.size(), 1);
#ifndef NOSFCACHE
    s.atoms.at(0).cacheSizePerElement = vector<size_t>(maxElements,
                                                       maxCacheSize);
#endif
    s.atoms.at(0).allocate(true);

    // Calculate symmetry function for atom 0.
    recalculateSymmetryFunction(s, s.atoms.at(0), *sf);

    BOOST_TEST_INFO(string("Symmetry function values, type ")
                    << type << "\n");
//    cerr << strpr("%24.16E\n", s.atoms.at(0).G.at(0));
    BOOST_REQUIRE_SMALL(s.atoms.at(0).G.at(0) - value, accuracy);

    delete sf;

    return;
}

void checkOperatorsAndInfo(ElementMap&  em,
                           size_t const type,
                           string const setupLine)
{
    SymFnc* sf1 = setupSymmetryFunction(em, type, setupLine);
    SymFnc* sf2 = setupSymmetryFunction(em, type, setupLine);

    // Identical parameters: equal, and neither strictly orders before the
    // other.
    BOOST_REQUIRE(*sf1 == *sf2);
    BOOST_REQUIRE(!(*sf1 <  *sf2));
    BOOST_REQUIRE(!(*sf2 <  *sf1));
    BOOST_REQUIRE(*sf1 <= *sf2);

    vector<string> info = sf1->parameterInfo();
    BOOST_REQUIRE(!info.empty());

    string line = sf1->parameterLine();
    BOOST_REQUIRE(!line.empty());

    string settingsLine = sf1->getSettingsLine();
    BOOST_REQUIRE(settingsLine.find("symfunction_short") != string::npos);

    // Must not crash.
    sf1->changeLengthUnit(2.0);

    delete sf1;
    delete sf2;

    return;
}

void checkOperatorsDiffer(ElementMap&  em,
                          size_t const type,
                          string const setupLine1,
                          string const setupLine2)
{
    SymFnc* sf1 = setupSymmetryFunction(em, type, setupLine1);
    SymFnc* sf2 = setupSymmetryFunction(em, type, setupLine2);

    BOOST_REQUIRE(!(*sf1 == *sf2));
    bool const lt12 = *sf1 < *sf2;
    bool const lt21 = *sf2 < *sf1;
    // A strict weak ordering cannot have both directions true.
    BOOST_REQUIRE(!(lt12 && lt21));

    delete sf1;
    delete sf2;

    return;
}

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_DATA_TEST_CASE_F(FixtureThreeAtomsMono,
                       CheckOperatorsAndInfoMono,
                       bdata::make(typesMono) ^ bdata::make(setupLinesMono),
                       type,
                       setupLine)
{
    checkOperatorsAndInfo(em, type, setupLine);
}

BOOST_AUTO_TEST_CASE(OperatorsDiffer_WeightedTypes)
{
    ElementMap em;
    em.registerElements("H");

    checkOperatorsDiffer(em, 12, "H 12 0.001 0.0 10.0",
                                 "H 12 0.030 1.0 10.0");
    checkOperatorsDiffer(em, 13, "H 13 0.001  1.0 1.0 10.0 0.0",
                                 "H 13 0.030 -1.0 4.0 10.0 2.0");
    checkOperatorsDiffer(em, 23, "H 23 -10.0 10.0 p2",
                                 "H 23   0.0  8.0 p2");
    checkOperatorsDiffer(em, 24, "H 24 -10.0 10.0  0.0 180.0 p2",
                                 "H 24   0.0 10.0 10.0  60.0 p2");
    checkOperatorsDiffer(em, 25, "H 25 -10.0 10.0  0.0 180.0 p2",
                                 "H 25   0.0 10.0 10.0  60.0 p2");
}

BOOST_AUTO_TEST_CASE(CompactType_InvalidSubtype_Throws)
{
    ElementMap em;
    em.registerElements("H");
    BOOST_REQUIRE_THROW(setupSymmetryFunction(em, 23, "H 23 -10.0 10.0 xx"),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(CompactType_InvalidPolynomNumber_Throws)
{
    ElementMap em;
    em.registerElements("H");
    BOOST_REQUIRE_THROW(setupSymmetryFunction(em, 23, "H 23 -10.0 10.0 p5"),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(CompactType_RadialBoundaryInverted_Throws)
{
    ElementMap em;
    em.registerElements("H");
    BOOST_REQUIRE_THROW(setupSymmetryFunction(em, 23, "H 23 10.0 5.0 p2"),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(CompactAngularType_AngleBoundaryInverted_Throws)
{
    ElementMap em;
    em.registerElements("H");
    BOOST_REQUIRE_THROW(
        setupSymmetryFunction(em, 24, "H 24 -10.0 10.0 100.0 50.0 p2"),
        runtime_error);
}

BOOST_AUTO_TEST_CASE(CompactAngularType_AngleOutOfBounds_Throws)
{
    ElementMap em;
    em.registerElements("H");
    BOOST_REQUIRE_THROW(
        setupSymmetryFunction(em, 24, "H 24 -10.0 10.0 -10.0 5.0 p2"),
        runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(IntegrationTests)

BOOST_DATA_TEST_CASE_F(FixtureThreeAtomsMono,
                       CompareAnalyticNumericDerivMono_EqualResults,
                       bdata::make(typesMono) ^ bdata::make(setupLinesMono),
                       type,
                       setupLine)
{
    compareAnalyticNumericDeriv(s, em, type, setupLine);
}

BOOST_DATA_TEST_CASE_F(FixtureThreeAtomsDual,
                       CompareAnalyticNumericDerivDual_EqualResults,
                       bdata::make(typesDual) ^ bdata::make(setupLinesDual),
                       type,
                       setupLine)
{
    compareAnalyticNumericDeriv(s, em, type, setupLine);
}

BOOST_DATA_TEST_CASE_F(FixtureFourAtomsMixed,
                       CompareAnalyticNumericDerivMixed_EqualResults,
                       bdata::make(typesMixed) ^ bdata::make(setupLinesMixed),
                       type,
                       setupLine)
{
    compareAnalyticNumericDeriv(s, em, type, setupLine);
}

BOOST_DATA_TEST_CASE_F(FixtureThreeAtomsMono,
                       CheckAbsoluteValuesMono_CorrectResults,
                       bdata::make(typesMono)
                       ^ bdata::make(setupLinesMono)
                       ^ bdata::make(valuesMono),
                       type,
                       setupLine,
                       value)
{
    checkAbsoluteValue(s, em, type, setupLine, value);
}

BOOST_DATA_TEST_CASE_F(FixtureThreeAtomsDual,
                       CheckAbsoluteValuesDual_CorrectResults,
                       bdata::make(typesDual)
                       ^ bdata::make(setupLinesDual)
                       ^ bdata::make(valuesDual),
                       type,
                       setupLine,
                       value)
{
    checkAbsoluteValue(s, em, type, setupLine, value);
}

BOOST_DATA_TEST_CASE_F(FixtureFourAtomsMixed,
                       CheckAbsoluteValuesMixed_CorrectResults,
                       bdata::make(typesMixed)
                       ^ bdata::make(setupLinesMixed)
                       ^ bdata::make(valuesMixed),
                       type,
                       setupLine,
                       value)
{
    checkAbsoluteValue(s, em, type, setupLine, value);
}

BOOST_AUTO_TEST_SUITE_END()
