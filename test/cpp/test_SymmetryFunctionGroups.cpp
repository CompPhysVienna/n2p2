#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SymGrps
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
#include "SymFncExpAngn.h"
#include "SymFncExpAngw.h"
#include "SymFncExpRadWeighted.h"
#include "SymFncExpAngnWeighted.h"
#include "SymFncCompRad.h"
#include "SymFncCompAngw.h"
#include "SymFncCompAngn.h"
#include "SymFncCompRadWeighted.h"
#include "SymFncCompAngnWeighted.h"
#include "SymFncCompAngwWeighted.h"
#include "SymGrp.h"
#include "SymGrpExpRad.h"
#include "SymGrpExpAngn.h"
#include "SymGrpExpAngw.h"
#include "SymGrpExpRadWeighted.h"
#include "SymGrpExpAngnWeighted.h"
#include "SymGrpCompRad.h"
#include "SymGrpCompAngw.h"
#include "SymGrpCompAngn.h"
#include "SymGrpCompRadWeighted.h"
#include "SymGrpCompAngnWeighted.h"
#include "SymGrpCompAngwWeighted.h"
#include "utility.h"
#include <cstddef> // std::size_t
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

SymGrp* setupSymmetryFunctionGroup(ElementMap em, SymFnc const& sf)
{
    SymGrp* sfg;
    size_t const type = sf.getType();
    if      (type ==  2)  sfg = new SymGrpExpRad(em);
    else if (type ==  3)  sfg = new SymGrpExpAngn(em);
    else if (type ==  9)  sfg = new SymGrpExpAngw(em);
    else if (type == 12)  sfg = new SymGrpExpRadWeighted(em);
    else if (type == 13)  sfg = new SymGrpExpAngnWeighted(em);
    else if (type == 20)  sfg = new SymGrpCompRad(em);
    else if (type == 21)  sfg = new SymGrpCompAngn(em);
    else if (type == 22)  sfg = new SymGrpCompAngw(em);
    else if (type == 23)  sfg = new SymGrpCompRadWeighted(em);
    else if (type == 24)  sfg = new SymGrpCompAngnWeighted(em);
    else if (type == 25)  sfg = new SymGrpCompAngwWeighted(em);
    else
    {
        throw runtime_error("ERROR: Unknown symmetry function type.\n");
    }

    sfg->addMember(&sf);
    sfg->sortMembers();
    sfg->setScalingFactors();

    return sfg;
}

void recalculateSymmetryFunctionGroup(Structure&    s,
                                      Atom&         a,
                                      SymGrp const& sfg)
{
    s.clearNeighborList();
    s.calculateNeighborList(10.0);
    a.allocate(true);
    sfg.calculate(a, true);

    return;
}

void compareAnalyticNumericDerivGroup(Structure&   s,
                                      ElementMap&  em,
                                      size_t const type,
                                      string const setupLine)
{
    SymFnc* sf = setupSymmetryFunction(em, type, setupLine);

    SymGrp* sfg = setupSymmetryFunctionGroup(em, *sf);

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
            recalculateSymmetryFunctionGroup(s, s.atoms.at(0), *sfg);
            double Gh = s.atoms.at(0).G.at(0);
            s.atoms.at(ia).r[ic] -= 2 * h;
            recalculateSymmetryFunctionGroup(s, s.atoms.at(0), *sfg);
            double Gl = s.atoms.at(0).G.at(0);
            s.atoms.at(ia).r[ic] += h;
            results.back().push_back((Gh - Gl)/(2 * h));
        }
    }

    // Calculate symmetry function for atom 0.
    recalculateSymmetryFunctionGroup(s, s.atoms.at(0), *sfg);

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

    delete sfg;
    delete sf;

    return;
}

void checkAbsoluteValueGroup(Structure&   s,
                             ElementMap&  em,
                             size_t const type,
                             string const setupLine,
                             double const value)
{
    SymFnc* sf = setupSymmetryFunction(em, type, setupLine);

    SymGrp* sfg = setupSymmetryFunctionGroup(em, *sf);

    // Allocate symmetry function arrays.
    s.atoms.at(0).numSymmetryFunctions = 1;
    s.atoms.at(0).numSymmetryFunctionDerivatives = vector<size_t>(em.size(), 1);
#ifndef NOSFCACHE
    s.atoms.at(0).cacheSizePerElement = vector<size_t>(maxElements,
                                                       maxCacheSize);
#endif
    s.atoms.at(0).allocate(true);

    // Calculate symmetry function for atom 0.
    recalculateSymmetryFunctionGroup(s, s.atoms.at(0), *sfg);

    BOOST_TEST_INFO(string("Symmetry function values, type ")
                    << type << "\n");
    BOOST_REQUIRE_SMALL(s.atoms.at(0).G.at(0) - value, accuracy);

    delete sf;

    return;
}

BOOST_AUTO_TEST_SUITE(IntegrationTests)

BOOST_DATA_TEST_CASE_F(FixtureThreeAtomsMono,
                       CompareAnalyticNumericDerivMono_EqualResults,
                       bdata::make(typesMono) ^ bdata::make(setupLinesMono),
                       type,
                       setupLine)
{
    compareAnalyticNumericDerivGroup(s, em, type, setupLine);
}

BOOST_DATA_TEST_CASE_F(FixtureThreeAtomsDual,
                       CompareAnalyticNumericDerivDual_EqualResults,
                       bdata::make(typesDual) ^ bdata::make(setupLinesDual),
                       type,
                       setupLine)
{
    compareAnalyticNumericDerivGroup(s, em, type, setupLine);
}

BOOST_DATA_TEST_CASE_F(FixtureFourAtomsMixed,
                       CompareAnalyticNumericDerivMixed_EqualResults,
                       bdata::make(typesMixed) ^ bdata::make(setupLinesMixed),
                       type,
                       setupLine)
{
    compareAnalyticNumericDerivGroup(s, em, type, setupLine);
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
    checkAbsoluteValueGroup(s, em, type, setupLine, value);
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
    checkAbsoluteValueGroup(s, em, type, setupLine, value);
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
    checkAbsoluteValueGroup(s, em, type, setupLine, value);
}

BOOST_AUTO_TEST_SUITE_END()
