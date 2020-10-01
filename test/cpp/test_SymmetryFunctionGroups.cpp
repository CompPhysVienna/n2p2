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
#include "SymFncRadExp.h"
#include "SymFncRadPoly.h"
#include "SymFncRadPolyA.h"
#include "SymFncAngnExp.h"
#include "SymFncAngwExp.h"
#include "SymFncAngwPoly.h"
#include "SymFncAngnPoly.h"
#include "SymFncAngwPolyA.h"
#include "SymFncAngnPolyA.h"
#include "SymFncRadExpWeighted.h"
#include "SymFncAngnExpWeighted.h"
#include "SymGrp.h"
#include "SymGrpRadExp.h"
#include "SymGrpRadPoly.h"
#include "SymGrpRadPolyA.h"
#include "SymGrpAngnExp.h"
#include "SymGrpAngwExp.h"
#include "SymGrpAngwPoly.h"
#include "SymGrpAngnPoly.h"
#include "SymGrpAngwPolyA.h"
#include "SymGrpAngnPolyA.h"
#include "SymGrpRadExpWeighted.h"
#include "SymGrpAngnExpWeighted.h"
#include <cstddef> // std::size_t
#include <limits> // std::numeric_limits
#include <string> // std::string
#include <vector> // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();
double const accuracyNumeric = 1E-6;

SymFnc* setupSymmetryFunction(ElementMap   em,
                                        size_t const type,
                                        string const setupLine)
{
    SymFnc* sf;
    if      (type ==  2)  sf = new SymFncRadExp(em);
    else if (type ==  3)  sf = new SymFncAngnExp(em);
    else if (type ==  9)  sf = new SymFncAngwExp(em);
    else if (type == 12)  sf = new SymFncRadExpWeighted(em);
    else if (type == 13)  sf = new SymFncAngnExpWeighted(em);
    else if (type == 28)  sf = new SymFncRadPoly(em);
    else if (type == 280) sf = new SymFncRadPolyA(em);
    else if (type == 89)  sf = new SymFncAngwPoly(em);
    else if (type == 890) sf = new SymFncAngwPolyA(em);
    else if (type == 99)  sf = new SymFncAngnPoly(em);
    else if (type == 990) sf = new SymFncAngnPolyA(em);
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
    sf->setCutoffFunction(CutoffFunction::CT_TANHU, 0.0);
    string scalingLine = "1 1 0.0 0.0 0.0 0.0";
    sf->setScalingType(SymFnc::ST_NONE, scalingLine, 0.0, 0.0);

    return sf;
}

SymGrp* setupSymmetryFunctionGroup(ElementMap em, SymFnc const& sf)
{
    SymGrp* sfg;
    size_t const type = sf.getType();
    if      (type ==  2)  sfg = new SymGrpRadExp(em);
    else if (type ==  3)  sfg = new SymGrpAngnExp(em);
    else if (type ==  9)  sfg = new SymGrpAngwExp(em);
    else if (type == 12)  sfg = new SymGrpRadExpWeighted(em);
    else if (type == 13)  sfg = new SymGrpAngnExpWeighted(em);
    else if (type == 28)  sfg = new SymGrpRadPoly(em);
    else if (type == 280) sfg = new SymGrpRadPolyA(em);
    else if (type == 89)  sfg = new SymGrpAngwPoly(em);
    else if (type == 890) sfg = new SymGrpAngwPolyA(em);
    else if (type == 99)  sfg = new SymGrpAngnPoly(em);
    else if (type == 990) sfg = new SymGrpAngnPolyA(em);
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
    s.atoms.at(0).allocate(true);

    // Calculate symmetry function for atom 0.
    recalculateSymmetryFunctionGroup(s, s.atoms.at(0), *sfg);

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
