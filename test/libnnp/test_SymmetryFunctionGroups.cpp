#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SymmetryFunctionGroups
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include "fixtures.h"

#include "Atom.h"
#include "ElementMap.h"
#include "Structure.h"
#include "SymmetryFunction.h"
#include "SymmetryFunctionRadial.h"
#include "SymmetryFunctionAngularNarrow.h"
#include "SymmetryFunctionAngularWide.h"
#include "SymmetryFunctionWeightedRadial.h"
#include "SymmetryFunctionWeightedAngular.h"
#include "SymmetryFunctionGroup.h"
#include "SymmetryFunctionGroupRadial.h"
#include "SymmetryFunctionGroupAngularNarrow.h"
#include "SymmetryFunctionGroupAngularWide.h"
#include "SymmetryFunctionGroupWeightedRadial.h"
#include "SymmetryFunctionGroupWeightedAngular.h"
#include <cstddef> // std::size_t
#include <limits> // std::numeric_limits
#include <string> // std::string
#include <vector> // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();
double const accuracyNumeric = 1E-6;

SymmetryFunction* setupSymmetryFunction(ElementMap   em,
                                        size_t const type,
                                        string const setupLine)
{
    SymmetryFunction* sf;
    if      (type ==  2) sf = new SymmetryFunctionRadial(em);
    else if (type ==  3) sf = new SymmetryFunctionAngularNarrow(em);
    else if (type ==  9) sf = new SymmetryFunctionAngularWide(em);
    else if (type == 12) sf = new SymmetryFunctionWeightedRadial(em);
    else if (type == 13) sf = new SymmetryFunctionWeightedAngular(em);
    else
    {
        throw runtime_error("ERROR: Unknown symmetry function type.\n");
    }

    sf->setIndex(0);
    sf->setParameters(setupLine);
    sf->setCutoffFunction(CutoffFunction::CT_TANHU, 0.0);
    string scalingLine = "1 1 0.0 0.0 0.0 0.0";
    sf->setScalingType(SymmetryFunction::ST_NONE, scalingLine, 0.0, 0.0);

    return sf;
}

SymmetryFunctionGroup* setupSymmetryFunctionGroup(ElementMap              em,
                                                  SymmetryFunction const& sf)
{
    SymmetryFunctionGroup* sfg;
    size_t const type = sf.getType();
    if      (type ==  2) sfg = new SymmetryFunctionGroupRadial(em);
    else if (type ==  3) sfg = new SymmetryFunctionGroupAngularNarrow(em);
    else if (type ==  9) sfg = new SymmetryFunctionGroupAngularWide(em);
    else if (type == 12) sfg = new SymmetryFunctionGroupWeightedRadial(em);
    else if (type == 13) sfg = new SymmetryFunctionGroupWeightedAngular(em);
    else
    {
        throw runtime_error("ERROR: Unknown symmetry function type.\n");
    }

    sfg->addMember(&sf);
    sfg->sortMembers();
    sfg->setScalingFactors();

    return sfg;
}

void recalculateSymmetryFunctionGroup(Structure&                   s,
                                      Atom&                        a,
                                      SymmetryFunctionGroup const& sfg)
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
    SymmetryFunction* sf = setupSymmetryFunction(em, type, setupLine);

    SymmetryFunctionGroup* sfg = setupSymmetryFunctionGroup(em, *sf);

    // Allocate symmetry function arrays.
    s.atoms.at(0).numSymmetryFunctions = 1;
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
    SymmetryFunction* sf = setupSymmetryFunction(em, type, setupLine);

    SymmetryFunctionGroup* sfg = setupSymmetryFunctionGroup(em, *sf);

    // Allocate symmetry function arrays.
    s.atoms.at(0).numSymmetryFunctions = 1;
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
