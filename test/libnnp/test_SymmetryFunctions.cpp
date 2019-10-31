#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SymmetryFunctions
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "Atom.h"
#include "ElementMap.h"
#include "Structure.h"
#include "SymmetryFunction.h"
#include "SymmetryFunctionRadial.h"
#include "SymmetryFunctionAngularNarrow.h"
#include "SymmetryFunctionAngularWide.h"
#include <cstddef> // std::size_t
#include <limits> // std::numeric_limits
#include <string> // std::string
#include <vector> // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 10.0 * numeric_limits<double>::epsilon(); 
double const accuracyNumeric = 1E-9; 

struct FixtureThreeAtoms
{
    FixtureThreeAtoms()
    {
        em.registerElements("H");
        s.setElementMap(em);

        // Atom 1.
        s.atoms.push_back(Atom());
        s.atoms.back().numNeighborsPerElement.resize(em.size());
        s.numAtoms++;
        s.numAtomsPerElement[0]++;

        // Atom 2.
        s.atoms.push_back(Atom());
        s.atoms.back().r[0] = 1.0;
        s.atoms.back().r[1] = 2.0;
        s.atoms.back().r[2] = 1.0;
        s.atoms.back().index = 1;
        s.atoms.back().numNeighborsPerElement.resize(em.size());
        s.numAtoms++;
        s.numAtomsPerElement[0]++;

        // Atom 3.
        s.atoms.push_back(Atom());
        s.atoms.back().r[0] = 0.0;
        s.atoms.back().r[1] = 3.0;
        s.atoms.back().r[2] = -1.0;
        s.atoms.back().index = 2;
        s.atoms.back().numNeighborsPerElement.resize(em.size());
        s.numAtoms++;
        s.numAtomsPerElement[0]++;

        // Calculate neighbor list.
        s.calculateNeighborList(10.0);
    }
    ~FixtureThreeAtoms() {};

    ElementMap em;
    Structure s;
};

vector<size_t> const types = {2, 3, 9}; 
vector<string> const setupLines = {"H 2 H 0.001 0.0 10.0",
                                   "H 3 H H 0.001 1.0 1.0 10.0 1.0",
                                   "H 9 H H 0.001 1.0 1.0 10.0 1.2"};

void recalculateSymmetryFunction(Structure&              s,
                                 Atom&                   a,
                                 SymmetryFunction const& sf)
{
    s.clearNeighborList();
    s.calculateNeighborList(10.0);
    a.allocate(true);
    sf.calculate(a, true);

    return;
}


BOOST_AUTO_TEST_SUITE(IntegrationTests)

BOOST_DATA_TEST_CASE_F(FixtureThreeAtoms,
                       CompareAnalyticNumericDerivatives_EqualResults,
                       bdata::make(types) ^ bdata::make(setupLines),
                       type,
                       setupLine)
{
    SymmetryFunction* sf;
    if      (type == 2) sf = new SymmetryFunctionRadial(em);
    else if (type == 3) sf = new SymmetryFunctionAngularNarrow(em);
    else if (type == 9) sf = new SymmetryFunctionAngularWide(em);
    else
    {
        throw runtime_error("ERROR: Unknown symmetry function type.\n");
    }

    sf->setIndex(0);
    sf->setParameters(setupLine);
    sf->setCutoffFunction(CutoffFunction::CT_TANHU, 0.0);
    string scalingLine = "1 1 0.0 0.0 0.0 0.0";
    sf->setScalingType(SymmetryFunction::ST_NONE, scalingLine, 0.0, 0.0);

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

    for (size_t ic = 0; ic < 3; ++ic)
    {
        BOOST_REQUIRE_SMALL(s.atoms.at(0).dGdr.at(0)[ic] - results.at(0).at(ic),
                          accuracyNumeric);
    }
    for (size_t ia = 1; ia < s.numAtoms; ++ia)
    {
        for (size_t ic = 0; ic < 3; ++ic)
        {
            BOOST_REQUIRE_SMALL(s.atoms.at(0).neighbors.at(ia-1).dGdr.at(0)[ic] - results.at(ia).at(ic),
                                accuracyNumeric);
        }
    }

    delete sf;
}


BOOST_AUTO_TEST_SUITE_END()
