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

void checkGroupOperatorsAndInfo(ElementMap&  em,
                                size_t const type,
                                string const setupLine)
{
    SymFnc* sf1 = setupSymmetryFunction(em, type, setupLine);
    SymFnc* sf2 = setupSymmetryFunction(em, type, setupLine);
    SymGrp* sfg1 = setupSymmetryFunctionGroup(em, *sf1);
    SymGrp* sfg2 = setupSymmetryFunctionGroup(em, *sf2);

    BOOST_REQUIRE(*sfg1 == *sfg2);
    BOOST_REQUIRE(!(*sfg1 <  *sfg2));
    BOOST_REQUIRE(!(*sfg2 <  *sfg1));
    BOOST_REQUIRE(*sfg1 <= *sfg2);

    vector<string> lines = sfg1->parameterLines();
    BOOST_REQUIRE(!lines.empty());

    delete sfg1;
    delete sfg2;
    delete sf1;
    delete sf2;

    return;
}

void checkGroupOperatorsDiffer(ElementMap&  em,
                               size_t const type,
                               string const setupLine1,
                               string const setupLine2)
{
    SymFnc* sf1 = setupSymmetryFunction(em, type, setupLine1);
    SymFnc* sf2 = setupSymmetryFunction(em, type, setupLine2);
    SymGrp* sfg1 = setupSymmetryFunctionGroup(em, *sf1);
    SymGrp* sfg2 = setupSymmetryFunctionGroup(em, *sf2);

    BOOST_REQUIRE(!(*sfg1 == *sfg2));
    bool const lt12 = *sfg1 < *sfg2;
    bool const lt21 = *sfg2 < *sfg1;
    BOOST_REQUIRE(!(lt12 && lt21));

    delete sfg1;
    delete sfg2;
    delete sf1;
    delete sf2;

    return;
}

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_DATA_TEST_CASE_F(FixtureThreeAtomsMono,
                       CheckGroupOperatorsAndInfoMono,
                       bdata::make(typesMono) ^ bdata::make(setupLinesMono),
                       type,
                       setupLine)
{
    checkGroupOperatorsAndInfo(em, type, setupLine);
}

BOOST_AUTO_TEST_CASE(GroupOperatorsDiffer_ByCutoffRadius)
{
    // SymGrpExpRadWeighted/SymGrpExpAngnWeighted::operator== compares
    // ec/type/cutoffType/cutoffAlpha/rc only (member-specific values like
    // eta/rs live on the individual SymFnc members, not the group), so
    // groups differ when rc differs.
    ElementMap em;
    em.registerElements("H");

    checkGroupOperatorsDiffer(em, 12, "H 12 0.001 0.0 10.0",
                                      "H 12 0.001 0.0  8.0");
    // Last token is rc (see SymFncExpAngnWeighted::setParameters); keep
    // eta/rs/lambda/zeta identical and vary only that.
    checkGroupOperatorsDiffer(em, 13, "H 13 0.001  1.0 1.0 10.0 0.0",
                                      "H 13 0.001  1.0 1.0 10.0 2.0");
}

BOOST_AUTO_TEST_CASE(GroupOperatorsEqual_CompactTypesIgnoreMemberParameters)
{
    // SymGrpCompRadWeighted/CompAngnWeighted/CompAngwWeighted::operator==
    // compares only ec/type: for these compact-core group types, rl/rc/
    // subtype/angles distinguish members within a group, not the group
    // itself. This documents/verifies that intentional behavior (as
    // opposed to GroupOperatorsDiffer_ByCutoffRadius above).
    ElementMap em;
    em.registerElements("H");

    SymFnc* sf1 = setupSymmetryFunction(em, 23, "H 23 -10.0 10.0 p2");
    SymFnc* sf2 = setupSymmetryFunction(em, 23, "H 23   0.0  8.0 p3");
    SymGrp* sfg1 = setupSymmetryFunctionGroup(em, *sf1);
    SymGrp* sfg2 = setupSymmetryFunctionGroup(em, *sf2);
    BOOST_REQUIRE(*sfg1 == *sfg2);
    delete sfg1; delete sfg2; delete sf1; delete sf2;

    sf1 = setupSymmetryFunction(em, 24, "H 24 -10.0 10.0  0.0 180.0 p2");
    sf2 = setupSymmetryFunction(em, 24, "H 24   0.0 10.0 10.0  60.0 p3");
    sfg1 = setupSymmetryFunctionGroup(em, *sf1);
    sfg2 = setupSymmetryFunctionGroup(em, *sf2);
    BOOST_REQUIRE(*sfg1 == *sfg2);
    delete sfg1; delete sfg2; delete sf1; delete sf2;

    sf1 = setupSymmetryFunction(em, 25, "H 25 -10.0 10.0  0.0 180.0 p2");
    sf2 = setupSymmetryFunction(em, 25, "H 25   0.0 10.0 10.0  60.0 p3");
    sfg1 = setupSymmetryFunctionGroup(em, *sf1);
    sfg2 = setupSymmetryFunctionGroup(em, *sf2);
    BOOST_REQUIRE(*sfg1 == *sfg2);
    delete sfg1; delete sfg2; delete sf1; delete sf2;
}

// Every other test in this suite only ever puts a single SymFnc into a
// SymGrp, so the "shared computation across members" optimizations inside
// SymGrp*::calculate() (calculateExp[]/calculateComp[] deciding whether an
// expensive radial term is recomputed or reused from the previous member)
// are never actually exercised -- with one member, that branch is always
// taken. Build genuine 2-member groups (sharing exactly the parameters the
// optimization keys on, differing in the parameters that make them
// distinct symmetry functions) and verify the grouped result exactly
// matches independently/ungrouped-computed values.
SymGrp* setupSymmetryFunctionGroupMulti(ElementMap em,
                                        vector<SymFnc const*> const& sfs)
{
    SymGrp* sfg;
    size_t const type = sfs.at(0)->getType();
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

    for (auto const* sf : sfs) sfg->addMember(sf);
    sfg->sortMembers();
    sfg->setScalingFactors();

    return sfg;
}

void checkMultiMemberGroupMatchesIndividual(Structure&   s,
                                            ElementMap&  em,
                                            size_t const type,
                                            string const line1,
                                            string const line2)
{
    size_t const nSF = 2;

    SymFnc* sf1 = setupSymmetryFunction(em, type, line1);
    SymFnc* sf2 = setupSymmetryFunction(em, type, line2);
    // setupSymmetryFunction() hardcodes index 0; give the second member its
    // own G/dGdr slot.
    sf2->setIndex(1);
    for (size_t i = 0; i < em.size(); ++i) sf2->setIndexPerElement(i, 1);

    s.atoms.at(0).numSymmetryFunctions = nSF;
    s.atoms.at(0).numSymmetryFunctionDerivatives =
        vector<size_t>(em.size(), nSF);
#ifndef NOSFCACHE
    s.atoms.at(0).cacheSizePerElement =
        vector<size_t>(maxElements, maxCacheSize);
#endif
    s.clearNeighborList();
    s.calculateNeighborList(10.0);
    s.atoms.at(0).allocate(true);

    // Reference: each member computed individually (ungrouped).
    sf1->calculate(s.atoms.at(0), true);
    sf2->calculate(s.atoms.at(0), true);
    vector<double> const refG = s.atoms.at(0).G;
    vector<Vec3D> const refDGdr = s.atoms.at(0).dGdr;
    vector<vector<Vec3D> > refNeighborDGdr;
    for (auto const& n : s.atoms.at(0).neighbors)
    {
        refNeighborDGdr.push_back(n.dGdr);
    }

    // Reset, then compute via a single 2-member group (exercises the
    // calculateExp[]/calculateComp[] reuse optimization for member index 1).
    s.clearNeighborList();
    s.calculateNeighborList(10.0);
    s.atoms.at(0).allocate(true);

    SymGrp* sfg = setupSymmetryFunctionGroupMulti(em, {sf1, sf2});
    sfg->calculate(s.atoms.at(0), true);

    for (size_t i = 0; i < nSF; ++i)
    {
        BOOST_TEST_INFO("G[" << i << "]");
        BOOST_REQUIRE_SMALL(s.atoms.at(0).G.at(i) - refG.at(i), accuracy);
        BOOST_TEST_INFO("dGdr[" << i << "]");
        BOOST_REQUIRE_SMALL(
            (s.atoms.at(0).dGdr.at(i) - refDGdr.at(i)).norm2(), accuracy);
    }
    for (size_t j = 0; j < s.atoms.at(0).neighbors.size(); ++j)
    {
        auto const& n = s.atoms.at(0).neighbors.at(j);
        for (size_t i = 0; i < nSF; ++i)
        {
            BOOST_TEST_INFO("neighbor " << j << " dGdr[" << i << "]");
            BOOST_REQUIRE_SMALL(
                (n.dGdr.at(i) - refNeighborDGdr.at(j).at(i)).norm2(),
                accuracy);
        }
    }

    delete sfg;
    delete sf1;
    delete sf2;

    return;
}

BOOST_FIXTURE_TEST_CASE(MultiMemberGroup_ExpAngnWeighted_MatchesIndividual,
                        FixtureThreeAtomsMono)
{
    // Members share cutoffType/cutoffAlpha/rc/eta/rs (the fields
    // calculateExp[] keys on), differing only in lambda/zeta.
    checkMultiMemberGroupMatchesIndividual(
        s, em, 13,
        "H 13 0.01  0.5  1.0 4.0 8.0",
        "H 13 0.01  0.5 -1.0 2.0 8.0");
}

BOOST_FIXTURE_TEST_CASE(MultiMemberGroup_CompAngnWeighted_MatchesIndividual,
                        FixtureThreeAtomsMono)
{
    // Members share subtype/rl/rc (the fields calculateComp[] keys on),
    // differing only in the angular range.
    checkMultiMemberGroupMatchesIndividual(
        s, em, 24,
        "H 24 -10.0 10.0  0.0  90.0 p2",
        "H 24 -10.0 10.0 90.0 180.0 p2");
}

BOOST_FIXTURE_TEST_CASE(MultiMemberGroup_CompAngwWeighted_MatchesIndividual,
                        FixtureThreeAtomsMono)
{
    checkMultiMemberGroupMatchesIndividual(
        s, em, 25,
        "H 25 -10.0 10.0  0.0  90.0 p2",
        "H 25 -10.0 10.0 90.0 180.0 p2");
}

BOOST_AUTO_TEST_SUITE_END()

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
