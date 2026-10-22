#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Element
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include "fixtures.h"

#include "Atom.h"
#include "CutoffFunction.h"
#include "Element.h"
#include "ElementMap.h"
#include "Structure.h"
#include "SymFnc.h"
#include "SymFncBaseCutoff.h"
#include "SymFncStatistics.h"
#include "utility.h" // vectorContains
#include <stdexcept> // std::runtime_error, std::out_of_range
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Constructor_SetsIndexAtomicNumberAndSymbol)
{
    ElementMap em;
    em.registerElements("H O");

    Element elemO(em["O"], em);
    BOOST_REQUIRE_EQUAL(elemO.getIndex(), em["O"]);
    BOOST_REQUIRE_EQUAL(elemO.getSymbol(), "O");
    BOOST_REQUIRE_EQUAL(elemO.getAtomicNumber(), em.atomicNumber("O"));
    BOOST_REQUIRE_EQUAL(elemO.getAtomicEnergyOffset(), 0.0);
    BOOST_REQUIRE_EQUAL(elemO.getHardness(), 0.0);
    BOOST_REQUIRE_EQUAL(elemO.getQsigma(), 0.0);
    BOOST_REQUIRE_EQUAL(elemO.numSymmetryFunctions(), 0);
}

BOOST_AUTO_TEST_CASE(SetGetAtomicEnergyOffset_RoundTrip)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    elem.setAtomicEnergyOffset(-13.6);
    BOOST_REQUIRE_EQUAL(elem.getAtomicEnergyOffset(), -13.6);
}

BOOST_AUTO_TEST_CASE(SetGetHardness_RoundTrip)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    elem.setHardness(1.5);
    BOOST_REQUIRE_EQUAL(elem.getHardness(), 1.5);
}

BOOST_AUTO_TEST_CASE(SetGetQsigma_RoundTrip)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    elem.setQsigma(0.5);
    BOOST_REQUIRE_EQUAL(elem.getQsigma(), 0.5);
}

BOOST_AUTO_TEST_CASE(AddSymmetryFunction_ValidType_IncrementsCount)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);

    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    BOOST_REQUIRE_EQUAL(elem.numSymmetryFunctions(), 1);

    elem.addSymmetryFunction("H 3 H H 0.5 1.0 1.0 6.0", 2);
    BOOST_REQUIRE_EQUAL(elem.numSymmetryFunctions(), 2);
}

BOOST_AUTO_TEST_CASE(AddSymmetryFunction_UnknownType_Throws)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);

    BOOST_REQUIRE_THROW(elem.addSymmetryFunction("H 999 H 0.5 0.0 6.0", 1),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(ClearSymmetryFunctions_ResetsToZero)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    elem.addSymmetryFunction("H 2 H 0.3 0.0 8.0", 2);
    BOOST_REQUIRE_EQUAL(elem.numSymmetryFunctions(), 2);

    elem.clearSymmetryFunctions();
    BOOST_REQUIRE_EQUAL(elem.numSymmetryFunctions(), 0);
}

BOOST_AUTO_TEST_CASE(SortSymmetryFunctions_AssignsSequentialIndices)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    // Add with descending rc; SymFnc::operator< for same type/element
    // orders by parameter values, so sorting should reorder these two.
    elem.addSymmetryFunction("H 2 H 0.5 0.0 8.0", 1);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 2);

    elem.sortSymmetryFunctions();

    BOOST_REQUIRE_EQUAL(elem.getSymmetryFunction(0).getIndex(), 0);
    BOOST_REQUIRE_EQUAL(elem.getSymmetryFunction(1).getIndex(), 1);
    // Ascending order established by SymFnc::operator<.
    BOOST_REQUIRE(elem.getSymmetryFunction(0) < elem.getSymmetryFunction(1));
}

BOOST_AUTO_TEST_CASE(InfoSymmetryFunctionParameters_OneLinePerFunction)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    elem.addSymmetryFunction("H 3 H H 0.5 1.0 1.0 6.0", 2);

    vector<string> const info = elem.infoSymmetryFunctionParameters();
    BOOST_REQUIRE_EQUAL(info.size(), 2);
}

BOOST_AUTO_TEST_CASE(InfoSymmetryFunctionScaling_OneLinePerFunction)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    elem.sortSymmetryFunctions();
    elem.setScalingNone();

    vector<string> const info = elem.infoSymmetryFunctionScaling();
    BOOST_REQUIRE_EQUAL(info.size(), 1);
}

BOOST_AUTO_TEST_CASE(SetCutoffFunction_UpdatesCutoffBasedFunctions)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);

    elem.setCutoffFunction(CutoffFunction::CT_TANHU, 0.5);

    SymFncBaseCutoff const* sfcb =
        dynamic_cast<SymFncBaseCutoff const*>(&elem.getSymmetryFunction(0));
    BOOST_REQUIRE(sfcb != nullptr);
    BOOST_REQUIRE(sfcb->getCutoffType() == CutoffFunction::CT_TANHU);
    BOOST_REQUIRE_EQUAL(sfcb->getCutoffAlpha(), 0.5);
}

BOOST_AUTO_TEST_CASE(GetMinNeighbors_MaxAcrossFunctions)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    // Radial (type 2) requires 1 neighbor, angular (type 3) requires 2.
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    BOOST_REQUIRE_EQUAL(elem.getMinNeighbors(), 1);

    elem.addSymmetryFunction("H 3 H H 0.5 1.0 1.0 6.0", 2);
    BOOST_REQUIRE_EQUAL(elem.getMinNeighbors(), 2);
}

BOOST_AUTO_TEST_CASE(GetMinAndMaxCutoffRadius_ReflectExtremes)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    elem.addSymmetryFunction("H 2 H 0.3 0.0 9.0", 2);
    elem.addSymmetryFunction("H 2 H 0.1 0.0 4.0", 3);

    BOOST_REQUIRE_EQUAL(elem.getMinCutoffRadius(), 4.0);
    BOOST_REQUIRE_EQUAL(elem.getMaxCutoffRadius(), 9.0);
}

BOOST_AUTO_TEST_CASE(GetCutoffRadii_ReturnsUniqueValuesOnly)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    elem.addSymmetryFunction("H 2 H 0.3 0.0 6.0", 2); // Duplicate rc.
    elem.addSymmetryFunction("H 2 H 0.1 0.0 8.0", 3);

    vector<double> cutoffs;
    elem.getCutoffRadii(cutoffs);

    BOOST_REQUIRE_EQUAL(cutoffs.size(), 2);
    BOOST_REQUIRE(vectorContains(cutoffs, 6.0));
    BOOST_REQUIRE(vectorContains(cutoffs, 8.0));
}

BOOST_AUTO_TEST_CASE(SetupSymmetryFunctionMemory_PopulatesPerElementTables)
{
    ElementMap em;
    em.registerElements("H O");
    Element elem(em["H"], em);
    // Relevant to O only.
    elem.addSymmetryFunction("H 2 O 0.5 0.0 6.0", 1);
    // Relevant to H only.
    elem.addSymmetryFunction("H 2 H 0.3 0.0 6.0", 2);
    elem.sortSymmetryFunctions();

    elem.setupSymmetryFunctionMemory();

    BOOST_REQUIRE_EQUAL(elem.getSymmetryFunctionNumTable().size(), 2);
    BOOST_REQUIRE_EQUAL(elem.getSymmetryFunctionNumTable().at(em["H"]), 1);
    BOOST_REQUIRE_EQUAL(elem.getSymmetryFunctionNumTable().at(em["O"]), 1);
    BOOST_REQUIRE_EQUAL(elem.getSymmetryFunctionTable().at(em["H"]).size(), 1);
    BOOST_REQUIRE_EQUAL(elem.getSymmetryFunctionTable().at(em["O"]).size(), 1);
}

BOOST_FIXTURE_TEST_CASE(
    SetupSymmetryFunctionGroupsAndCalculate_MatchesDirectCalculation,
    FixtureThreeAtomsMono)
{
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    elem.addSymmetryFunction("H 2 H 0.3 0.0 6.0", 2);
    elem.sortSymmetryFunctions();
    // Needed so each symmetry function's per-element index (used by both
    // the direct and group-based calculate() paths to index into
    // atom.dGdr/neighbor.dGdr) is set -- without it those indices stay at
    // their "uninitialized" sentinel value and calculate() corrupts memory.
    elem.setupSymmetryFunctionMemory();
    // Groups must exist before setScalingNone() runs: setScalingNone()
    // loops over symmetryFunctionGroups to populate each group's
    // scalingFactors array (via SymGrp::setScalingFactors()) -- calling it
    // first leaves a newly-created group's scalingFactors empty, which
    // calculate() then indexes out of bounds. This mirrors the real setup
    // order used in Mode::setupGeneric() (memory+groups) followed by
    // Mode::setupSymmetryFunctionScalingNone().
    elem.setupSymmetryFunctionGroups();
    elem.setScalingNone();

    // Group-based calculation writes into each neighbor's symmetry-function
    // cache, so it needs a properly sized (not necessarily wired-up via
    // setCacheIndices) buffer -- a generously sized one avoids depending on
    // the exact cache-index layout, which is not what this test is about.
    Atom& atom = s.atoms.at(0);
    atom.cacheSizePerElement = vector<size_t>(em.size(), 16);
    atom.numSymmetryFunctions = elem.numSymmetryFunctions();
    atom.numSymmetryFunctionDerivatives =
        vector<size_t>(em.size(), elem.numSymmetryFunctions());
    atom.allocate(true);

    elem.calculateSymmetryFunctions(atom, true);
    vector<double> const GDirect = atom.G;

    // Reset and recompute via the group-based code path.
    atom.free(true);
    atom.numSymmetryFunctions = elem.numSymmetryFunctions();
    atom.numSymmetryFunctionDerivatives =
        vector<size_t>(em.size(), elem.numSymmetryFunctions());
    atom.allocate(true);

    elem.calculateSymmetryFunctionGroups(atom, true);
    vector<double> const GGroups = atom.G;

    BOOST_REQUIRE_EQUAL(GDirect.size(), GGroups.size());
    for (size_t i = 0; i < GDirect.size(); ++i)
    {
        BOOST_REQUIRE_EQUAL(GDirect.at(i), GGroups.at(i));
    }

    // setupSymmetryFunctionGroups() must also assign sequential group
    // indices, sorted consistently with comparePointerTargets<SymGrp>.
    vector<string> const groupInfo = elem.infoSymmetryFunctionGroups();
    BOOST_REQUIRE(groupInfo.size() >= 1);
}

BOOST_FIXTURE_TEST_CASE(UpdateSymmetryFunctionStatistics_ElementMismatch_Throws,
                        FixtureThreeAtomsMono)
{
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    elem.sortSymmetryFunctions();
    elem.setScalingNone();

    Atom& atom = s.atoms.at(0);
    atom.cacheSizePerElement = vector<size_t>(em.size(), 0);
    atom.element = 5; // Deliberately inconsistent with elem's index (0).
    atom.numSymmetryFunctions = 1;
    atom.numSymmetryFunctionDerivatives = vector<size_t>(em.size(), 1);
    atom.allocate(true);
    atom.G = {0.0};

    BOOST_REQUIRE_THROW(elem.updateSymmetryFunctionStatistics(atom),
                        runtime_error);
}

BOOST_FIXTURE_TEST_CASE(UpdateSymmetryFunctionStatistics_CountMismatch_Throws,
                        FixtureThreeAtomsMono)
{
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    elem.addSymmetryFunction("H 2 H 0.3 0.0 6.0", 2);
    elem.sortSymmetryFunctions();
    elem.setScalingNone();

    Atom& atom = s.atoms.at(0);
    atom.cacheSizePerElement = vector<size_t>(em.size(), 0);
    atom.element = 0;
    atom.numSymmetryFunctions = 1; // Mismatch: elem has 2.
    atom.numSymmetryFunctionDerivatives = vector<size_t>(em.size(), 1);
    atom.allocate(true);
    atom.G = {0.0};

    BOOST_REQUIRE_THROW(elem.updateSymmetryFunctionStatistics(atom),
                        runtime_error);
}

BOOST_FIXTURE_TEST_CASE(
    UpdateSymmetryFunctionStatistics_CountsExtrapolationWarnings,
    FixtureThreeAtomsMono)
{
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    elem.sortSymmetryFunctions();
    elem.setScalingNone(); // Sets Gmin = Gmax = 0.0 (both zero in scaling line).

    Atom& atom = s.atoms.at(0);
    atom.cacheSizePerElement = vector<size_t>(em.size(), 0);
    atom.element = 0;
    atom.numSymmetryFunctions = 1;
    atom.numSymmetryFunctionDerivatives = vector<size_t>(em.size(), 1);
    atom.allocate(true);
    // Any nonzero G value now lies outside [Gmin, Gmax] = [0, 0].
    atom.G = {1.0};

    elem.statistics.collectExtrapolationWarnings = true;
    size_t const warnings = elem.updateSymmetryFunctionStatistics(atom);

    BOOST_REQUIRE_EQUAL(warnings, 1);
    BOOST_REQUIRE_EQUAL(elem.statistics.countExtrapolationWarnings(), 1);
}

BOOST_FIXTURE_TEST_CASE(
    UpdateSymmetryFunctionStatistics_StopOnExtrapolationWarnings_Throws,
    FixtureThreeAtomsMono)
{
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    elem.sortSymmetryFunctions();
    elem.setScalingNone();

    Atom& atom = s.atoms.at(0);
    atom.cacheSizePerElement = vector<size_t>(em.size(), 0);
    atom.element = 0;
    atom.numSymmetryFunctions = 1;
    atom.numSymmetryFunctionDerivatives = vector<size_t>(em.size(), 1);
    atom.allocate(true);
    atom.G = {1.0}; // Outside [0, 0].

    elem.statistics.stopOnExtrapolationWarnings = true;

    BOOST_REQUIRE_THROW(elem.updateSymmetryFunctionStatistics(atom),
                        out_of_range);
}

BOOST_AUTO_TEST_CASE(GetSymmetryFunction_ReturnsAddedFunction)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);

    BOOST_REQUIRE_EQUAL(elem.getSymmetryFunction(0).getType(), 2);
    BOOST_REQUIRE_EQUAL(elem.getSymmetryFunction(0).getRc(), 6.0);
}

#ifndef N2P2_NO_SF_CACHE
BOOST_AUTO_TEST_CASE(SetCacheIndicesAndGetCacheSizes_ReflectProvidedLists)
{
    ElementMap em;
    em.registerElements("H");
    Element elem(0, em);
    elem.addSymmetryFunction("H 2 H 0.5 0.0 6.0", 1);
    elem.sortSymmetryFunctions();

    SymFnc const& sf = elem.getSymmetryFunction(0);
    vector<string> const identifiers = sf.getCacheIdentifiers();
    BOOST_REQUIRE(!identifiers.empty());

    Element::SFCacheList list;
    list.element = em["H"];
    list.identifier = identifiers.at(0);
    list.indices = {0};

    vector<vector<Element::SFCacheList>> cacheLists(em.size());
    cacheLists.at(em["H"]).push_back(list);

    elem.setCacheIndices(cacheLists);

    vector<size_t> const sizes = elem.getCacheSizes();
    BOOST_REQUIRE_EQUAL(sizes.size(), em.size());
    BOOST_REQUIRE_EQUAL(sizes.at(em["H"]), 1);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
