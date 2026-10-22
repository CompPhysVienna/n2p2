#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Structure
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/filesystem.hpp>
#include "fileHelpers.h"

#include "Atom.h"
#include "ElementMap.h"
#include "EwaldSetup.h"
#include "Structure.h"
#include "Vec3D.h"
#include <cmath>     // std::fabs
#include <limits>    // std::numeric_limits
#include <map>       // std::map
#include <stdexcept> // std::runtime_error
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;
namespace bfs = boost::filesystem;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();
double const accuracyNumeric = 1E-10;

Structure makeCubicStructure(ElementMap const& em, double const boxLength)
{
    Structure s;
    s.setElementMap(em);
    s.isPeriodic = true;
    s.box[0] = Vec3D(boxLength, 0.0, 0.0);
    s.box[1] = Vec3D(0.0, boxLength, 0.0);
    s.box[2] = Vec3D(0.0, 0.0, boxLength);
    s.calculateInverseBox();
    s.calculateVolume();

    return s;
}

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Constructor_Defaults)
{
    Structure s;
    BOOST_REQUIRE_EQUAL(s.isPeriodic, false);
    BOOST_REQUIRE_EQUAL(s.isTriclinic, false);
    BOOST_REQUIRE_EQUAL(s.hasNeighborList, false);
    BOOST_REQUIRE_EQUAL(s.numAtoms, 0);
    BOOST_REQUIRE_EQUAL(s.numElements, 0);
    BOOST_REQUIRE_SMALL(s.energy, accuracy);
    BOOST_REQUIRE_SMALL(s.volume, accuracy);
    BOOST_REQUIRE_EQUAL(s.sampleType, Structure::ST_UNKNOWN);
    BOOST_REQUIRE_EQUAL(s.comment, "");
    for (size_t i = 0; i < 3; ++i)
    {
        BOOST_REQUIRE_EQUAL(s.pbc[i], 0);
    }
}

BOOST_AUTO_TEST_CASE(SetElementMap_SetsNumElementsAndResizesAtomsPerElement)
{
    ElementMap em;
    em.registerElements("H O");
    Structure s;
    s.setElementMap(em);

    BOOST_REQUIRE_EQUAL(s.numElements, 2);
    BOOST_REQUIRE_EQUAL(s.numAtomsPerElement.size(), 2);
}

BOOST_AUTO_TEST_CASE(AddAtom_IncrementsCountsAndSetsIndices)
{
    ElementMap em;
    em.registerElements("H O");
    Structure s;
    s.setElementMap(em);
    s.index = 3;

    Atom a1;
    a1.r = Vec3D(0.0, 0.0, 0.0);
    s.addAtom(a1, "H");
    Atom a2;
    a2.r = Vec3D(1.0, 0.0, 0.0);
    s.addAtom(a2, "O");

    BOOST_REQUIRE_EQUAL(s.numAtoms, 2);
    BOOST_REQUIRE_EQUAL(s.atoms.size(), 2);
    BOOST_REQUIRE_EQUAL(s.atoms.at(0).index, 0);
    BOOST_REQUIRE_EQUAL(s.atoms.at(1).index, 1);
    BOOST_REQUIRE_EQUAL(s.atoms.at(0).indexStructure, 3);
    BOOST_REQUIRE_EQUAL(s.atoms.at(0).element, em.index("H"));
    BOOST_REQUIRE_EQUAL(s.atoms.at(1).element, em.index("O"));
    BOOST_REQUIRE_EQUAL(s.numAtomsPerElement.at(em.index("H")), 1);
    BOOST_REQUIRE_EQUAL(s.numAtomsPerElement.at(em.index("O")), 1);
}

BOOST_AUTO_TEST_CASE(CalculateVolume_OrthorhombicBox)
{
    Structure s;
    s.box[0] = Vec3D(2.0, 0.0, 0.0);
    s.box[1] = Vec3D(0.0, 3.0, 0.0);
    s.box[2] = Vec3D(0.0, 0.0, 4.0);
    s.calculateVolume();

    BOOST_REQUIRE_SMALL(s.volume - 24.0, accuracy);
}

BOOST_AUTO_TEST_CASE(CalculateVolume_TriclinicBox)
{
    Structure s;
    s.box[0] = Vec3D(2.0, 0.0, 0.0);
    s.box[1] = Vec3D(1.0, 3.0, 0.0);
    s.box[2] = Vec3D(0.0, 1.0, 4.0);
    s.calculateVolume();

    // V = a . (b x c)
    Vec3D const bxc = s.box[1].cross(s.box[2]);
    double const expected = fabs(s.box[0] * bxc);
    BOOST_REQUIRE_SMALL(s.volume - expected, accuracy);
}

BOOST_AUTO_TEST_CASE(CalculateInverseBox_CubicBox_DiagonalInverse)
{
    Structure s;
    s.box[0] = Vec3D(10.0, 0.0, 0.0);
    s.box[1] = Vec3D(0.0, 10.0, 0.0);
    s.box[2] = Vec3D(0.0, 0.0, 10.0);
    s.calculateInverseBox();

    BOOST_REQUIRE_SMALL(s.invbox[0][0] - 0.1, accuracy);
    BOOST_REQUIRE_SMALL(s.invbox[1][1] - 0.1, accuracy);
    BOOST_REQUIRE_SMALL(s.invbox[2][2] - 0.1, accuracy);
    BOOST_REQUIRE_SMALL(s.invbox[0][1], accuracy);
    BOOST_REQUIRE_SMALL(s.invbox[0][2], accuracy);
    BOOST_REQUIRE_SMALL(s.invbox[1][0], accuracy);
}

BOOST_AUTO_TEST_CASE(CalculateInverseBox_TriclinicBox_RoundTrip)
{
    Structure s;
    s.box[0] = Vec3D(5.0, 0.0, 0.0);
    s.box[1] = Vec3D(1.0, 6.0, 0.0);
    s.box[2] = Vec3D(2.0, 1.0, 7.0);
    s.calculateInverseBox();

    // invbox * box[0] should give fractional coordinates (1, 0, 0), i.e.
    // h^-1 * h is the identity (using the same "matrix * vector" operator
    // applyMinimumImageConvention() itself relies on).
    Vec3D const f0 = s.invbox * s.box[0];
    BOOST_REQUIRE_SMALL(f0[0] - 1.0, accuracyNumeric);
    BOOST_REQUIRE_SMALL(f0[1], accuracyNumeric);
    BOOST_REQUIRE_SMALL(f0[2], accuracyNumeric);
}

BOOST_AUTO_TEST_CASE(CanMinimumImageConventionBeApplied_CubicBox)
{
    Structure s;
    s.box[0] = Vec3D(10.0, 0.0, 0.0);
    s.box[1] = Vec3D(0.0, 10.0, 0.0);
    s.box[2] = Vec3D(0.0, 0.0, 10.0);

    BOOST_REQUIRE(s.canMinimumImageConventionBeApplied(4.9));
    BOOST_REQUIRE(!s.canMinimumImageConventionBeApplied(5.1));
}

BOOST_AUTO_TEST_CASE(ApplyMinimumImageConvention_CubicBox_WrapsToNearestImage)
{
    ElementMap em;
    em.registerElements("H");
    Structure s = makeCubicStructure(em, 10.0);

    Vec3D const dr(7.0, 0.0, 0.0);
    Vec3D const drMin = s.applyMinimumImageConvention(dr);

    // 7.0 along a 10.0-periodic box wraps to -3.0 (shorter image).
    BOOST_REQUIRE_SMALL(drMin[0] - (-3.0), accuracyNumeric);
    BOOST_REQUIRE_SMALL(drMin[1], accuracyNumeric);
    BOOST_REQUIRE_SMALL(drMin[2], accuracyNumeric);
}

BOOST_AUTO_TEST_CASE(Remap_AtomOutsideBox_WrapsInside)
{
    ElementMap em;
    em.registerElements("H");
    Structure s = makeCubicStructure(em, 10.0);

    Atom a;
    a.r = Vec3D(12.0, -3.0, 5.0);
    s.remap(a);

    BOOST_REQUIRE_SMALL(a.r[0] - 2.0, accuracyNumeric);
    BOOST_REQUIRE_SMALL(a.r[1] - 7.0, accuracyNumeric);
    BOOST_REQUIRE_SMALL(a.r[2] - 5.0, accuracyNumeric);
}

BOOST_AUTO_TEST_CASE(Remap_AllAtoms_WrapsEachOne)
{
    ElementMap em;
    em.registerElements("H");
    Structure s = makeCubicStructure(em, 10.0);

    Atom a1;
    a1.r = Vec3D(11.0, 0.0, 0.0);
    s.addAtom(a1, "H");
    Atom a2;
    a2.r = Vec3D(-1.0, 0.0, 0.0);
    s.addAtom(a2, "H");

    s.remap();

    BOOST_REQUIRE_SMALL(s.atoms.at(0).r[0] - 1.0, accuracyNumeric);
    BOOST_REQUIRE_SMALL(s.atoms.at(1).r[0] - 9.0, accuracyNumeric);
}

BOOST_AUTO_TEST_CASE(ToNormalizedToPhysicalUnits_RoundTrip)
{
    ElementMap em;
    em.registerElements("H");
    Structure s = makeCubicStructure(em, 10.0);
    s.numAtoms = 2;
    s.energy = 5.0;
    s.energyRef = 4.0;
    s.charge = 0.2;
    s.chargeRef = -0.2;

    double const meanEnergy = 1.0;
    double const convEnergy = 2.0;
    double const convLength = 3.0;
    double const convCharge = 4.0;

    double const volumeBefore = s.volume;
    Vec3D const box0Before = s.box[0];

    s.toNormalizedUnits(meanEnergy, convEnergy, convLength, convCharge);
    BOOST_REQUIRE(fabs(s.volume - volumeBefore) > accuracy);

    s.toPhysicalUnits(meanEnergy, convEnergy, convLength, convCharge);

    BOOST_REQUIRE_SMALL(s.volume - volumeBefore, accuracyNumeric);
    BOOST_REQUIRE_SMALL((s.box[0] - box0Before).norm2(), accuracyNumeric);
    BOOST_REQUIRE_SMALL(s.energy - 5.0, accuracyNumeric);
    BOOST_REQUIRE_SMALL(s.energyRef - 4.0, accuracyNumeric);
    BOOST_REQUIRE_SMALL(s.charge - 0.2, accuracyNumeric);
    BOOST_REQUIRE_SMALL(s.chargeRef - (-0.2), accuracyNumeric);
}

BOOST_AUTO_TEST_CASE(CalculateNeighborList_NonPeriodic_CorrectNeighbors)
{
    ElementMap em;
    em.registerElements("H");
    Structure s;
    s.setElementMap(em);

    Atom a0;
    a0.r = Vec3D(0.0, 0.0, 0.0);
    s.addAtom(a0, "H");
    Atom a1;
    a1.r = Vec3D(1.0, 0.0, 0.0);
    s.addAtom(a1, "H");
    Atom a2;
    a2.r = Vec3D(5.0, 0.0, 0.0);
    s.addAtom(a2, "H");

    s.calculateNeighborList(2.0);

    BOOST_REQUIRE_EQUAL(s.atoms.at(0).numNeighbors, 1);
    BOOST_REQUIRE_EQUAL(s.atoms.at(0).neighbors.at(0).index, 1);
    BOOST_REQUIRE_SMALL(s.atoms.at(0).neighbors.at(0).d - 1.0, accuracyNumeric);
    BOOST_REQUIRE_EQUAL(s.atoms.at(2).numNeighbors, 0);
    BOOST_REQUIRE(s.hasNeighborList);
}

BOOST_AUTO_TEST_CASE(CalculateNeighborList_Periodic_FindsImageAcrossBoundary)
{
    ElementMap em;
    em.registerElements("H");
    Structure s = makeCubicStructure(em, 10.0);

    Atom a0;
    a0.r = Vec3D(0.5, 0.0, 0.0);
    s.addAtom(a0, "H");
    Atom a1;
    a1.r = Vec3D(9.5, 0.0, 0.0);
    s.addAtom(a1, "H");

    // Direct distance is 9.0, but the periodic image is only 1.0 away.
    s.calculateNeighborList(2.0);

    BOOST_REQUIRE_EQUAL(s.atoms.at(0).numNeighbors, 1);
    BOOST_REQUIRE_SMALL(s.atoms.at(0).neighbors.at(0).d - 1.0, accuracyNumeric);
}

BOOST_AUTO_TEST_CASE(SortNeighborList_OrdersByAscendingDistance)
{
    ElementMap em;
    em.registerElements("H");
    Structure s;
    s.setElementMap(em);

    Atom a0;
    a0.r = Vec3D(0.0, 0.0, 0.0);
    s.addAtom(a0, "H");
    Atom a1;
    a1.r = Vec3D(3.0, 0.0, 0.0);
    s.addAtom(a1, "H");
    Atom a2;
    a2.r = Vec3D(1.0, 0.0, 0.0);
    s.addAtom(a2, "H");

    s.calculateNeighborList(10.0, true);

    BOOST_REQUIRE(s.NeighborListIsSorted);
    auto const& n = s.atoms.at(0).neighbors;
    for (size_t i = 1; i < n.size(); ++i)
    {
        BOOST_REQUIRE(n.at(i - 1).d <= n.at(i).d);
    }
}

BOOST_AUTO_TEST_CASE(GetMaxNumNeighbors_ReturnsLargestCount)
{
    ElementMap em;
    em.registerElements("H");
    Structure s;
    s.setElementMap(em);

    Atom a0;
    a0.r = Vec3D(0.0, 0.0, 0.0);
    s.addAtom(a0, "H");
    Atom a1;
    a1.r = Vec3D(1.0, 0.0, 0.0);
    s.addAtom(a1, "H");
    Atom a2;
    a2.r = Vec3D(2.0, 0.0, 0.0);
    s.addAtom(a2, "H");

    s.calculateNeighborList(10.0);

    // Middle atom (index 1) has 2 neighbors, end atoms have 1 each.
    BOOST_REQUIRE_EQUAL(s.getMaxNumNeighbors(), 2);
}

BOOST_AUTO_TEST_CASE(Reset_ClearsEverythingButElementMap)
{
    ElementMap em;
    em.registerElements("H O");
    Structure s;
    s.setElementMap(em);
    s.isPeriodic = true;
    s.numAtoms = 5;
    s.energy = 3.0;
    s.comment = "not empty";
    Atom a;
    s.addAtom(a, "H");

    s.reset();

    BOOST_REQUIRE_EQUAL(s.isPeriodic, false);
    BOOST_REQUIRE_EQUAL(s.numAtoms, 0);
    BOOST_REQUIRE_SMALL(s.energy, accuracy);
    BOOST_REQUIRE_EQUAL(s.comment, "");
    BOOST_REQUIRE(s.atoms.empty());
    // elementMap itself survives reset().
    BOOST_REQUIRE_EQUAL(s.numElements, 2);
    BOOST_REQUIRE_EQUAL(s.numAtomsPerElement.size(), 2);
}

BOOST_AUTO_TEST_CASE(ClearNeighborList_ResetsAtomsAndStaleCounts)
{
    ElementMap em;
    em.registerElements("H O");
    Structure s;
    s.setElementMap(em);

    Atom a0;
    a0.r = Vec3D(0.0, 0.0, 0.0);
    s.addAtom(a0, "H");
    Atom a1;
    a1.r = Vec3D(1.0, 0.0, 0.0);
    s.addAtom(a1, "O");

    s.calculateNeighborList(10.0);
    BOOST_REQUIRE(s.atoms.at(0).numNeighbors > 0);

    s.clearNeighborList();

    BOOST_REQUIRE_EQUAL(s.hasNeighborList, false);
    for (auto const& a : s.atoms)
    {
        BOOST_REQUIRE_EQUAL(a.numNeighbors, 0);
        BOOST_REQUIRE(a.neighbors.empty());
        for (auto const& c : a.numNeighborsPerElement)
        {
            BOOST_REQUIRE_EQUAL(c, 0);
        }
    }

    // Recomputing after clearing must not double-count (regression check
    // for the numNeighborsPerElement reset fix, exercised through the
    // Structure-level API actually used throughout the codebase).
    s.calculateNeighborList(10.0);
    BOOST_REQUIRE_EQUAL(s.atoms.at(0).numNeighborsPerElement.at(
                             em.index("O")), 1);
}

BOOST_AUTO_TEST_CASE(FreeAtoms_ClearsSymmetryFunctionData)
{
    ElementMap em;
    em.registerElements("H");
    Structure s;
    s.setElementMap(em);
    Atom a;
    s.addAtom(a, "H");
    s.atoms.at(0).numSymmetryFunctions = 2;
    s.atoms.at(0).allocate(false);
    BOOST_REQUIRE_EQUAL(s.atoms.at(0).G.size(), 2);

    s.freeAtoms(true);

    BOOST_REQUIRE_EQUAL(s.atoms.at(0).G.size(), 0);
    BOOST_REQUIRE_EQUAL(s.hasSymmetryFunctions, false);
}

BOOST_AUTO_TEST_CASE(UpdateError_Energy_AccumulatesMetrics)
{
    Structure s;
    s.numAtoms = 2;
    s.energyRef = 10.0;
    s.energy = 8.0;
    map<string, double> error = {{"RMSEpa", 0.0}, {"RMSE", 0.0},
                                 {"MAEpa", 0.0}, {"MAE", 0.0}};
    size_t count = 0;

    s.updateError("energy", error, count);

    BOOST_REQUIRE_EQUAL(count, 1);
    double const diff = 10.0 - 8.0;
    BOOST_REQUIRE_SMALL(error.at("RMSE") - diff * diff, accuracy);
    BOOST_REQUIRE_SMALL(error.at("RMSEpa") - diff * diff / (2.0 * 2.0), accuracy);
    BOOST_REQUIRE_SMALL(error.at("MAE") - fabs(diff), accuracy);
    BOOST_REQUIRE_SMALL(error.at("MAEpa") - fabs(diff) / 2.0, accuracy);
}

BOOST_AUTO_TEST_CASE(UpdateError_Force_DelegatesToAtoms)
{
    ElementMap em;
    em.registerElements("H");
    Structure s;
    s.setElementMap(em);
    Atom a;
    a.fRef = Vec3D(1.0, 0.0, 0.0);
    a.f = Vec3D(0.0, 0.0, 0.0);
    s.addAtom(a, "H");

    map<string, double> error = {{"RMSE", 0.0}, {"MAE", 0.0}};
    size_t count = 0;
    s.updateError("force", error, count);

    BOOST_REQUIRE_EQUAL(count, 3);
    BOOST_REQUIRE_SMALL(error.at("RMSE") - 1.0, accuracy);
}

BOOST_AUTO_TEST_CASE(UpdateError_UnknownProperty_Throws)
{
    Structure s;
    map<string, double> error;
    size_t count = 0;
    BOOST_REQUIRE_THROW(s.updateError("stress", error, count), runtime_error);
}

BOOST_AUTO_TEST_CASE(CalculateMaxCutoffRadiusOverall_NonPeriodic_MaxOfInputs)
{
    Structure s;
    s.isPeriodic = false;
    EwaldSetup ewaldSetup;

    s.calculateMaxCutoffRadiusOverall(ewaldSetup, 3.0, 5.0);
    BOOST_REQUIRE_SMALL(s.maxCutoffRadiusOverall - 5.0, accuracy);

    s.calculateMaxCutoffRadiusOverall(ewaldSetup, 7.0, 5.0);
    BOOST_REQUIRE_SMALL(s.maxCutoffRadiusOverall - 7.0, accuracy);
}

BOOST_AUTO_TEST_CASE(GetEnergyLine_PerAtomValues)
{
    Structure s;
    s.index = 2;
    s.numAtoms = 2;
    s.energyRef = 10.0;
    s.energy = 8.0;

    string const line = s.getEnergyLine();
    BOOST_REQUIRE(line.find("2") != string::npos);
}

BOOST_AUTO_TEST_CASE(GetForcesLines_AggregatesAllAtoms)
{
    ElementMap em;
    em.registerElements("H");
    Structure s;
    s.setElementMap(em);
    Atom a1;
    s.addAtom(a1, "H");
    Atom a2;
    s.addAtom(a2, "H");

    vector<string> const lines = s.getForcesLines();
    BOOST_REQUIRE_EQUAL(lines.size(), 6); // 3 lines per atom.
}

BOOST_AUTO_TEST_CASE(GetChargesLines_AggregatesAllAtoms)
{
    ElementMap em;
    em.registerElements("H");
    Structure s;
    s.setElementMap(em);
    Atom a1;
    s.addAtom(a1, "H");
    Atom a2;
    s.addAtom(a2, "H");

    vector<string> const lines = s.getChargesLines();
    BOOST_REQUIRE_EQUAL(lines.size(), 2);
}

BOOST_AUTO_TEST_CASE(ReadFromLines_ParsesBasicStructure)
{
    ElementMap em;
    em.registerElements("H O");
    Structure s;
    s.setElementMap(em);

    vector<string> lines = {
        "begin",
        "comment test structure",
        "lattice 10.0 0.0 0.0",
        "lattice 0.0 10.0 0.0",
        "lattice 0.0 0.0 10.0",
        "atom 1.0 2.0 3.0 H 0.1 0.0 0.01 0.02 0.03",
        "atom 4.0 5.0 6.0 O -0.1 0.0 -0.04 -0.05 -0.06",
        "energy -12.5",
        "charge 0.0",
        "end"
    };

    s.readFromLines(lines);

    BOOST_REQUIRE_EQUAL(s.comment, "test structure");
    BOOST_REQUIRE(s.isPeriodic);
    BOOST_REQUIRE_SMALL(s.box[0][0] - 10.0, accuracy);
    BOOST_REQUIRE_EQUAL(s.numAtoms, 2);
    BOOST_REQUIRE_SMALL(s.energyRef - (-12.5), accuracy);
    BOOST_REQUIRE_SMALL(s.chargeRef, accuracy);

    Atom const& a0 = s.atoms.at(0);
    BOOST_REQUIRE_SMALL(a0.r[0] - 1.0, accuracy);
    BOOST_REQUIRE_SMALL(a0.r[1] - 2.0, accuracy);
    BOOST_REQUIRE_SMALL(a0.r[2] - 3.0, accuracy);
    BOOST_REQUIRE_EQUAL(a0.element, em.index("H"));
    BOOST_REQUIRE_SMALL(a0.chargeRef - 0.1, accuracy);
    BOOST_REQUIRE_SMALL(a0.fRef[0] - 0.01, accuracy);

    Atom const& a1 = s.atoms.at(1);
    BOOST_REQUIRE_EQUAL(a1.element, em.index("O"));
    BOOST_REQUIRE_SMALL(a1.fRef[2] - (-0.06), accuracy);

    BOOST_REQUIRE_EQUAL(s.numElementsPresent, 2);
}

BOOST_AUTO_TEST_CASE(ReadFromLines_UnknownKeyword_Throws)
{
    ElementMap em;
    em.registerElements("H");
    Structure s;
    s.setElementMap(em);

    vector<string> lines = {"begin", "foobar 1.0", "end"};
    BOOST_REQUIRE_THROW(s.readFromLines(lines), runtime_error);
}

BOOST_AUTO_TEST_CASE(ReadFromLines_TooManyLatticeVectors_Throws)
{
    ElementMap em;
    em.registerElements("H");
    Structure s;
    s.setElementMap(em);

    vector<string> lines = {
        "begin",
        "lattice 10.0 0.0 0.0",
        "lattice 0.0 10.0 0.0",
        "lattice 0.0 0.0 10.0",
        "lattice 0.0 0.0 10.0",
        "end"
    };
    BOOST_REQUIRE_THROW(s.readFromLines(lines), runtime_error);
}

BOOST_AUTO_TEST_CASE(ReadFromLines_PeriodicNonNeutralCharge_Throws)
{
    ElementMap em;
    em.registerElements("H");
    Structure s;
    s.setElementMap(em);

    vector<string> lines = {
        "begin",
        "lattice 10.0 0.0 0.0",
        "lattice 0.0 10.0 0.0",
        "lattice 0.0 0.0 10.0",
        "atom 0.0 0.0 0.0 H 0.0 0.0 0.0 0.0 0.0",
        "energy 0.0",
        "charge 1.0",
        "end"
    };
    BOOST_REQUIRE_THROW(s.readFromLines(lines), runtime_error);
}

BOOST_FIXTURE_TEST_CASE(WriteToFile_ReadFromFile_RoundTrip, FixtureRepairDir)
{
    ElementMap em;
    em.registerElements("H O");
    Structure s;
    s.setElementMap(em);
    s.energyRef = -7.5;
    s.chargeRef = 0.0;
    Atom a0;
    a0.r = Vec3D(1.0, 2.0, 3.0);
    a0.chargeRef = 0.2;
    a0.fRef = Vec3D(0.1, 0.2, 0.3);
    s.addAtom(a0, "H");
    Atom a1;
    a1.r = Vec3D(4.0, 5.0, 6.0);
    a1.chargeRef = -0.2;
    a1.fRef = Vec3D(-0.1, -0.2, -0.3);
    s.addAtom(a1, "O");

    BOOST_REQUIRE(bfs::create_directory("test"));
    bfs::current_path("test");

    s.writeToFile("structure.data", true, false);

    Structure s2;
    s2.setElementMap(em);
    s2.readFromFile("structure.data");

    BOOST_REQUIRE_EQUAL(s2.numAtoms, 2);
    BOOST_REQUIRE_SMALL(s2.energyRef - s.energyRef, accuracyNumeric);
    BOOST_REQUIRE_SMALL((s2.atoms.at(0).r - s.atoms.at(0).r).norm2(),
                        accuracyNumeric);
    BOOST_REQUIRE_SMALL(s2.atoms.at(0).chargeRef - s.atoms.at(0).chargeRef,
                        accuracyNumeric);
    BOOST_REQUIRE_SMALL(
        (s2.atoms.at(1).fRef - s.atoms.at(1).fRef).norm2(), accuracyNumeric);
    BOOST_REQUIRE_EQUAL(s2.atoms.at(1).element, em.index("O"));

    bfs::current_path("..");
    bfs::remove_all("test");
}

BOOST_AUTO_TEST_CASE(Info_NonEmpty)
{
    ElementMap em;
    em.registerElements("H");
    Structure s;
    s.setElementMap(em);
    Atom a;
    s.addAtom(a, "H");

    vector<string> const info = s.info();
    BOOST_REQUIRE(!info.empty());
}

BOOST_AUTO_TEST_SUITE_END()
