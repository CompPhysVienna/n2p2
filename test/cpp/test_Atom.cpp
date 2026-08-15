#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Atom
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "Atom.h"
#include <cmath>     // std::fabs
#include <limits>    // std::numeric_limits
#include <map>       // std::map
#include <stdexcept> // std::range_error, std::runtime_error
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Constructor_Defaults)
{
    Atom a;
    BOOST_REQUIRE_EQUAL(a.hasNeighborList, false);
    BOOST_REQUIRE_EQUAL(a.NeighborListIsSorted, false);
    BOOST_REQUIRE_EQUAL(a.hasSymmetryFunctions, false);
    BOOST_REQUIRE_EQUAL(a.hasSymmetryFunctionDerivatives, false);
    BOOST_REQUIRE_EQUAL(a.useChargeNeuron, false);
    BOOST_REQUIRE_EQUAL(a.index, 0);
    BOOST_REQUIRE_EQUAL(a.indexStructure, 0);
    BOOST_REQUIRE_EQUAL(a.tag, 0);
    BOOST_REQUIRE_EQUAL(a.element, 0);
    BOOST_REQUIRE_EQUAL(a.numNeighbors, 0);
    BOOST_REQUIRE_EQUAL(a.numNeighborsUnique, 0);
    BOOST_REQUIRE_EQUAL(a.numSymmetryFunctions, 0);
    BOOST_REQUIRE_SMALL(a.energy, accuracy);
    BOOST_REQUIRE_SMALL(a.chi, accuracy);
    BOOST_REQUIRE_SMALL(a.charge, accuracy);
    BOOST_REQUIRE_SMALL(a.chargeRef, accuracy);
}

BOOST_AUTO_TEST_CASE(NeighborConstructor_Defaults)
{
    Atom::Neighbor n;
    BOOST_REQUIRE_EQUAL(n.index, 0);
    BOOST_REQUIRE_EQUAL(n.tag, 0);
    BOOST_REQUIRE_EQUAL(n.element, 0);
    BOOST_REQUIRE_SMALL(n.d, accuracy);
}

BOOST_AUTO_TEST_CASE(NeighborOperators_ElementAndDistanceBased)
{
    Atom::Neighbor n1;
    n1.element = 0;
    n1.d = 1.0;
    Atom::Neighbor n2;
    n2.element = 0;
    n2.d = 1.0;
    Atom::Neighbor n3;
    n3.element = 0;
    n3.d = 2.0;

    BOOST_REQUIRE(n1 == n2);
    BOOST_REQUIRE(!(n1 != n2));
    BOOST_REQUIRE(n1 != n3);
    BOOST_REQUIRE(n1 < n3);
    BOOST_REQUIRE(n3 > n1);
    BOOST_REQUIRE(n1 <= n2);
    BOOST_REQUIRE(n1 >= n2);
    BOOST_REQUIRE(n1 <= n3);
    BOOST_REQUIRE(n3 >= n1);
    BOOST_REQUIRE(!(n3 <= n1));
    BOOST_REQUIRE(!(n1 >= n3));
}

BOOST_AUTO_TEST_CASE(ToNormalizedToPhysicalUnits_RoundTrip)
{
    Atom a;
    a.energy = 2.0;
    a.r = Vec3D(1.0, 2.0, 3.0);
    a.f = Vec3D(0.1, 0.2, 0.3);
    a.fRef = Vec3D(0.4, 0.5, 0.6);
    a.charge = 0.5;
    a.chargeRef = -0.5;

    double const convEnergy = 2.0;
    double const convLength = 3.0;
    double const convCharge = 4.0;

    Atom const original = a;

    a.toNormalizedUnits(convEnergy, convLength, convCharge);
    // Values should have actually changed.
    BOOST_REQUIRE(fabs(a.energy - original.energy) > accuracy);

    a.toPhysicalUnits(convEnergy, convLength, convCharge);
    BOOST_REQUIRE_SMALL(a.energy - original.energy, accuracy);
    BOOST_REQUIRE_SMALL((a.r - original.r).norm2(), accuracy);
    BOOST_REQUIRE_SMALL((a.f - original.f).norm2(), accuracy);
    BOOST_REQUIRE_SMALL((a.fRef - original.fRef).norm2(), accuracy);
    BOOST_REQUIRE_SMALL(a.charge - original.charge, accuracy);
    BOOST_REQUIRE_SMALL(a.chargeRef - original.chargeRef, accuracy);
}

BOOST_AUTO_TEST_CASE(ToNormalizedUnits_ScalesSymmetryFunctionDerivatives)
{
    Atom a;
    a.numSymmetryFunctions = 2;
    a.hasSymmetryFunctionDerivatives = true;
    a.useChargeNeuron = false;
    a.dEdG = {1.0, 2.0};
    a.dGdr = {Vec3D(1.0, 0.0, 0.0), Vec3D(0.0, 1.0, 0.0)};

    double const convEnergy = 2.0;
    double const convLength = 5.0;

    a.toNormalizedUnits(convEnergy, convLength, 1.0);

    BOOST_REQUIRE_SMALL(a.dEdG.at(0) - 1.0 * convEnergy, accuracy);
    BOOST_REQUIRE_SMALL(a.dEdG.at(1) - 2.0 * convEnergy, accuracy);
    BOOST_REQUIRE_SMALL(a.dGdr.at(0)[0] - 1.0 / convLength, accuracy);
    BOOST_REQUIRE_SMALL(a.dGdr.at(1)[1] - 1.0 / convLength, accuracy);
}

BOOST_AUTO_TEST_CASE(Allocate_ZeroSymmetryFunctions_Throws)
{
    Atom a;
    BOOST_REQUIRE_THROW(a.allocate(false), range_error);
}

BOOST_AUTO_TEST_CASE(Allocate_MissingDerivativeInfo_Throws)
{
    Atom a;
    a.numSymmetryFunctions = 2;
    // numSymmetryFunctionDerivatives left empty.
    BOOST_REQUIRE_THROW(a.allocate(true), range_error);
}

BOOST_AUTO_TEST_CASE(Allocate_GOnly_ResizesOnlyG)
{
    Atom a;
    a.numSymmetryFunctions = 3;
    a.allocate(false);

    BOOST_REQUIRE_EQUAL(a.G.size(), 3);
    BOOST_REQUIRE_EQUAL(a.dEdG.size(), 0);
    BOOST_REQUIRE_EQUAL(a.dGdr.size(), 0);
    BOOST_REQUIRE_EQUAL(a.hasSymmetryFunctions, false);
    BOOST_REQUIRE_EQUAL(a.hasSymmetryFunctionDerivatives, false);
}

BOOST_AUTO_TEST_CASE(Allocate_All_ResizesDerivativeVectorsToo)
{
    Atom a;
    a.numSymmetryFunctions = 3;
    a.numSymmetryFunctionDerivatives = {1, 1};
    a.useChargeNeuron = false;
    a.allocate(true);

    BOOST_REQUIRE_EQUAL(a.G.size(), 3);
    BOOST_REQUIRE_EQUAL(a.dEdG.size(), 3);
    BOOST_REQUIRE_EQUAL(a.dQdG.size(), 3);
    BOOST_REQUIRE_EQUAL(a.dChidG.size(), 3);
    BOOST_REQUIRE_EQUAL(a.dGdr.size(), 3);
}

BOOST_AUTO_TEST_CASE(Allocate_All_WithChargeNeuron_ExtraDEdGEntry)
{
    Atom a;
    a.numSymmetryFunctions = 3;
    a.numSymmetryFunctionDerivatives = {1};
    a.useChargeNeuron = true;
    a.allocate(true);

    BOOST_REQUIRE_EQUAL(a.dEdG.size(), 4);
}

BOOST_AUTO_TEST_CASE(Free_All_ClearsGAndDerivatives)
{
    Atom a;
    a.numSymmetryFunctions = 2;
    a.numSymmetryFunctionDerivatives = {1};
    a.allocate(true);

    a.free(true);

    BOOST_REQUIRE_EQUAL(a.G.size(), 0);
    BOOST_REQUIRE_EQUAL(a.dEdG.size(), 0);
    BOOST_REQUIRE_EQUAL(a.dGdr.size(), 0);
    BOOST_REQUIRE_EQUAL(a.hasSymmetryFunctions, false);
    BOOST_REQUIRE_EQUAL(a.hasSymmetryFunctionDerivatives, false);
}

BOOST_AUTO_TEST_CASE(Free_NotAll_KeepsGButClearsDerivatives)
{
    Atom a;
    a.numSymmetryFunctions = 2;
    a.numSymmetryFunctionDerivatives = {1};
    a.allocate(true);

    a.free(false);

    BOOST_REQUIRE_EQUAL(a.G.size(), 2);
    BOOST_REQUIRE_EQUAL(a.dEdG.size(), 0);
    BOOST_REQUIRE_EQUAL(a.dGdr.size(), 0);
    BOOST_REQUIRE_EQUAL(a.hasSymmetryFunctionDerivatives, false);
}

BOOST_AUTO_TEST_CASE(ClearNeighborList_ResetsCountsAndNeighborVector)
{
    Atom a;
    a.numNeighborsPerElement.resize(2, 0);
    // Simulate a previously computed neighbor list.
    a.numNeighbors = 5;
    a.numNeighborsUnique = 3;
    a.neighborsUnique = {0, 1, 2};
    a.neighbors.resize(5);
    a.hasNeighborList = true;

    a.clearNeighborList();

    BOOST_REQUIRE_EQUAL(a.numNeighbors, 0);
    BOOST_REQUIRE_EQUAL(a.numNeighborsUnique, 0);
    BOOST_REQUIRE(a.neighborsUnique.empty());
    BOOST_REQUIRE(a.neighbors.empty());
    BOOST_REQUIRE_EQUAL(a.hasNeighborList, false);
}

BOOST_AUTO_TEST_CASE(ClearNeighborList_ResetsNumNeighborsPerElementContents)
{
    // Structure::calculateNeighborList() only ever increments
    // numNeighborsPerElement[e]++ (see Structure.cpp), relying on
    // clearNeighborList() to have reset it to all-zero beforehand. Since
    // std::vector::resize(n, 0) is a no-op on existing elements when the
    // vector is already size n (the common case: element count doesn't
    // change between calls), verify the actual per-element counts are
    // zeroed, not just that the vector's size is preserved.
    Atom a;
    a.numNeighborsPerElement = {3, 7};

    a.clearNeighborList();

    BOOST_REQUIRE_EQUAL(a.numNeighborsPerElement.size(), 2);
    BOOST_REQUIRE_EQUAL(a.numNeighborsPerElement.at(0), 0);
    BOOST_REQUIRE_EQUAL(a.numNeighborsPerElement.at(1), 0);
}

BOOST_AUTO_TEST_CASE(ClearNeighborListWithNumElements_ResizesPerElementVector)
{
    Atom a;
    a.numNeighborsPerElement = {3};

    a.clearNeighborList(3);

    BOOST_REQUIRE_EQUAL(a.numNeighborsPerElement.size(), 3);
    for (size_t i = 0; i < 3; ++i)
    {
        BOOST_REQUIRE_EQUAL(a.numNeighborsPerElement.at(i), 0);
    }
}

BOOST_AUTO_TEST_CASE(CalculateNumNeighbors_CountsWithinCutoff)
{
    Atom a;
    Atom::Neighbor n1;
    n1.d = 1.0;
    Atom::Neighbor n2;
    n2.d = 2.0;
    Atom::Neighbor n3;
    n3.d = 5.0;
    a.neighbors = {n1, n2, n3};

    BOOST_REQUIRE_EQUAL(a.calculateNumNeighbors(2.0), 2);
    BOOST_REQUIRE_EQUAL(a.calculateNumNeighbors(10.0), 3);
    BOOST_REQUIRE_EQUAL(a.calculateNumNeighbors(0.5), 0);
}

BOOST_AUTO_TEST_CASE(GetStoredMinNumNeighbors_WithoutCutoffMap_ReturnsTotal)
{
    Atom a;
    a.numNeighbors = 4;
    BOOST_REQUIRE_EQUAL(a.getStoredMinNumNeighbors(1.0), 4);
    BOOST_REQUIRE_EQUAL(a.getStoredMinNumNeighbors(100.0), 4);
}

BOOST_AUTO_TEST_CASE(IsNeighbor_TrueForPresentIndexFalseOtherwise)
{
    Atom a;
    Atom::Neighbor n1;
    n1.index = 3;
    Atom::Neighbor n2;
    n2.index = 7;
    a.neighbors = {n1, n2};

    BOOST_REQUIRE(a.isNeighbor(3));
    BOOST_REQUIRE(a.isNeighbor(7));
    BOOST_REQUIRE(!a.isNeighbor(5));
}

BOOST_AUTO_TEST_CASE(UpdateError_Force_AccumulatesRmseAndMae)
{
    Atom a;
    a.fRef = Vec3D(1.0, 2.0, 3.0);
    a.f    = Vec3D(1.5, 1.0, 3.0);
    map<string, double> error = {{"RMSE", 0.0}, {"MAE", 0.0}};
    size_t count = 0;

    a.updateError("force", error, count);

    BOOST_REQUIRE_EQUAL(count, 3);
    Vec3D const diff = a.fRef - a.f;
    BOOST_REQUIRE_SMALL(error.at("RMSE") - diff.norm2(), accuracy);
    BOOST_REQUIRE_SMALL(error.at("MAE") - diff.l1norm(), accuracy);
}

BOOST_AUTO_TEST_CASE(UpdateError_Charge_AccumulatesRmseAndMae)
{
    Atom a;
    a.chargeRef = 0.8;
    a.charge = 0.5;
    map<string, double> error = {{"RMSE", 0.0}, {"MAE", 0.0}};
    size_t count = 0;

    a.updateError("charge", error, count);

    BOOST_REQUIRE_EQUAL(count, 1);
    double const diff = a.chargeRef - a.charge;
    BOOST_REQUIRE_SMALL(error.at("RMSE") - diff * diff, accuracy);
    BOOST_REQUIRE_SMALL(error.at("MAE") - fabs(diff), accuracy);
}

BOOST_AUTO_TEST_CASE(UpdateError_UnknownProperty_Throws)
{
    Atom a;
    map<string, double> error = {{"RMSE", 0.0}, {"MAE", 0.0}};
    size_t count = 0;
    BOOST_REQUIRE_THROW(a.updateError("stress", error, count), runtime_error);
}

BOOST_AUTO_TEST_CASE(CalculateSelfForceShort_CorrectValue)
{
    Atom a;
    a.numSymmetryFunctions = 2;
    a.dEdG = {2.0, 3.0};
    a.dGdr = {Vec3D(1.0, 0.0, 0.0), Vec3D(0.0, 1.0, 0.0)};

    Vec3D const f = a.calculateSelfForceShort();

    Vec3D expected{};
    for (size_t i = 0; i < 2; ++i) expected -= a.dEdG.at(i) * a.dGdr.at(i);
    BOOST_REQUIRE_SMALL((f - expected).norm2(), accuracy);
}

BOOST_AUTO_TEST_CASE(CalculatePairForceShort_NullTable_Throws)
{
    Atom a;
    Atom::Neighbor n;
    n.element = 0;
    n.dGdr = {Vec3D(1.0, 0.0, 0.0)};
    BOOST_REQUIRE_THROW(a.calculatePairForceShort(n, nullptr), runtime_error);
}

BOOST_AUTO_TEST_CASE(CalculatePairForceShort_CorrectValue)
{
    Atom a;
    a.dEdG = {1.0, 2.0, 3.0};

    Atom::Neighbor n;
    n.element = 0;
    n.dGdr = {Vec3D(1.0, 0.0, 0.0), Vec3D(0.0, 1.0, 0.0), Vec3D(0.0, 0.0, 1.0)};

    vector<vector<size_t> > tableFull = {{0, 1, 2}};

    Vec3D const f = a.calculatePairForceShort(n, &tableFull);

    Vec3D expected{};
    for (size_t i = 0; i < n.dGdr.size(); ++i)
    {
        expected -= a.dEdG.at(tableFull.at(0).at(i)) * n.dGdr.at(i);
    }
    BOOST_REQUIRE_SMALL((f - expected).norm2(), accuracy);
}

BOOST_AUTO_TEST_CASE(CalculateDChidr_OwnAtomContribution)
{
    Atom a;
    a.index = 5;
    a.numSymmetryFunctions = 2;
    a.dChidG = {1.5, -2.0};
    a.dGdr = {Vec3D(1.0, 0.0, 0.0), Vec3D(0.0, 1.0, 0.0)};

    vector<vector<size_t> > tableFull = {{0, 1}};
    Vec3D const dChidr = a.calculateDChidr(5, 10.0, &tableFull);

    Vec3D expected{};
    for (size_t k = 0; k < 2; ++k) expected += a.dChidG.at(k) * a.dGdr.at(k);
    BOOST_REQUIRE_SMALL((dChidr - expected).norm2(), accuracy);
}

BOOST_AUTO_TEST_CASE(CalculateDChidr_NeighborContribution)
{
    Atom a;
    a.index = 5;
    a.dChidG = {1.5, -2.0};

    Atom::Neighbor n;
    n.tag = 7;
    n.element = 0;
    n.dGdr = {Vec3D(1.0, 0.0, 0.0), Vec3D(0.0, 1.0, 0.0)};
    a.neighbors = {n};
    a.numNeighbors = 1;

    vector<vector<size_t> > tableFull = {{0, 1}};
    // atomIndexOfR (7) differs from a.index (5), so only the neighbor
    // contribution (tag == 7) is included.
    Vec3D const dChidr = a.calculateDChidr(7, 10.0, &tableFull);

    Vec3D expected{};
    for (size_t k = 0; k < 2; ++k) expected += a.dChidG.at(k) * n.dGdr.at(k);
    BOOST_REQUIRE_SMALL((dChidr - expected).norm2(), accuracy);
}

BOOST_AUTO_TEST_CASE(GetForcesLines_ThreeLinesWithExpectedIndices)
{
    Atom a;
    a.indexStructure = 2;
    a.index = 9;
    a.fRef = Vec3D(1.0, 2.0, 3.0);
    a.f = Vec3D(1.1, 2.2, 3.3);

    vector<string> const lines = a.getForcesLines();
    BOOST_REQUIRE_EQUAL(lines.size(), 3);
    for (auto const& line : lines)
    {
        BOOST_REQUIRE(line.find("2") != string::npos);
        BOOST_REQUIRE(line.find("9") != string::npos);
    }
}

BOOST_AUTO_TEST_CASE(GetChargeLine_ContainsIndicesAndValues)
{
    Atom a;
    a.indexStructure = 1;
    a.index = 4;
    a.chargeRef = 0.5;
    a.charge = 0.3;

    string const line = a.getChargeLine();
    BOOST_REQUIRE(line.find("1") != string::npos);
    BOOST_REQUIRE(line.find("4") != string::npos);
}

BOOST_AUTO_TEST_CASE(Info_NonEmpty)
{
    Atom a;
    a.numSymmetryFunctions = 1;
    a.allocate(false);
    vector<string> const info = a.info();
    BOOST_REQUIRE(!info.empty());
}

BOOST_AUTO_TEST_CASE(NeighborInfo_NonEmpty)
{
    Atom::Neighbor n;
    vector<string> const info = n.info();
    BOOST_REQUIRE(!info.empty());
}

BOOST_AUTO_TEST_SUITE_END()
