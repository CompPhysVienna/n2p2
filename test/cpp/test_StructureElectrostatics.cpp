#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE StructureElectrostatics
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "Atom.h"
#include "Element.h"
#include "ElementMap.h"
#include "ErfcBuf.h"
#include "EwaldSetup.h"
#include "ScreeningFunction.h"
#include "Structure.h"
#include "Vec3D.h"
#include <Eigen/Core>
#include <Eigen/QR>
#include <cmath>   // std::erf
#include <iomanip> // std::setprecision
#include <limits>  // std::numeric_limits
#include <sstream> // std::ostringstream
#include <vector>  // std::vector

using namespace std;
using namespace nnp;
using namespace Eigen;

namespace bdata = boost::unit_test::data;

double const accuracyNumeric = 1.0E-6;

// Two-atom, non-periodic, single-element (H) fixture shared by most tests
// below. Screening is inactive by default (outer cutoff far below the
// interatomic separation), isolating the core charge-equilibration
// formulas from the screening correction.
struct ElectrostaticsFixture
{
    ElectrostaticsFixture() :
        hardness(1),
        sigmaSqrtPi(1),
        gammaSqrt2(1, 1),
        fourPiEps(1.0)
    {
        em.registerElements("H");
        s.setElementMap(em);
        s.isPeriodic = false;

        Atom a0;
        a0.r = Vec3D(0.0, 0.0, 0.0);
        a0.chi = 0.5;
        s.addAtom(a0, "H");

        Atom a1;
        a1.r = Vec3D(2.0, 0.0, 0.0);
        a1.chi = -0.3;
        s.addAtom(a1, "H");

        s.chargeRef = 0.0;

        hardness(0) = 1.0;
        sigmaSqrtPi(0) = 0.5;
        gammaSqrt2(0, 0) = 0.8;

        fs.setInnerOuter(0.001, 0.002); // Inactive at rij = 2.0.
    }

    ElementMap em;
    Structure s;
    VectorXd hardness;
    VectorXd sigmaSqrtPi;
    MatrixXd gammaSqrt2;
    double fourPiEps;
    ScreeningFunction fs;
    EwaldSetup ewaldSetup;
    ErfcBuf erfcBuf;
};

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_FIXTURE_TEST_CASE(
    CalculateElectrostaticEnergy_NonPeriodic_MatchesIndependentSolve,
    ElectrostaticsFixture)
{
    double const error = s.calculateElectrostaticEnergy(
        ewaldSetup, hardness, gammaSqrt2, sigmaSqrtPi, fs, fourPiEps, erfcBuf);

    BOOST_REQUIRE_SMALL(error, accuracyNumeric);
    BOOST_REQUIRE(s.hasAMatrix);
    BOOST_REQUIRE(s.hasCharges);

    // Reference values from an independent (pure Python, Gaussian
    // elimination) solve of the same 3x3 linear system (2 charges +
    // Lagrange multiplier for total-charge conservation).
    double const Q0Ref         = -1.599869785952591E-01;
    double const Q1Ref         =  1.599869785952591E-01;
    double const lambdaRef     = -1.000000000000000E-01;
    // energyElec = 0.5*Q^T*(A - diag(hardness))*Q (quadratic part) plus the
    // unconditional per-atom self-energy term from calculateScreeningEnergy
    // (-Qi^2 / (2*sigmaSqrtPi*fourPiEps)), which the original reference
    // value below omitted.
    double const energyElecQuadraticRef =  3.839895811806372E-02;
    double const energyElecSelfRef      = -5.119166664007978E-02;
    double const energyElecRef = energyElecQuadraticRef + energyElecSelfRef;

    BOOST_REQUIRE_SMALL(s.atoms.at(0).charge - Q0Ref, accuracyNumeric);
    BOOST_REQUIRE_SMALL(s.atoms.at(1).charge - Q1Ref, accuracyNumeric);
    BOOST_REQUIRE_SMALL(s.lambda - lambdaRef, accuracyNumeric);
    BOOST_REQUIRE_SMALL(s.energyElec - energyElecRef, accuracyNumeric);
}

BOOST_AUTO_TEST_CASE(CalculateElectrostaticEnergy_ChargeConservation_ThreeAtoms)
{
    ElementMap em;
    em.registerElements("H O");
    Structure s;
    s.setElementMap(em);
    s.isPeriodic = false;

    Atom a0;
    a0.r = Vec3D(0.0, 0.0, 0.0);
    a0.chi = 0.3;
    s.addAtom(a0, "H");
    Atom a1;
    a1.r = Vec3D(1.5, 0.0, 0.0);
    a1.chi = -0.6;
    s.addAtom(a1, "O");
    Atom a2;
    a2.r = Vec3D(0.0, 1.5, 0.0);
    a2.chi = 0.2;
    s.addAtom(a2, "H");
    s.chargeRef = 0.5;

    VectorXd hardness(2);
    hardness << 1.0, 1.2;
    VectorXd sigmaSqrtPi(2);
    sigmaSqrtPi << 0.5, 0.6;
    MatrixXd gammaSqrt2(2, 2);
    gammaSqrt2 << 0.8, 0.9, 0.9, 1.0;
    double const fourPiEps = 1.0;
    ScreeningFunction fs;
    fs.setInnerOuter(0.001, 0.002);
    EwaldSetup ewaldSetup;
    ErfcBuf erfcBuf;

    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);

    double sumQ = 0.0;
    for (auto const& a : s.atoms) sumQ += a.charge;
    BOOST_REQUIRE_SMALL(sumQ - s.chargeRef, accuracyNumeric);
}

BOOST_FIXTURE_TEST_CASE(CalculateScreeningEnergy_NonPeriodic_MatchesFormula,
                        ElectrostaticsFixture)
{
    // calculateScreeningEnergy()'s non-periodic branch reads the member
    // matrix A(i,j) directly rather than recomputing it, so A must already
    // be populated. A's pairwise entries do not depend on the screening
    // function, so calling calculateElectrostaticEnergy() with the
    // fixture's (inactive) screening settings is sufficient to populate it
    // before activating screening and overriding the charges below.
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);

    // Enable active screening at the fixture's separation (rij = 2.0).
    fs.setInnerOuter(1.0, 3.0);

    s.atoms.at(0).charge = 0.3;
    s.atoms.at(1).charge = -0.3;

    double const energyScreen = s.calculateScreeningEnergy(
        gammaSqrt2, sigmaSqrtPi, fs, fourPiEps);

    double const rij = 2.0;
    double const Qi = 0.3;
    double const Qj = -0.3;
    double const selfTerm = -Qi * Qi / (2.0 * sigmaSqrtPi(0) * fourPiEps)
                           -  Qj * Qj / (2.0 * sigmaSqrtPi(0) * fourPiEps);
    double const pairTerm = Qi * Qj * erf(rij / gammaSqrt2(0, 0))
                           * (fs.f(rij) - 1.0) / (rij * fourPiEps);
    double const expected = selfTerm + pairTerm;

    BOOST_REQUIRE_SMALL(energyScreen - expected, accuracyNumeric);
}

BOOST_FIXTURE_TEST_CASE(CalculateDQdChi_MatchesFiniteDifference,
                        ElectrostaticsFixture)
{
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);

    vector<VectorXd> dQdChi;
    s.calculateDQdChi(dQdChi);
    BOOST_REQUIRE_EQUAL(dQdChi.size(), 2);

    double const h = 1.0E-6;
    double const chi0Orig = s.atoms.at(0).chi;

    s.atoms.at(0).chi = chi0Orig + h;
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);
    double const Q0Hi = s.atoms.at(0).charge;
    double const Q1Hi = s.atoms.at(1).charge;

    s.atoms.at(0).chi = chi0Orig - h;
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);
    double const Q0Lo = s.atoms.at(0).charge;
    double const Q1Lo = s.atoms.at(1).charge;

    s.atoms.at(0).chi = chi0Orig;

    double const dQ0dChi0Numeric = (Q0Hi - Q0Lo) / (2.0 * h);
    double const dQ1dChi0Numeric = (Q1Hi - Q1Lo) / (2.0 * h);

    BOOST_REQUIRE_SMALL(dQdChi.at(0)(0) - dQ0dChi0Numeric, accuracyNumeric);
    BOOST_REQUIRE_SMALL(dQdChi.at(0)(1) - dQ1dChi0Numeric, accuracyNumeric);
}

BOOST_FIXTURE_TEST_CASE(CalculateDQdJ_MatchesFiniteDifference,
                        ElectrostaticsFixture)
{
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);

    vector<VectorXd> dQdJ;
    s.calculateDQdJ(dQdJ);
    BOOST_REQUIRE_EQUAL(dQdJ.size(), 1); // Single element (H).

    double const h = 1.0E-6;
    double const hardness0Orig = hardness(0);

    hardness(0) = hardness0Orig + h;
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);
    double const Q0Hi = s.atoms.at(0).charge;
    double const Q1Hi = s.atoms.at(1).charge;

    hardness(0) = hardness0Orig - h;
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);
    double const Q0Lo = s.atoms.at(0).charge;
    double const Q1Lo = s.atoms.at(1).charge;

    hardness(0) = hardness0Orig;

    double const dQ0dJNumeric = (Q0Hi - Q0Lo) / (2.0 * h);
    double const dQ1dJNumeric = (Q1Hi - Q1Lo) / (2.0 * h);

    BOOST_REQUIRE_SMALL(dQdJ.at(0)(0) - dQ0dJNumeric, accuracyNumeric);
    BOOST_REQUIRE_SMALL(dQdJ.at(0)(1) - dQ1dJNumeric, accuracyNumeric);
}

// energyElec (as reported by calculateElectrostaticEnergy()) is 0.5*Q^T*
// (A - diag(hardness))*Q + screeningEnergy(Q). The linear system that is
// actually solved for Q, A*Q = b, is instead the stationarity condition of
// a DIFFERENT functional: E_solve(Q,r) = 0.5*Q^T*A*Q - Q^T*b (dE_solve/dQ
// = A*Q - b = 0). Because b (built from chi and chargeRef) does not depend
// on atom positions, the envelope theorem applies cleanly to E_solve: the
// TOTAL derivative of E_solve with respect to r (re-solving Q at each
// perturbed r) equals its PARTIAL derivative at fixed Q, namely
// 0.5*Q^T*(dA/dr)*Q. It does NOT apply to energyElec directly, since
// energyElec differs from E_solve by Q-dependent terms (the hardness
// removal and the screening energy) whose gradient w.r.t. Q is not zero
// at the solution.
//
// With screening inactive (as in this fixture), pEelecpr_i reduces to
// exactly 0.5*Q^T*(dA/dr_i)*Q (see calculateElectrostaticEnergyDerivatives
// -- the screening-correction Tij loop is skipped for rij >= rcutScreen),
// so finite-differencing E_solve is a valid, independent check of
// calculateDAdrQ() + calculateElectrostaticEnergyDerivatives() together.
static double calculateESolve(Structure const& s)
{
    size_t const n = s.numAtoms;
    VectorXd Q(n + 1);
    VectorXd b(n + 1);
    for (size_t i = 0; i < n; ++i)
    {
        Q(i) = s.atoms.at(i).charge;
        b(i) = -s.atoms.at(i).chi;
    }
    Q(n) = s.lambda;
    b(n) = s.chargeRef;

    return (0.5 * Q.transpose() * s.A * Q - Q.transpose() * b).value();
}

BOOST_FIXTURE_TEST_CASE(
    CalculateDAdrQAndDerivatives_PEelecpr_MatchesNumericDerivative,
    ElectrostaticsFixture)
{
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);
    s.calculateDAdrQ(ewaldSetup, gammaSqrt2, fourPiEps, erfcBuf);
    s.calculateElectrostaticEnergyDerivatives(hardness, gammaSqrt2,
                                              sigmaSqrtPi, fs, fourPiEps);

    double const h = 1.0E-6;
    for (size_t i = 0; i < 2; ++i)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            double const orig = s.atoms.at(i).r[c];

            s.atoms.at(i).r[c] = orig + h;
            s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                           sigmaSqrtPi, fs, fourPiEps,
                                           erfcBuf);
            double const eHi = calculateESolve(s);

            s.atoms.at(i).r[c] = orig - h;
            s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                           sigmaSqrtPi, fs, fourPiEps,
                                           erfcBuf);
            double const eLo = calculateESolve(s);

            s.atoms.at(i).r[c] = orig;

            double const numDeriv = (eHi - eLo) / (2.0 * h);
            BOOST_TEST_INFO("atom " << i << " component " << c);
            BOOST_REQUIRE_SMALL(s.atoms.at(i).pEelecpr[c] - numDeriv,
                                accuracyNumeric);
        }
    }
}

BOOST_FIXTURE_TEST_CASE(
    CalculateElectrostaticEnergyDerivatives_DEelecdQ_MatchesDirectFormula,
    ElectrostaticsFixture)
{
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);
    s.calculateDAdrQ(ewaldSetup, gammaSqrt2, fourPiEps, erfcBuf);
    s.calculateElectrostaticEnergyDerivatives(hardness, gammaSqrt2,
                                              sigmaSqrtPi, fs, fourPiEps);

    // Screening is inactive at rij = 2.0, and the non-periodic branch adds
    // no diagonal (i==j) contribution to dEelecdQ (see
    // calculateElectrostaticEnergyDerivatives()): dEelecdQ_i = sum_{j!=i}
    // A(i,j) * Q_j.
    double const rij = 2.0;
    double const Aoff = erf(rij / gammaSqrt2(0, 0)) / (rij * fourPiEps);
    double const expected0 = Aoff * s.atoms.at(1).charge;
    double const expected1 = Aoff * s.atoms.at(0).charge;

    BOOST_REQUIRE_SMALL(s.atoms.at(0).dEelecdQ - expected0, accuracyNumeric);
    BOOST_REQUIRE_SMALL(s.atoms.at(1).dEelecdQ - expected1, accuracyNumeric);
}

BOOST_FIXTURE_TEST_CASE(CalculateForceLambdaElec_MatchesDirectSolve,
                        ElectrostaticsFixture)
{
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);
    s.calculateDAdrQ(ewaldSetup, gammaSqrt2, fourPiEps, erfcBuf);
    s.calculateElectrostaticEnergyDerivatives(hardness, gammaSqrt2,
                                              sigmaSqrtPi, fs, fourPiEps);

    VectorXd const lambdaElec = s.calculateForceLambdaElec();

    // Independent solve of the same system: A * lambdaElec = -dEelecdQ
    // (with the Lagrange-multiplier component set to zero).
    VectorXd dEelecdQ(3);
    dEelecdQ(0) = s.atoms.at(0).dEelecdQ;
    dEelecdQ(1) = s.atoms.at(1).dEelecdQ;
    dEelecdQ(2) = 0.0;
    VectorXd const expected = s.A.colPivHouseholderQr().solve(-dEelecdQ);

    BOOST_REQUIRE_EQUAL(lambdaElec.size(), 3);
    for (int i = 0; i < 3; ++i)
    {
        BOOST_REQUIRE_SMALL(lambdaElec(i) - expected(i), accuracyNumeric);
    }
}

BOOST_FIXTURE_TEST_CASE(CalculateForceLambdaTotal_MatchesDirectSolve,
                        ElectrostaticsFixture)
{
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);
    s.calculateDAdrQ(ewaldSetup, gammaSqrt2, fourPiEps, erfcBuf);
    s.calculateElectrostaticEnergyDerivatives(hardness, gammaSqrt2,
                                              sigmaSqrtPi, fs, fourPiEps);

    // calculateForceLambdaTotal() additionally reads each atom's dEdG.back()
    // (the short-range/NN energy gradient w.r.t. its last symmetry
    // function). Fabricate values directly -- no full symmetry-function/NN
    // pipeline is needed to exercise the linear-algebra assembly under
    // test.
    s.atoms.at(0).dEdG = {0.0, 0.4};
    s.atoms.at(1).dEdG = {0.0, -0.7};

    VectorXd const lambdaTotal = s.calculateForceLambdaTotal();

    VectorXd dEdQ(3);
    dEdQ(0) = s.atoms.at(0).dEelecdQ + s.atoms.at(0).dEdG.back();
    dEdQ(1) = s.atoms.at(1).dEelecdQ + s.atoms.at(1).dEdG.back();
    dEdQ(2) = 0.0;
    VectorXd const expected = s.A.colPivHouseholderQr().solve(-dEdQ);

    BOOST_REQUIRE_EQUAL(lambdaTotal.size(), 3);
    for (int i = 0; i < 3; ++i)
    {
        BOOST_REQUIRE_SMALL(lambdaTotal(i) - expected(i), accuracyNumeric);
    }
}

BOOST_FIXTURE_TEST_CASE(CalculateDQdr_MatchesFiniteDifference,
                        ElectrostaticsFixture)
{
    // calculateDQdr() combines calculateDAdrQ()'s output with
    // Atom::calculateDChidr() (already unit-tested directly in
    // test_Atom.cpp) to assemble and solve a linear system for dQ/dr. To
    // exercise it as a genuine, independent finite-difference check
    // (rather than a self-consistency check reusing the same solve),
    // fabricate a per-atom chi(r) that is an explicit *linear* function of
    // the atom's own position via one fake "symmetry function" per
    // Cartesian component, with dG/dr = identity and dChi/dG = k (so
    // calculateDChidr() returns exactly k, the true gradient of
    // chi(r) = k . r). No neighbor-based contribution is exercised
    // (neighbor lists are left empty, so the cross term -- d(chi_j)/dr_p
    // for j != p -- is correctly zero).
    Vec3D const k0(0.4, -0.2, 0.1);
    Vec3D const k1(-0.3, 0.5, -0.1);

    auto setChiFromLinearModel = [&]()
    {
        s.atoms.at(0).chi = k0 * s.atoms.at(0).r;
        s.atoms.at(1).chi = k1 * s.atoms.at(1).r;
    };

    for (size_t i = 0; i < 2; ++i)
    {
        Atom& a = s.atoms.at(i);
        Vec3D const& k = (i == 0 ? k0 : k1);
        a.numSymmetryFunctions = 3;
        a.dChidG = {k[0], k[1], k[2]};
        a.dGdr = {Vec3D(1.0, 0.0, 0.0), Vec3D(0.0, 1.0, 0.0),
                  Vec3D(0.0, 0.0, 1.0)};
    }
    setChiFromLinearModel();

    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);
    s.calculateDAdrQ(ewaldSetup, gammaSqrt2, fourPiEps, erfcBuf);

    vector<size_t> const atomIndices = {0, 0, 0, 1, 1, 1};
    vector<size_t> const compIndices = {0, 1, 2, 0, 1, 2};
    vector<Element> const elements   = {Element(0, em)};
    double const maxCutoffRadius     = 10.0;

    s.calculateDQdr(atomIndices, compIndices, maxCutoffRadius, elements);

    double const h = 1.0E-6;
    for (size_t p = 0; p < 2; ++p)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            double const orig = s.atoms.at(p).r[c];

            s.atoms.at(p).r[c] = orig + h;
            setChiFromLinearModel();
            s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                           sigmaSqrtPi, fs, fourPiEps,
                                           erfcBuf);
            double const Q0Hi = s.atoms.at(0).charge;
            double const Q1Hi = s.atoms.at(1).charge;

            s.atoms.at(p).r[c] = orig - h;
            setChiFromLinearModel();
            s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                           sigmaSqrtPi, fs, fourPiEps,
                                           erfcBuf);
            double const Q0Lo = s.atoms.at(0).charge;
            double const Q1Lo = s.atoms.at(1).charge;

            s.atoms.at(p).r[c] = orig;
            setChiFromLinearModel();

            double const dQ0dr = (Q0Hi - Q0Lo) / (2.0 * h);
            double const dQ1dr = (Q1Hi - Q1Lo) / (2.0 * h);

            BOOST_TEST_INFO("perturbed atom " << p << " component " << c);
            BOOST_REQUIRE_SMALL(s.atoms.at(p).dQdr.at(0)[c] - dQ0dr,
                                accuracyNumeric);
            BOOST_REQUIRE_SMALL(s.atoms.at(p).dQdr.at(1)[c] - dQ1dr,
                                accuracyNumeric);
        }
    }
}

// The QR factorization of A is computed once in calculateElectrostaticEnergy()
// and reused by the later solves: after A is rebuilt (here with another
// hardness and geometry) they have to use the new factorization, and
// clearElectrostatics() has to release it together with A.
BOOST_FIXTURE_TEST_CASE(StoredFactorization_FollowsRebuiltMatrix,
                        ElectrostaticsFixture)
{
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);
    hardness(0) = 3.0;
    s.atoms.at(1).r = Vec3D(1.5, 0.5, 0.0);
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);

    vector<VectorXd> dQdChi;
    s.calculateDQdChi(dQdChi);
    vector<VectorXd> dQdJ;
    s.calculateDQdJ(dQdJ);
    BOOST_REQUIRE_EQUAL(dQdChi.size(), 2);
    BOOST_REQUIRE_EQUAL(dQdJ.size(), 1);
    for (size_t k = 0; k < 2; ++k)
    {
        VectorXd b = VectorXd::Zero(3);
        b(k) = -1.0;
        VectorXd const expected = s.A.colPivHouseholderQr().solve(b);
        for (size_t i = 0; i < 2; ++i)
        {
            BOOST_REQUIRE_SMALL(dQdChi.at(k)(i) - expected(i), 1.0E-14);
        }
    }
    VectorXd b(3);
    b << -s.atoms.at(0).charge, -s.atoms.at(1).charge, 0.0;
    VectorXd const expected = s.A.colPivHouseholderQr().solve(b);
    for (size_t i = 0; i < 2; ++i)
    {
        BOOST_REQUIRE_SMALL(dQdJ.at(0)(i) - expected(i), 1.0E-14);
    }

    s.clearElectrostatics();
    BOOST_REQUIRE_EQUAL(s.AQr.rows(), 0);
}

// Periodic water molecule in a cubic box of edge L (bohr), electronegativities
// set so that the charge equilibration has a non-trivial solution.
Structure periodicWater(ElementMap const& em, double const L)
{
    Structure s;
    s.setElementMap(em);
    string const l = to_string(L);
    s.readFromLines({"begin",
                     "lattice " + l + " 0.0 0.0",
                     "lattice 0.0 " + l + " 0.0",
                     "lattice 0.0 0.0 " + l,
                     "atom 3.0 3.0 3.0 O 0.0 0.0 0.0 0.0 0.0",
                     "atom 4.8 3.0 3.0 H 0.0 0.0 0.0 0.0 0.0",
                     "atom 2.4 4.7 3.0 H 0.0 0.0 0.0 0.0 0.0",
                     "energy 0.0",
                     "charge 0.0",
                     "end"});
    for (auto& a : s.atoms) a.chi = (a.element == em["O"]) ? -0.6 : 0.3;
    return s;
}

// Training calculates the neighbor lists of all structures (and with them the
// Ewald parameters in the shared EwaldSetup) before the first charge
// equilibration. The charges and the electrostatic energy of a structure must
// not depend on which structure was set up last: they have to agree with a
// solve right after the setup of the structure itself (as in prediction).
BOOST_AUTO_TEST_CASE(
    CalculateElectrostaticEnergy_Periodic_UsesEwaldParametersOfThisStructure)
{
    ElementMap em;
    em.registerElements("H O");
    VectorXd hardness(2);
    hardness << 1.0, 1.2;
    VectorXd sigmaSqrtPi(2);
    sigmaSqrtPi << 0.5, 0.6;
    MatrixXd gammaSqrt2(2, 2);
    gammaSqrt2 << 0.8, 0.9, 0.9, 1.0;
    double const fourPiEps = 1.0;
    ScreeningFunction fs;
    fs.setInnerOuter(0.001, 0.002);
    // Optimized eta: eta, rCut and kCut depend on volume and number of atoms.
    EwaldSetup ewaldSetup;
    ewaldSetup.setTruncMethod(EWALDTruncMethod::KOLAFA_PERRAM);
    ewaldSetup.readFromArgs({"1.0E-6", "1.0"});
    ewaldSetup.setMaxQSigma(0.6 / sqrt(M_PI));
    double const cutoffSF = 4.0;
    vector<vector<double>> cutoffs = {{cutoffSF}, {cutoffSF}};
    auto setup = [&](Structure& s)
    {
        s.calculateMaxCutoffRadiusOverall(ewaldSetup, fs.getOuter(), cutoffSF);
        s.calculateNeighborList(cutoffSF, cutoffs);
    };

    Structure reference = periodicWater(em, 9.0);
    setup(reference);
    ErfcBuf erfcBufReference;
    reference.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                           sigmaSqrtPi, fs, fourPiEps,
                                           erfcBufReference);

    Structure s = periodicWater(em, 9.0);
    Structure large = periodicWater(em, 20.0);
    setup(s);
    setup(large);
    ErfcBuf erfcBufS;
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBufS);

    for (size_t i = 0; i < s.numAtoms; ++i)
    {
        BOOST_TEST_INFO("atom " << i);
        BOOST_REQUIRE_SMALL(s.atoms.at(i).charge
                            - reference.atoms.at(i).charge, 1.0E-12);
    }
    BOOST_REQUIRE_SMALL(s.energyElec - reference.energyElec, 1.0E-12);
}

// Element parameters of the NaCl charge-equilibration reference (atomic units),
// the same as in tools/misc/qeq-reference.py. Elements are sorted by atomic
// number: index 0 = Na, 1 = Cl.
struct NaClReferenceParameters
{
    NaClReferenceParameters() :
        hardness(2),
        sigmaSqrtPi(2),
        gammaSqrt2(2, 2),
        fourPiEps(1.0)
    {
        em.registerElements("Na Cl");
        double const sigma[2] = {1.6, 2.0};
        for (size_t i = 0; i < 2; ++i)
        {
            hardness(i) = 0.40;
            sigmaSqrtPi(i) = sigma[i] * sqrt(M_PI);
            for (size_t j = 0; j < 2; ++j)
            {
                gammaSqrt2(i, j) = sqrt(2.0 * (sigma[i] * sigma[i]
                                               + sigma[j] * sigma[j]));
            }
        }
        fs.setInnerOuter(0.001, 0.002); // No screening.
    }

    void setChi(Structure& s) const
    {
        for (auto& a : s.atoms) a.chi = (a.element == em["Na"]) ? -0.35 : 0.35;
    }

    ElementMap em;
    VectorXd hardness;
    VectorXd sigmaSqrtPi;
    MatrixXd gammaSqrt2;
    double fourPiEps;
    ScreeningFunction fs;
};

// Na-Cl pair without periodic boundary conditions: the charge equilibration has
// a closed-form solution. With Q_Na = -Q_Cl = q, minimizing
// E_Qeq = q^2/2 (J_Na + J_Cl + 1/(sigma_Na sqrt(pi)) + 1/(sigma_Cl sqrt(pi))
//         - 2 erf(r/gamma')/r) + q (chi_Na - chi_Cl)
// gives q = (chi_Cl - chi_Na) / D with D the bracket, gamma' = sqrt(2) gamma.
// n2p2 reports the electrostatic energy without the Gaussian self-energy
// sum_i Q_i^2 / (2 sigma_i sqrt(pi)) of Ko et al. (2021), Eq. 3, i.e. here
// -q^2 erf(r/gamma')/r.
BOOST_FIXTURE_TEST_CASE(CalculateElectrostaticEnergy_NonPeriodicPair_ClosedForm,
                        NaClReferenceParameters)
{
    double const r = 5.0;
    Structure s;
    s.setElementMap(em);
    s.isPeriodic = false;
    Atom a0;
    a0.r = Vec3D(0.0, 0.0, 0.0);
    s.addAtom(a0, "Na");
    Atom a1;
    a1.r = Vec3D(r, 0.0, 0.0);
    s.addAtom(a1, "Cl");
    s.chargeRef = 0.0;
    setChi(s);

    EwaldSetup ewaldSetup;
    ErfcBuf erfcBuf;
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);

    double const pair = erf(r / gammaSqrt2(0, 1)) / r;
    double const D = hardness(0) + hardness(1) + 1.0 / sigmaSqrtPi(0)
                   + 1.0 / sigmaSqrtPi(1) - 2.0 * pair;
    double const q = (0.35 - (-0.35)) / D;
    BOOST_REQUIRE_SMALL(s.atoms.at(0).charge - q, 1.0E-14);
    BOOST_REQUIRE_SMALL(s.atoms.at(1).charge + q, 1.0E-14);
    BOOST_REQUIRE_SMALL(s.energyElec - (-q * q * pair), 1.0E-14);
}

// Periodic disordered rock-salt NaCl (64 atoms, cubic box of 23.8 bohr):
// charges and electrostatic energy against tools/misc/qeq-reference.py, an
// independent implementation of the published equations with the Ewald sum
// converged to machine precision (two splittings agree to 2E-15 e).
BOOST_FIXTURE_TEST_CASE(CalculateElectrostaticEnergy_PeriodicNaCl_IndependentReference,
                        NaClReferenceParameters)
{
    // Structure built as in qeq-reference.py: rock-salt sites of lattice
    // constant 11.9 bohr, shifted by 0.3 sin(1.3 idx + 0.7 c + 0.1) bohr.
    size_t const nCell = 2;
    double const a = 11.9;
    double const L = nCell * a;
    vector<string> lines = {"begin"};
    for (size_t c = 0; c < 3; ++c)
    {
        ostringstream l;
        l << setprecision(17) << "lattice";
        for (size_t d = 0; d < 3; ++d) l << " " << (c == d ? L : 0.0);
        lines.push_back(l.str());
    }
    size_t const n = 2 * nCell;
    size_t idx = 0;
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            for (size_t k = 0; k < n; ++k)
            {
                size_t const site[3] = {i, j, k};
                ostringstream l;
                l << setprecision(17) << "atom";
                for (size_t d = 0; d < 3; ++d)
                {
                    l << " " << site[d] * a / 2
                         + 0.3 * sin(1.3 * idx + 0.7 * d + 0.1);
                }
                l << ((i + j + k) % 2 == 0 ? " Na" : " Cl")
                  << " 0.0 0.0 0.0 0.0 0.0";
                lines.push_back(l.str());
                idx++;
            }
        }
    }
    lines.insert(lines.end(), {"energy 0.0", "charge 0.0", "end"});
    Structure s;
    s.setElementMap(em);
    s.readFromLines(lines);
    setChi(s);

    EwaldSetup ewaldSetup;
    ewaldSetup.setTruncMethod(EWALDTruncMethod::KOLAFA_PERRAM);
    ewaldSetup.readFromArgs({"1.0E-12", "1.0"});
    ewaldSetup.setMaxQSigma(2.0);
    double const cutoffSF = 4.0;
    vector<vector<double>> cutoffs = {{cutoffSF}, {cutoffSF}};
    s.calculateMaxCutoffRadiusOverall(ewaldSetup, fs.getOuter(), cutoffSF);
    s.calculateNeighborList(cutoffSF, cutoffs);
    ErfcBuf erfcBuf;
    s.calculateElectrostaticEnergy(ewaldSetup, hardness, gammaSqrt2,
                                   sigmaSqrtPi, fs, fourPiEps, erfcBuf);

    vector<double> const QRef = {
        7.748029004662423e-01, -8.055403050131953e-01, 8.071444391755310e-01,
        -7.826145840182414e-01, -7.830115543769592e-01, 7.932985500440645e-01,
        -8.041428096746129e-01, 7.906084137136117e-01, 7.865299804746939e-01,
        -7.843568360743498e-01, 7.960735286911221e-01, -8.015625929018663e-01,
        -8.094526943562262e-01, 7.728674857744114e-01, -7.700143709104463e-01,
        8.099178846234159e-01, -8.083450485289324e-01, 8.027078517173110e-01,
        -7.852692184474613e-01, 7.812690047971037e-01, 7.885032036678894e-01,
        -7.971433742359507e-01, 7.915805709952733e-01, -7.921828783772585e-01,
        -7.856389645976772e-01, 7.961712067254048e-01, -7.959071318336896e-01,
        7.883798611984164e-01, 7.733290479810073e-01, -7.857945637851560e-01,
        8.100773657203300e-01, -7.992166465160530e-01, 8.086374604665425e-01,
        -7.763665205006746e-01, 7.805525236197565e-01, -7.956676067659284e-01,
        -7.974754163335451e-01, 7.906661344116106e-01, -7.881339007079030e-01,
        7.981250790197548e-01, 7.932632419928004e-01, -7.930972526576104e-01,
        7.896065700056747e-01, -7.969869502458121e-01, -7.913898396827915e-01,
        8.115820265636313e-01, -7.942058533505443e-01, 7.725529575630636e-01,
        -7.729533332966243e-01, 7.841191432221931e-01, -7.959031454283758e-01,
        8.051973887502387e-01, 8.010976812423193e-01, -7.914295028055278e-01,
        7.892575455140064e-01, -7.825537028229593e-01, -8.029235915045421e-01,
        7.956895570633724e-01, -7.913309433563283e-01, 7.836910109479117e-01,
        7.957274724402520e-01, -7.895928585648652e-01, 7.869153872824787e-01,
        -7.997384841993270e-01
    };
    // Ko et al. (2021), Eq. 3, minus the Gaussian self-energy (n2p2 convention).
    double const energyElecRef = -5.535827198786720E+00;

    BOOST_REQUIRE_EQUAL(s.numAtoms, QRef.size());
    for (size_t i = 0; i < s.numAtoms; ++i)
    {
        BOOST_TEST_INFO("atom " << i);
        BOOST_REQUIRE_SMALL(s.atoms.at(i).charge - QRef.at(i), 1.0E-9);
    }
    BOOST_REQUIRE_SMALL(s.energyElec - energyElecRef, 1.0E-9);
}

BOOST_AUTO_TEST_SUITE_END()
