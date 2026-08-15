#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE StructureElectrostatics
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "Atom.h"
#include "ElementMap.h"
#include "ErfcBuf.h"
#include "EwaldSetup.h"
#include "ScreeningFunction.h"
#include "Structure.h"
#include "Vec3D.h"
#include <Eigen/Core>
#include <cmath>  // std::erf
#include <limits> // std::numeric_limits
#include <vector> // std::vector

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

BOOST_AUTO_TEST_SUITE_END()
