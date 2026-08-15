#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Kspace
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "EwaldSetup.h"
#include "Kspace.h"
#include "Vec3D.h"
#include <cmath>  // std::exp, std::fabs
#include <limits> // std::numeric_limits

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();
double const accuracyNumeric = 1E-10;

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Kvector_DefaultConstructor_Zero)
{
    Kvector kv;
    BOOST_REQUIRE_SMALL(kv.k[0], accuracy);
    BOOST_REQUIRE_SMALL(kv.k[1], accuracy);
    BOOST_REQUIRE_SMALL(kv.k[2], accuracy);
    BOOST_REQUIRE_SMALL(kv.knorm2, accuracy);
    BOOST_REQUIRE_SMALL(kv.coeff, accuracy);
}

BOOST_AUTO_TEST_CASE(Kvector_VectorConstructor_KnormIsSquaredNorm)
{
    // "knorm2" is used everywhere else in Kspace.cpp (the cutoff check
    // kv.norm2() < kCut*kCut, and the Gaussian coefficient formula
    // exp(-0.5*eta*eta*knorm2)) as the SQUARED norm. The constructor must
    // be consistent with that, not the plain (unsquared) norm.
    Vec3D const v(3.0, 4.0, 0.0); // norm = 5, norm2 = 25.
    Kvector kv(v);

    BOOST_REQUIRE_SMALL(kv.knorm2 - v.norm2(), accuracyNumeric);
}

BOOST_AUTO_TEST_CASE(KspaceGrid_DefaultConstructor_Zero)
{
    KspaceGrid grid;
    BOOST_REQUIRE_SMALL(grid.eta, accuracy);
    BOOST_REQUIRE_SMALL(grid.kCut, accuracy);
    BOOST_REQUIRE_SMALL(grid.volume, accuracy);
    BOOST_REQUIRE_SMALL(grid.pre, accuracy);
    BOOST_REQUIRE(grid.kvectors.empty());
}

BOOST_AUTO_TEST_CASE(Setup_CubicBox_CorrectReciprocalBoxAndPrefactor)
{
    Vec3D box[3];
    box[0] = Vec3D(10.0, 0.0, 0.0);
    box[1] = Vec3D(0.0, 10.0, 0.0);
    box[2] = Vec3D(0.0, 0.0, 10.0);

    EwaldSetup ewaldSetup;
    ewaldSetup.params.eta = 1.0;
    ewaldSetup.params.kCut = 1.0;

    KspaceGrid grid;
    grid.setup(box, ewaldSetup);

    double const volumeExpected = 1000.0;
    double const preExpected = 2.0 * M_PI / volumeExpected;
    BOOST_REQUIRE_SMALL(grid.volume - volumeExpected, accuracyNumeric);
    BOOST_REQUIRE_SMALL(grid.pre - preExpected, accuracyNumeric);

    // Reciprocal box of a cubic real-space box of side L is diagonal with
    // entries 2*pi/L.
    double const kboxDiag = 2.0 * M_PI / 10.0;
    BOOST_REQUIRE_SMALL(grid.kbox[0][0] - kboxDiag, accuracyNumeric);
    BOOST_REQUIRE_SMALL(grid.kbox[1][1] - kboxDiag, accuracyNumeric);
    BOOST_REQUIRE_SMALL(grid.kbox[2][2] - kboxDiag, accuracyNumeric);
    BOOST_REQUIRE_SMALL(grid.kbox[0][1], accuracyNumeric);
    BOOST_REQUIRE_SMALL(grid.kbox[0][2], accuracyNumeric);

    BOOST_REQUIRE_SMALL(grid.eta - 1.0, accuracy);
    BOOST_REQUIRE_SMALL(grid.kCut - 1.0, accuracy);
}

BOOST_AUTO_TEST_CASE(Setup_AllKvectorsWithinCutoff)
{
    Vec3D box[3];
    box[0] = Vec3D(10.0, 0.0, 0.0);
    box[1] = Vec3D(0.0, 10.0, 0.0);
    box[2] = Vec3D(0.0, 0.0, 10.0);

    EwaldSetup ewaldSetup;
    ewaldSetup.params.eta = 1.0;
    ewaldSetup.params.kCut = 1.0;

    KspaceGrid grid;
    grid.setup(box, ewaldSetup);

    BOOST_REQUIRE(!grid.kvectors.empty());
    double const kCut2 = ewaldSetup.params.kCut * ewaldSetup.params.kCut;
    for (auto const& kv : grid.kvectors)
    {
        BOOST_REQUIRE(kv.knorm2 < kCut2);
        // Field must actually hold the squared norm of the stored vector.
        BOOST_REQUIRE_SMALL(kv.knorm2 - kv.k.norm2(), accuracyNumeric);
    }
}

BOOST_AUTO_TEST_CASE(Setup_AxisAlignedKvector_CorrectCoefficient)
{
    Vec3D box[3];
    box[0] = Vec3D(10.0, 0.0, 0.0);
    box[1] = Vec3D(0.0, 10.0, 0.0);
    box[2] = Vec3D(0.0, 0.0, 10.0);

    EwaldSetup ewaldSetup;
    ewaldSetup.params.eta = 1.0;
    ewaldSetup.params.kCut = 1.0;

    KspaceGrid grid;
    grid.setup(box, ewaldSetup);

    // The (0, 0, 1) reciprocal-lattice point: k = (0, 0, 2*pi/10).
    double const kz = 2.0 * M_PI / 10.0;
    bool found = false;
    for (auto const& kv : grid.kvectors)
    {
        if (fabs(kv.k[0]) < accuracyNumeric &&
            fabs(kv.k[1]) < accuracyNumeric &&
            fabs(kv.k[2] - kz) < accuracyNumeric)
        {
            found = true;
            double const knorm2Expected = kz * kz;
            double const coeffExpected = exp(-0.5 * grid.eta * grid.eta
                                             * knorm2Expected)
                                       * 2.0 * grid.pre / knorm2Expected;
            BOOST_REQUIRE_SMALL(kv.knorm2 - knorm2Expected, accuracyNumeric);
            BOOST_REQUIRE_SMALL(kv.coeff - coeffExpected, accuracyNumeric);
        }
    }
    BOOST_REQUIRE(found);
}

BOOST_AUTO_TEST_SUITE_END()
