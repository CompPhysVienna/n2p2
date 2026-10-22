#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE KalmanFilter
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "KalmanFilter.h"
#include <algorithm> // std::max
#include <cmath>     // std::exp, std::fabs, std::sin
#include <cstddef>   // std::size_t
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <string>    // std::string
#include <vector>    // std::vector

using namespace std;
using namespace nnp;

namespace bdata = boost::unit_test::data;

double const accuracy = 1000.0 * numeric_limits<double>::epsilon();

BOOST_AUTO_TEST_SUITE(UnitTests)

BOOST_AUTO_TEST_CASE(Constructor_ZeroSizeState_Throws)
{
    BOOST_REQUIRE_THROW(KalmanFilter(0, KalmanFilter::KT_STANDARD),
                        runtime_error);
}

BOOST_AUTO_TEST_CASE(Constructor_UnknownType_Throws)
{
    BOOST_REQUIRE_THROW(
        KalmanFilter(1, static_cast<KalmanFilter::KalmanType>(2)),
        runtime_error);
}

BOOST_AUTO_TEST_CASE(Standard_Update_CorrectValues)
{
    // Scalar case (sizeState = sizeObservation = 1): reference values
    // obtained from an independent Python re-implementation of the exact
    // update formula (see KalmanFilter::update(), KT_STANDARD branch),
    // replicating the same operation order/precision.
    KalmanFilter kf(1, KalmanFilter::KT_STANDARD);
    kf.setParametersStandard(2.0, 0.01, 0.1, 0.001, 1.0, 0.1, 10.0);

    vector<double> state = {1.0};
    double const error = 0.4;
    vector<double> const jacobian = {0.5};

    kf.setState(state.data());
    kf.setError(&error);
    kf.setJacobian(jacobian.data());

    kf.update();
    BOOST_REQUIRE_SMALL(state.at(0) - 1.0971027059695633, accuracy);

    // Second consecutive update: also exercises the learning-rate increase
    // and process-noise annealing recurrences between calls.
    kf.update();
    BOOST_REQUIRE_SMALL(state.at(0) - 1.193618839027742, accuracy);

    BOOST_REQUIRE_EQUAL(kf.getNumUpdates(), 2);
}

BOOST_AUTO_TEST_CASE(FadingMemory_Update_CorrectValues)
{
    // Same scalar-case approach as Standard_Update_CorrectValues, for the
    // KT_FADINGMEMORY branch.
    KalmanFilter kf(1, KalmanFilter::KT_FADINGMEMORY);
    kf.setParametersFadingMemory(2.0, 0.01, 0.1, 0.001, 0.99, 0.9);

    vector<double> state = {1.0};
    double const error = 0.4;
    vector<double> const jacobian = {0.5};

    kf.setState(state.data());
    kf.setError(&error);
    kf.setJacobian(jacobian.data());

    kf.update();

    BOOST_REQUIRE_SMALL(state.at(0) - 1.0896860986547086, accuracy);
    BOOST_REQUIRE_SMALL(kf.getLambda() - 0.991, accuracy);
    BOOST_REQUIRE_SMALL(kf.getGamma() - 0.5022601707684581, accuracy);
}

BOOST_AUTO_TEST_CASE(Standard_InfoAndStatus_ReportCorrectType)
{
    KalmanFilter kf(2, KalmanFilter::KT_STANDARD);
    kf.setParametersStandard(1.0, 0.01, 0.1, 0.001, 1.0, 0.1, 10.0);

    vector<double> state = {0.0, 0.0};
    double const error = 1.0;
    vector<double> const jacobian = {0.3, -0.2};
    kf.setState(state.data());
    kf.setError(&error);
    kf.setJacobian(jacobian.data());
    kf.update();

    vector<string> const info = kf.info();
    BOOST_REQUIRE(!info.empty());
    BOOST_REQUIRE(info.at(0).find("KT_STANDARD") != string::npos);

    string const s = kf.status(3);
    BOOST_REQUIRE(s.find("3") != string::npos);

    vector<string> const header = kf.statusHeader();
    BOOST_REQUIRE(!header.empty());
}

BOOST_AUTO_TEST_CASE(FadingMemory_InfoAndStatus_ReportCorrectType)
{
    KalmanFilter kf(2, KalmanFilter::KT_FADINGMEMORY);
    kf.setParametersFadingMemory(1.0, 0.01, 0.1, 0.001, 0.99, 0.9);

    vector<double> state = {0.0, 0.0};
    double const error = 1.0;
    vector<double> const jacobian = {0.3, -0.2};
    kf.setState(state.data());
    kf.setError(&error);
    kf.setJacobian(jacobian.data());
    kf.update();

    vector<string> const info = kf.info();
    BOOST_REQUIRE(!info.empty());
    BOOST_REQUIRE(info.at(0).find("KT_FADINGMEMORY") != string::npos);

    string const s = kf.status(1);
    BOOST_REQUIRE(s.find("1") != string::npos);

    vector<string> const header = kf.statusHeader();
    BOOST_REQUIRE(!header.empty());
}

BOOST_AUTO_TEST_CASE(Getters_ReflectSetParameters)
{
    KalmanFilter kf(2, KalmanFilter::KT_STANDARD);
    kf.setParametersStandard(2.0, 0.02, 0.05, 0.001, 0.5, 0.2, 5.0);

    BOOST_REQUIRE(kf.getType() == KalmanFilter::KT_STANDARD);
    BOOST_REQUIRE_SMALL(kf.getEpsilon() - 2.0, accuracy);
    BOOST_REQUIRE_SMALL(kf.getQ0() - 0.02, accuracy);
    BOOST_REQUIRE_SMALL(kf.getQtau() - 0.05, accuracy);
    BOOST_REQUIRE_SMALL(kf.getQmin() - 0.001, accuracy);
    BOOST_REQUIRE_SMALL(kf.getEta() - 0.5, accuracy);
    BOOST_REQUIRE_EQUAL(kf.getNumUpdates(), 0);

    kf.setSizeObservation(3);
    BOOST_REQUIRE_EQUAL(kf.getSizeObservation(), 3);
}


// Independent reference implementation of KalmanFilter::update() with plain
// loops and a dense, symmetric covariance matrix P (no Eigen), for a state of
// size n and an observation of size m. Jacobian layout as in KalmanFilter:
// column-major n x m, H(i, j) = jacobian[j * n + i].
struct ReferenceKalman
{
    size_t                 n;
    bool                   fading;
    double                 q, qtau, qmin, eta, etatau, etamax, lambda, nu;
    vector<vector<double>> P;

    ReferenceKalman(size_t n, bool fading, double epsilon) :
        n(n), fading(fading), q(0), qtau(0), qmin(0), eta(0), etatau(0),
        etamax(0), lambda(0), nu(0),
        P(n, vector<double>(n, 0.0))
    {
        for (size_t i = 0; i < n; ++i) P[i][i] = 1.0 / epsilon;
    }

    void update(vector<double>&       w,
                vector<double> const& xi,
                vector<double> const& jacobian)
    {
        size_t const m = xi.size();
        auto H = [&](size_t i, size_t j) { return jacobian[j * n + i]; };
        // X = P . H (n x m), A = H^T . X + R (m x m).
        vector<vector<double>> X(n, vector<double>(m, 0.0));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                for (size_t k = 0; k < n; ++k) X[i][j] += P[i][k] * H(k, j);
        if (!fading && eta < etamax) eta *= exp(etatau);
        vector<vector<double>> A(m, vector<double>(m, 0.0));
        for (size_t a = 0; a < m; ++a)
        {
            for (size_t b = 0; b < m; ++b)
                for (size_t k = 0; k < n; ++k) A[a][b] += H(k, a) * X[k][b];
            A[a][a] += fading ? lambda : 1.0 / eta;
        }
        // Ainv by Gauss-Jordan elimination with partial pivoting.
        vector<vector<double>> Ainv(m, vector<double>(m, 0.0));
        for (size_t a = 0; a < m; ++a) Ainv[a][a] = 1.0;
        for (size_t c = 0; c < m; ++c)
        {
            size_t piv = c;
            for (size_t r = c + 1; r < m; ++r)
                if (fabs(A[r][c]) > fabs(A[piv][c])) piv = r;
            swap(A[c], A[piv]);
            swap(Ainv[c], Ainv[piv]);
            double const d = A[c][c];
            for (size_t k = 0; k < m; ++k) { A[c][k] /= d; Ainv[c][k] /= d; }
            for (size_t r = 0; r < m; ++r)
            {
                if (r == c) continue;
                double const f = A[r][c];
                for (size_t k = 0; k < m; ++k)
                {
                    A[r][k] -= f * A[c][k];
                    Ainv[r][k] -= f * Ainv[c][k];
                }
            }
        }
        // K = X . Ainv, P = P - K . X^T (then forgetting factor), P += q I.
        vector<vector<double>> K(n, vector<double>(m, 0.0));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                for (size_t k = 0; k < m; ++k) K[i][j] += X[i][k] * Ainv[k][j];
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                for (size_t k = 0; k < m; ++k) P[i][j] -= K[i][k] * X[j][k];
                if (fading) P[i][j] /= lambda;
            }
        for (size_t i = 0; i < n; ++i) P[i][i] += q;
        // w = w + K . xi
        for (size_t i = 0; i < n; ++i)
            for (size_t k = 0; k < m; ++k) w[i] += K[i][k] * xi[k];
        if (q > qmin) q *= exp(-qtau);
        if (fading) lambda = nu * lambda + 1.0 - nu;
    }
};

// Deterministic, irregular test inputs for update number u.
void fillInputs(size_t                u,
                vector<double>&       xi,
                vector<double>&       jacobian)
{
    for (size_t j = 0; j < xi.size(); ++j)
        xi[j] = 0.3 * sin(1.7 * (j + 1) + 0.9 * u);
    for (size_t k = 0; k < jacobian.size(); ++k)
        jacobian[k] = sin(0.37 * (k + 1) * (u + 2)) + 0.1 * (k % 3);
}

// Multi-dimensional updates (state 7, observation 3, three consecutive
// updates) against the reference above. Unlike the scalar tests, these see
// errors in matrix layout, transposes and in the use of the lower triangle
// of P (selfadjointView), all of which vanish at size 1.
BOOST_DATA_TEST_CASE(Update_MultiDimensional_MatchesReference,
                     bdata::make({false, true}),
                     fading)
{
    size_t const n = 7;
    size_t const m = 3;
    KalmanFilter kf(n, fading ? KalmanFilter::KT_FADINGMEMORY
                              : KalmanFilter::KT_STANDARD);
    ReferenceKalman ref(n, fading, 0.5);
    ref.q = 0.02;
    ref.qtau = 0.1;
    ref.qmin = 0.001;
    if (fading)
    {
        kf.setParametersFadingMemory(0.5, 0.02, 0.1, 0.001, 0.98, 0.9);
        ref.lambda = 0.98;
        ref.nu = 0.9;
    }
    else
    {
        kf.setParametersStandard(0.5, 0.02, 0.1, 0.001, 0.8, 0.1, 10.0);
        ref.eta = 0.8;
        ref.etatau = 0.1;
        ref.etamax = 10.0;
    }

    vector<double> state(n), stateRef(n);
    for (size_t i = 0; i < n; ++i) state[i] = stateRef[i] = 0.1 * i - 0.2;
    vector<double> xi(m);
    vector<double> jacobian(n * m);
    kf.setState(state.data());
    kf.setSizeObservation(m);
    for (size_t u = 0; u < 3; ++u)
    {
        fillInputs(u, xi, jacobian);
        kf.setError(xi.data(), m);
        kf.setJacobian(jacobian.data(), m);
        kf.update(m);
        ref.update(stateRef, xi, jacobian);
        for (size_t i = 0; i < n; ++i)
        {
            BOOST_TEST_INFO("update " << u << ", state component " << i);
            BOOST_REQUIRE_SMALL(state[i] - stateRef[i],
                                1.0E-12 * max(1.0, fabs(stateRef[i])));
        }
    }
    BOOST_REQUIRE_EQUAL(kf.getNumUpdates(), 3);
}

BOOST_AUTO_TEST_SUITE_END()
