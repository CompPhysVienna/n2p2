// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "KalmanFilter.h"
#include "utility.h"
#include "mpi-extra.h"
#include <Eigen/LU>
#include <iostream>
#include <stdexcept>
#include <map>       // std::map

using namespace Eigen;
using namespace std;
using namespace nnp;

KalmanFilter::KalmanFilter(size_t const sizeState,
                           KalmanType const type) :
    Updater(sizeState),
    myRank         (0   ),
    numProcs       (0   ),
    sizeObservation(0   ),
    numUpdates     (0   ),
    sizeP          (0   ),
    epsilon        (0.0 ),
    q              (0.0 ),
    q0             (0.0 ),
    qtau           (0.0 ),
    qmin           (0.0 ),
    eta            (0.0 ),
    eta0           (0.0 ),
    etatau         (0.0 ),
    etamax         (0.0 ),
    lambda         (0.0 ),
    nu             (0.0 ),
    gamma          (0.0 ),
    w              (NULL),
    xi             (NULL),
    H              (NULL)
{
    if (!(type == KT_STANDARD ||
          type == KT_FADINGMEMORY))
    {
        throw runtime_error("ERROR: Unknown Kalman filter type.\n");
    }

    if (sizeState < 1)
    {
        throw runtime_error("ERROR: Wrong Kalman filter dimensions.\n");
    }

    this->type            = type;
    sizeObservation = 1;

    w  = new Map<VectorXd      >(0, sizeState);
    xi = new Map<VectorXd const>(0, sizeObservation);
    H  = new Map<MatrixXd const>(0, sizeState, sizeObservation);
}

KalmanFilter::~KalmanFilter()
{
}

void KalmanFilter::setupMPI(MPI_Comm* communicator)
{
    MPI_Comm_dup(*communicator, &comm);
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &numProcs);

    return;
}

void KalmanFilter::setupDecoupling(vector<pair<size_t, size_t>> groupLimits)
{
    this->groupLimits = groupLimits;

    // Check consistency of decoupling group limits.
    if (groupLimits.at(0).first != 0)
    {
        auto const& l = groupLimits.at(0);
        throw runtime_error(strpr("ERROR: Inconsistent decoupling group "
                                  "limits, first group must start with index "
                                  "0 (is %zu).\n",
                                  l.first));

    }
    for (size_t i = 0; i < groupLimits.size(); ++i)
    {
        auto const& l = groupLimits.at(i);
        if (l.first > l.second)
        {
            throw runtime_error(
                strpr("ERROR: Inconsistent decoupling group limits, "
                      "group %zu: start %zu > end %zu.\n",
                      i, l.first, l.second));
        }
    }
    for (size_t i = 0; i < groupLimits.size() - 1; ++i)
    {
        auto const& l = groupLimits.at(i);
        auto const& r = groupLimits.at(i + 1);
        if (l.second + 1 != r.first)
        {
            throw runtime_error(
                strpr("ERROR: Inconsistent decoupling group limits, "
                      "group %zu end %zu + 1 != group %zu start %zu.\n",
                      i, l.second, i + 1, r.first));
        }
    }
    if (groupLimits.back().second != sizeState - 1)
    {
        auto const& l = groupLimits.back();
        throw runtime_error(strpr("ERROR: Inconsistent decoupling group "
                                  "limits, last group must end with index "
                                  "%zu (is %zu).\n",
                                  sizeState - 1,
                                  l.second));

    }

    // Distribute groups to MPI procs.
    numGroupsPerProc.resize(numProcs, 0);
    groupOffsetPerProc.resize(numProcs, 0);
    int quotient = groupLimits.size() / numProcs;
    int remainder = groupLimits.size() % numProcs;
    int groupSum = 0;
    for (int i = 0; i < numProcs; i++)
    {
        numGroupsPerProc.at(i) = quotient;
        if (remainder > 0 && i < remainder)
        {
            numGroupsPerProc.at(i)++;
        }
        groupOffsetPerProc.at(i) = groupSum;
        groupSum += numGroupsPerProc.at(i);
    }
    for (size_t i = 0; i < numGroupsPerProc.at(myRank); ++i)
    {
        size_t index = groupOffsetPerProc.at(myRank) + i;
        size_t size = groupLimits.at(index).second
                    - groupLimits.at(index).first + 1;
        myGroups.push_back(make_pair(index, size));
    }

    // Allocate per-processor matrices for all groups.
    for (auto g : myGroups)
    {
        P.push_back(Eigen::MatrixXd(g.second, g.second));
        P.back().setIdentity();

        X.push_back(Eigen::MatrixXd(g.second, sizeObservation));

        // Prevent problems with unallocated K when log starts.
        K.push_back(Eigen::MatrixXd(g.second, sizeObservation));
        K.back().setZero();

        sizeP += g.second * g.second;
    }

    MPI_Allreduce(MPI_IN_PLACE, &sizeP, 1, MPI_SIZE_T, MPI_SUM, comm);

    return;
}

void KalmanFilter::setSizeObservation(size_t const size)
{
    sizeObservation = size;

    return;
}

void KalmanFilter::setState(double* state)
{
    new (w) Map<VectorXd>(state, sizeState);

    return;
}

void KalmanFilter::setError(double const* const error)
{
    setError(error, sizeObservation);

    return;
}

void KalmanFilter::setError(double const* const error, size_t const size)
{
    new (xi) Map<VectorXd const>(error, size);

    return;
}

void KalmanFilter::setJacobian(double const* const jacobian)
{
    setJacobian(jacobian, sizeObservation);

    return;
}

void KalmanFilter::setJacobian(double const* const jacobian,
                               size_t const columns)
{
    new (H) Map<MatrixXd const>(jacobian, sizeState, columns);

    return;
}

void KalmanFilter::update()
{
    update(sizeObservation);

    return;
}

void KalmanFilter::update(size_t const sizeObservation)
{
    X.at(0).resize(sizeState, sizeObservation);

    // Calculate temporary result.
    // X = P . H
    X.at(0) = P.at(0).selfadjointView<Lower>() * (*H);

    // Calculate scaling matrix.
    // A = H^T . X
    MatrixXd A = H->transpose() * X.at(0);

    // Increase learning rate.
    // eta(n) = eta(0) * exp(n * tau)
    if (type == KT_STANDARD && eta < etamax) eta *= exp(etatau);

    // Add measurement noise.
    // A = A + R
    if (type == KT_STANDARD)
    {
        A.diagonal() += VectorXd::Constant(sizeObservation, 1.0 / eta);
    }
    else if (type == KT_FADINGMEMORY)
    {
        A.diagonal() += VectorXd::Constant(sizeObservation, lambda);
    }

    // Calculate Kalman gain matrix.
    // K = X . A^-1
    K.at(0).resize(sizeState, sizeObservation);
    K.at(0) = X.at(0) * A.inverse();

    // Update error covariance matrix.
    // P = P - K . X^T
    P.at(0).noalias() -= K.at(0) * X.at(0).transpose();

    // Apply forgetting factor.
    if (type == KT_FADINGMEMORY)
    {
        P.at(0) *= 1.0 / lambda;
    }
    // Add process noise.
    // P = P + Q
    P.at(0).diagonal() += VectorXd::Constant(sizeState, q);

    // Update state vector.
    // w =  w + K . xi
    (*w) += K.at(0) * (*xi);

    // Anneal process noise.
    // q(n) = q(0) * exp(-n * tau)
    if (q > qmin) q *= exp(-qtau);

    // Update forgetting factor.
    if (type == KT_FADINGMEMORY)
    {
        lambda = nu * lambda + 1.0 - nu;
        gamma = 1.0 / (1.0 + lambda / gamma);
    }

    numUpdates++;

    return;
}

void KalmanFilter::setParametersStandard(double const epsilon,
                                         double const q0,
                                         double const qtau,
                                         double const qmin,
                                         double const eta0,
                                         double const etatau,
                                         double const etamax)
{
    this->epsilon = epsilon;
    this->q0      = q0     ;
    this->qtau    = qtau   ;
    this->qmin    = qmin   ;
    this->eta0    = eta0   ;
    this->etatau  = etatau ;
    this->etamax  = etamax ;

    q = q0;
    eta = eta0;
    for (auto& p : P)
    {
        p /= epsilon;
    }

    return;
}

void KalmanFilter::setParametersFadingMemory(double const epsilon,
                                             double const q0,
                                             double const qtau,
                                             double const qmin,
                                             double const lambda,
                                             double const nu)
{
    this->epsilon = epsilon;
    this->q0      = q0     ;
    this->qtau    = qtau   ;
    this->qmin    = qmin   ;
    this->lambda  = lambda ;
    this->nu      = nu     ;

    q = q0;
    for (auto& p : P)
    {
        p /= epsilon;
    }
    gamma = 1.0;

    return;
}

string KalmanFilter::status(size_t epoch) const
{

    double Pasym    = 0.0;
    double Pdiag    = 0.0;
    double Poffdiag = 0.0;
    double Kmean    = 0.0;

    for (size_t i = 0; i < myGroups.size(); ++i)
    {
        auto const& p = P.at(i);
        auto const& k = K.at(i);
        Pasym += 0.5 * (p - p.transpose()).array().abs().sum();
        double pdiag = p.diagonal().array().abs().sum(); 
        Pdiag += pdiag; 
        Poffdiag += p.array().abs().sum() - pdiag;
        Kmean += k.array().abs().sum();
    }

    MPI_Allreduce(MPI_IN_PLACE, &Pasym   , 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &Pdiag   , 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &Poffdiag, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &Kmean   , 1, MPI_DOUBLE, MPI_SUM, comm);
    Pasym /= (sizeP - sizeState);
    Pdiag /= sizeState;
    Poffdiag /= (sizeP - sizeState);
    Kmean /= (sizeState * sizeObservation);

    string s = strpr("%10zu %10zu %16.8E %16.8E %16.8E %16.8E %16.8E",
                     epoch, numUpdates, Pdiag, Poffdiag, Pasym, Kmean, q);
    if (type == KT_STANDARD)
    {
        s += strpr(" %16.8E", eta);
    }
    else if (type == KT_FADINGMEMORY)
    {
        s += strpr(" %16.8E %16.8E", lambda, numUpdates * gamma);
    }
    s += '\n';

    return s;
}

vector<string> KalmanFilter::statusHeader() const
{
    vector<string> header;

    vector<string> title;
    vector<string> colName;
    vector<string> colInfo;
    vector<size_t> colSize;
    title.push_back("Kalman filter status report.");
    colSize.push_back(10);
    colName.push_back("epoch");
    colInfo.push_back("Training epoch.");
    colSize.push_back(10);
    colName.push_back("nupdates");
    colInfo.push_back("Number of updates performed.");
    colSize.push_back(16);
    colName.push_back("Pdiag");
    colInfo.push_back("Mean of absolute diagonal values of error covariance "
                      "matrix P.");
    colSize.push_back(16);
    colName.push_back("Poffdiag");
    colInfo.push_back("Mean of absolute off-diagonal values of error "
                      "covariance matrix P.");
    colSize.push_back(16);
    colName.push_back("Pasym");
    colInfo.push_back("Asymmetry of P, i.e. mean of absolute values of "
                      "asymmetric part 0.5*(P - P^T).");
    colSize.push_back(16);
    colName.push_back("Kmean");
    colInfo.push_back("Mean of abolute compontents of Kalman gain matrix K.");
    colSize.push_back(16);
    colName.push_back("q");
    colInfo.push_back("Magnitude of process noise (= diagonal entries of Q).");
    if (type == KT_STANDARD)
    {
        colSize.push_back(16);
        colName.push_back("eta");
        colInfo.push_back("Learning rate.");
    }
    else if (type == KT_FADINGMEMORY)
    {
        colSize.push_back(16);
        colName.push_back("lambda");
        colInfo.push_back("Forgetting factor for fading memory KF.");
        colSize.push_back(16);
        colName.push_back("kgamma");
        colInfo.push_back("Forgetting gain k * gamma(k).");
    }
    header = createFileHeader(title, colSize, colName, colInfo);

    return header;
}

vector<string> KalmanFilter::info() const
{
    vector<string> v;

    if (type == KT_STANDARD)
    {
        v.push_back(strpr("KalmanType::KT_STANDARD (%d)\n", type));
        v.push_back(strpr("myRank          = %d\n", myRank));
        v.push_back(strpr("numProcs        = %d\n", numProcs));
        v.push_back(strpr("epsilon         = %12.4E\n", epsilon));
        v.push_back(strpr("q0              = %12.4E\n", q0     ));
        v.push_back(strpr("qtau            = %12.4E\n", qtau   ));
        v.push_back(strpr("qmin            = %12.4E\n", qmin   ));
        v.push_back(strpr("eta0            = %12.4E\n", eta0   ));
        v.push_back(strpr("etatau          = %12.4E\n", etatau ));
        v.push_back(strpr("etamax          = %12.4E\n", etamax ));
    }
    else if (type == KT_FADINGMEMORY)
    {
        v.push_back(strpr("KalmanType::KT_FADINGMEMORY (%d)\n", type));
        v.push_back(strpr("sizeState       = %zu\n", sizeState));
        v.push_back(strpr("sizeObservation = %zu\n", sizeObservation));
        v.push_back(strpr("epsilon         = %12.4E\n", epsilon));
        v.push_back(strpr("q0              = %12.4E\n", q0     ));
        v.push_back(strpr("qtau            = %12.4E\n", qtau   ));
        v.push_back(strpr("qmin            = %12.4E\n", qmin   ));
        v.push_back(strpr("lambda          = %12.4E\n", lambda));
        v.push_back(strpr("nu              = %12.4E\n", nu    ));
    }
    v.push_back(strpr("sizeState       = %zu\n", sizeState));
    v.push_back(strpr("sizeObservation = %zu\n", sizeObservation));
    v.push_back(strpr("OpenMP threads used: %d\n", nbThreads()));
    v.push_back(strpr("Number of decoupling groups: %zu\n",
                      groupLimits.size()));
    v.push_back(strpr("P matrix size ratio (compared to GEKF): %6.2f\n",
                      100.0 * sizeP / (sizeState * sizeState)));
    for (size_t i = 0; i < groupLimits.size(); ++i)
    {
        v.push_back(strpr(" - group %5zu size: %zu\n",
                          i,
                          groupLimits.at(i).second
                          - groupLimits.at(i).first + 1));
    }
    v.push_back(strpr("Number of decoupling groups of proc %d: %zu\n",
                      myRank, myGroups.size()));
    for (auto g : myGroups)
    {
        v.push_back(strpr(" - group %5zu size: %zu\n", g.first, g.second));
    }

    return v;
}
