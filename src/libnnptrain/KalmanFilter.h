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

#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include <mpi.h>
#include "Updater.h"
#include <Eigen/Core>
#include <cstddef>
#include <string>
#include <vector>

namespace nnp
{

/// Implementation of the Kalman filter method.
class KalmanFilter : public Updater
{
public:
    /// Enumerate different Kalman filter types.
    enum KalmanType
    {
        /// Regular Kalman filter.
        KT_STANDARD,
        /// Kalman filtering with fading memory modification.
        KT_FADINGMEMORY
    };

    /** Kalman filter class constructor.
     *
     * @param[in] sizeState Size of the state vector #w.
     * @param[in] type Kalman filter type (see #KalmanType).
     */
    KalmanFilter(std::size_t const sizeState,
                 KalmanType const  type);
    /** Destructor.
     */
    virtual ~KalmanFilter();
    /** Set observation vector size.
     *
     * @param[in] size Size of the observation vector.
     *
     * If the size of the observation vector is known in advance it can be set
     * here.
     */
    void                     setSizeObservation(
                                            std::size_t const sizeObservation);
    /** Set pointer to current state.
     *
     * @param[in,out] state Pointer to state vector (weights vector), will be
     *                      changed in-place upon calling update().
     */
    void                     setState(double* state);
    /** Set pointer to current error vector.
     *
     * @param[in] error Pointer to error (difference between reference and
     *                  neural network potential output).
     */
    void                     setError(double const* const error);
    /** Set pointer to current error vector.
     *
     * @param[in] error Pointer to error (difference between reference and
     *                  neural network potential output).
     * @param[in] size Number of error vector entries.
     */
    void                     setError(double const* const error,
                                      std::size_t const   size);
    /** Set pointer to current Jacobi matrix.
     *
     * @param[in] jacobian Derivatives of error with respect to weights.
     */
    void                     setJacobian(double const* const jacobian);
    /** Set pointer to current Jacobi matrix.
     *
     * @param[in] jacobian Derivatives of error with respect to weights.
     * @param[in] columns Number of gradients provided.
     *
     * @note
     * If there are @f$m@f$ errors and @f$n@f$ weights, the Jacobi matrix
     * is a @f$n \times m@f$ matrix stored in column-major order.
     */
    void                     setJacobian(double const* const jacobian,
                                         std::size_t const   columns);
    /** Update error covariance matrix and state vector.
     */
    void                     update();
    /** Update error covariance matrix and state vector.
     *
     * @param[in] sizeObservation Size of the observation vector.
     */
    void                     update(std::size_t const sizeObservation);
    /** Set parameters for standard Kalman filter.
     *
     * @param[in] epsilon Error covariance initialization parameter
     *                    @f$\epsilon@f$.
     * @param[in] q0 Process noise initial value @f$q_0@f$.
     * @param[in] qtau Process noise exponential decay parameter
     *                 @f$q_{\tau}@f$.
     * @param[in] qmin Process noise minimum value @f$q_{\text{min}}@f$.
     * @param[in] eta0 Initial learning rate @f$\eta_0@f$.
     * @param[in] etatau Learning rate exponential increase parameter
     *                   @f$\eta_{\tau}@f$.
     * @param[in] etamax Learning rate maximum value @f$\eta_{\text{max}}@f$.
     */
    void                     setParametersStandard(double const epsilon,
                                                   double const q0,
                                                   double const qtau,
                                                   double const qmin,
                                                   double const eta0,
                                                   double const etatau,
                                                   double const etamax);
    /** Set parameters for fading memory Kalman filter.
     *
     * @param[in] epsilon Error covariance initialization parameter
     *                    @f$\epsilon@f$.
     * @param[in] q0 Process noise initial value @f$q_0@f$.
     * @param[in] qtau Process noise exponential decay parameter
     *                 @f$q_{\tau}@f$.
     * @param[in] qmin Process noise minimum value @f$q_{\text{min}}@f$.
     * @param[in] lambda Forgetting factor @f$\lambda@f$.
     * @param[in] nu Fading memory parameter @f$\nu@f$.
     */
    void                     setParametersFadingMemory(double const epsilon,
                                                       double const q0,
                                                       double const qtau,
                                                       double const qmin,
                                                       double const lambda,
                                                       double const nu);
    /** Status report.
     *
     * @param[in] epoch Current epoch.
     *
     * @return Line with current status information.
     */
    std::string              status(std::size_t epoch) const;
    /** Header for status report file.
     *
     * @return Vector with header lines.
     */
    std::vector<std::string> statusHeader() const;
    /** Information about Kalman filter settings.
     *
     * @return Vector with info lines.
     */
    std::vector<std::string> info() const;
    /** Getter for #type.
     */
    KalmanType               getType() const;
    /** Getter for #sizeObservation.
     */
    std::size_t              getSizeObservation() const;
    /** Getter for #numUpdates.
     */
    std::size_t              getNumUpdates() const;
    /** Getter for #eta.
     */
    double                   getEta() const;
    /** Getter for #epsilon.
     */
    double                   getEpsilon() const;
    /** Getter for #q0.
     */
    double                   getQ0() const;
    /** Getter for #qtau.
     */
    double                   getQtau() const;
    /** Getter for #qmin.
     */
    double                   getQmin() const;
    /** Getter for #lambda.
     */
    double                   getLambda() const;
    /** Getter for #nu.
     */
    double                   getNu() const;
    /** Getter for #gamma.
     */
    double                   getGamma() const;

private:
    /// Kalman filter type.
    KalmanType                         type;
    /// Size of observation (measurement) vector.
    std::size_t                        sizeObservation;
    /// Total number of updates performed.
    std::size_t                        numUpdates;
    /// Error covariance initialization parameter @f$\epsilon@f$.
    double                             epsilon;
    /// Process noise @f$q@f$.
    double                             q;
    /// Process noise initial value @f$q_0@f$.
    double                             q0;
    /// Process noise exponential decay parameter @f$q_{\tau}@f$.
    double                             qtau;
    /// Process noise minimum value @f$q_{\text{min}}@f$.
    double                             qmin;
    /// Learning rate @f$\eta@f$.
    double                             eta;
    /// Learning rate initial value @f$\eta_0@f$.
    double                             eta0;
    /// Learning rate exponential increase parameter @f$\eta_{\tau}@f$.
    double                             etatau;
    /// Learning rate maximum value @f$\eta_{\text{max}}@f$.
    double                             etamax;
    /// Forgetting factor for fading memory Kalman filter.
    double                             lambda;
    /// Parameter for fading memory Kalman filter.
    double                             nu;
    /// Forgetting gain factor gamma for fading memory Kalman filter.
    double                             gamma;
    /// State vector.
    Eigen::Map<Eigen::VectorXd>*       w;
    /// Error vector.
    Eigen::Map<Eigen::VectorXd const>* xi;
    /// Derivative matrix.
    Eigen::Map<Eigen::MatrixXd const>* H;
    /// Error covariance matrix.
    Eigen::MatrixXd                    P;
    /// Kalman gain matrix.
    Eigen::MatrixXd                    K;
    /// Intermediate result X = P . H.
    Eigen::MatrixXd                    X;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline KalmanFilter::KalmanType KalmanFilter::getType() const
{
    return type;
}

inline std::size_t KalmanFilter::getSizeObservation() const
{
    return sizeObservation;
}

inline std::size_t KalmanFilter::getNumUpdates() const
{
    return numUpdates;
}

inline double KalmanFilter::getEta() const
{
    return eta;
}

inline double KalmanFilter::getEpsilon() const
{
    return epsilon;
}

inline double KalmanFilter::getQ0() const
{
    return q0;
}

inline double KalmanFilter::getQtau() const
{
    return qtau;
}

inline double KalmanFilter::getQmin() const
{
    return qmin;
}

inline double KalmanFilter::getLambda() const
{
    return lambda;
}

inline double KalmanFilter::getNu() const
{
    return nu;
}

}

#endif
