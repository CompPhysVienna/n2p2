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

#ifndef GRADIENTDESCENT_H
#define GRADIENTDESCENT_H

#include "Updater.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/// Weight updates based on simple gradient descent methods.
class GradientDescent : public Updater
{
public:
    /// Enumerate different gradient descent variants.
    enum DescentType
    {
        /// Fixed step size.
        DT_FIXED,
        /// Adaptive moment estimation (Adam).
        DT_ADAM
    };

    /** %GradientDescent class constructor.
     *
     * @param[in] sizeState Number of neural network connections (weights
     *                      and biases).
     * @param[in] type Descent type used for step size.
     */
    GradientDescent(std::size_t const sizeState, DescentType const type);
    /** Destructor
     */
    virtual ~GradientDescent() {};
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
     * @param[in] size Number of error vector entries.
     */
    void                     setError(double const* const error,
                                      std::size_t const   size = 1);
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
                                         std::size_t const   columns = 1);
    /** Perform connection update.
     *
     * Update the connections via steepest descent method.
     */
    void                     update();
    /** Set parameters for fixed step gradient descent algorithm.
     *
     * @param[in] eta Step size = ratio of gradient subtracted from current
     *                weights.
     */
    void                     setParametersFixed(double const eta);
    /** Set parameters for Adam algorithm.
     *
     * @param[in] eta Step size (corresponds to @f$\alpha@f$ in Adam
     *                publication).
     * @param[in] beta1 Decay rate 1 (first moment).
     * @param[in] beta2 Decay rate 2 (second moment).
     * @param[in] epsilon Small scalar.
     */
    void                     setParametersAdam(double const eta,
                                               double const beta1,
                                               double const beta2,
                                               double const epsilon);
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
    /** Information about gradient descent settings.
     *
     * @return Vector with info lines.
     */
    std::vector<std::string> info() const;

private:
    DescentType         type;
    /// Learning rate @f$\eta@f$.
    double              eta;
    /// Decay rate 1 (Adam).
    double              beta1;
    /// Decay rate 2 (Adam).
    double              beta2;
    /// Small scalar.
    double              epsilon;
    /// Initial learning rate.
    double              eta0;
    /// Decay rate 1 to the power of t (Adam).
    double              beta1t;
    /// Decay rate 2 to the power of t (Adam).
    double              beta2t;
    /// State vector pointer.
    double*             state;
    /// Error pointer (single double value).
    double const*       error;
    /// Gradient vector pointer.
    double const*       gradient;
    /// First moment estimate (Adam).
    std::vector<double> m;
    /// Second moment estimate (Adam).
    std::vector<double> v;
};

}

#endif
