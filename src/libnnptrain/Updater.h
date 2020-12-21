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

#ifndef UPDATER_H
#define UPDATER_H

#include "Stopwatch.h"

#include <cstddef> // std::size_t
#include <map>     // std::map
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/// Base class for different weight update methods.
class Updater
{
public:
    /** Set pointer to current state.
     *
     * @param[in,out] state Pointer to state vector (weights vector), will be
     *                      changed in-place upon calling update().
     */
    virtual void                     setState(double* state) = 0;
    /** Set pointer to current error vector.
     *
     * @param[in] error Pointer to error (difference between reference and
     *                  neural network potential output).
     * @param[in] size Number of error vector entries.
     */
    virtual void                     setError(
                                             double const* const error,
                                             std::size_t const   size = 1) = 0;
    /** Set pointer to current Jacobi matrix.
     *
     * @param[in] jacobian Derivatives of error with respect to weights.
     * @param[in] columns Number of gradients provided.
     *
     * @note
     * If there are @f$m@f$ errors and @f$n@f$ weights, the Jacobi matrix
     * is a @f$n \times m@f$ matrix stored in column-major order.
     */
    virtual void                     setJacobian(
                                          double const* const jacobian,
                                          std::size_t const   columns = 1) = 0;
    /** Perform single update of state vector.
     */
    virtual void                     update() = 0;
    /** Status report.
     *
     * @param[in] epoch Current epoch.
     *
     * @return Line with current status information.
     */
    virtual std::string              status(std::size_t epoch) const = 0;
    /** Header for status report file.
     *
     * @return Vector with header lines.
     */
    virtual std::vector<std::string> statusHeader() const = 0;
    /** Information about this updater.
     *
     * @return Vector of information lines.
     */
    virtual std::vector<std::string> info() const = 0;
    /** Activate detailed timing.
     *
     * @param[in] prefix Prefix used for stopwatch map entries.
     */
    virtual void                     setupTiming(
                                            std::string const& prefix = "upd");
    /** Start a new timing loop (e.g. epoch).
     */
    virtual void                     resetTimingLoop();
    /** Return timings gathered in stopwatch map.
     *
     * @return Stopwatch map.
     */
    virtual
    std::map<std::string, Stopwatch> getTiming() const;

protected:
    /** Constructor
     *
     * @param[in] sizeState Size of the state vector (number of parameters to
     *                      optimize).
     */
    Updater(std::size_t const sizeState);

    /// Whether detailed timing is enabled.
    bool                             timing;
    /// Internal loop timer reset switch.
    bool                             timingReset;
    /// Number of neural network connections (weights + biases).
    std::size_t                      sizeState;
    /// Prefix for timing stopwatches.
    std::string                      prefix;
    /// Stopwatch map for timing.
    std::map<std::string, Stopwatch> sw;
};

}

#endif
