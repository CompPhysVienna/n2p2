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

#include <string>
#include <vector>

namespace nnp
{

/// Base class for different weight update methods.
class Updater
{
public:
    /** Constructor
     */
    virtual ~Updater() {};
    /** Set pointer to current state.
     *
     * @param[in,out] state Pointer to state (weights vector).
     */
    virtual void                     setState(double* state) = 0;
    /** Set pointer to current error.
     *
     * @param[in] error Pointer to error (difference between reference and
     *                  neural network potential output).
     */
    virtual void                     setError(double const* const error) = 0;
    /** Set pointer to derivative matrix.
     *
     * @param[in] derivatives Derivatives of error with respect to weights.
     */
    virtual void                     setDerivativeMatrix(
                                          double const* const derivatives) = 0;
    /** Perform update of state vector.
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

protected:
    /** Constructor.
     */
    Updater();
};

}

#endif
