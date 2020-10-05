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

#ifndef SYMFUNCCUTOFFBASED_H
#define SYMFUNCCUTOFFBASED_H

#include "SymFnc.h"
#include "CutoffFunction.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/** Symmetry function class based on cutoff functions.
 *
 * Actual symmetry functions which make use of a dedicated cutoff function
 * derive from this class.
 */
class SymFncCutoffBased : public SymFnc
{
public:
    /** Get description with parameter names and values.
     *
     * @return Vector of parameter description strings.
     */
    virtual
    std::vector<std::string> parameterInfo() const;
    /** Set cutoff function type and parameter.
     *
     * @param[in] cutoffType Desired cutoff function for this symmetry
     *                       function.
     * @param[in] cutoffAlpha Cutoff function parameter @f$\alpha@f$.
     */
    void                     setCutoffFunction(CutoffFunction::
                                          CutoffType cutoffType,
                                          double     cutoffAlpha);
    /** Get private #cutoffAlpha member variable.
     */
    double                   getCutoffAlpha() const;
    /** Get private #cutoffType member variable.
     */
    CutoffFunction::
    CutoffType               getCutoffType() const;

protected:
    /// Cutoff parameter @f$\alpha@f$.
    double                     cutoffAlpha;
    /// Cutoff function used by this symmetry function.
    CutoffFunction             fc;
    /// Cutoff type used by this symmetry function.
    CutoffFunction::CutoffType cutoffType;

    /** Constructor, initializes #type.
     */
    SymFncCutoffBased(std::size_t type, ElementMap const&);
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline double SymFncCutoffBased::getCutoffAlpha() const
{
    return cutoffAlpha;
}

inline CutoffFunction::CutoffType SymFncCutoffBased::getCutoffType() const
{
    return cutoffType;
}

}

#endif
