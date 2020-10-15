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

#ifndef SYMFNCBASECUTOFF_H
#define SYMFNCBASECUTOFF_H

#include "SymFnc.h"
#include "CutoffFunction.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/// Intermediate class for SFs based on cutoff functions.
class SymFncBaseCutoff : public SymFnc
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
    /** Get private #subtype member variable.
     */
    std::string              getSubtype() const;
    /** Get private #cutoffType member variable.
     */
    CutoffFunction::
    CutoffType               getCutoffType() const;

protected:
    /// Cutoff parameter @f$\alpha@f$.
    double                     cutoffAlpha;
    /// Subtype string (specifies cutoff type).
    std::string                subtype;
    /// Cutoff function used by this symmetry function.
    CutoffFunction             fc;
    /// Cutoff type used by this symmetry function.
    CutoffFunction::CutoffType cutoffType;

    /** Constructor, initializes #type.
     */
    SymFncBaseCutoff(std::size_t type, ElementMap const&);
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline double SymFncBaseCutoff::getCutoffAlpha() const { return cutoffAlpha; }
inline std::string SymFncBaseCutoff::getSubtype() const { return subtype; } 

inline CutoffFunction::CutoffType SymFncBaseCutoff::getCutoffType() const
{
    return cutoffType;
}

}

#endif
