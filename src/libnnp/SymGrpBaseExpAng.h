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

#ifndef SYMGRPBASEEXPANG_H
#define SYMGRPBASEEXPANG_H

#include "SymGrpBaseCutoff.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

class ElementMap;
class SymFncBaseExpAng;

class SymGrpBaseExpAng : public SymGrpBaseCutoff
{
public:
    /** Fill #scalingFactors with values from member symmetry functions.
     */
    virtual void                     setScalingFactors();
    /** Give symmetry function group parameters on multiple lines.
     *
     * @return Vector of string containing symmetry function parameters lines.
     */
    virtual std::vector<std::string> parameterLines() const;

protected:
    /** Constructor, sets type.
     *
     * @param[in] type Type of symmetry functions grouped.
     * @param[in] elementMap Element Map used.
     */
    SymGrpBaseExpAng(std::size_t type, ElementMap const& elementMap);
    /** Get symmetry function members.
     *
     * @return Vector of pointers casted to base class.
     */
    virtual std::vector<SymFncBaseExpAng const*> getMembers() const = 0;

    /// Element index of neighbor atom 1 (common feature).
    std::size_t         e1;
    /// Element index of neighbor atom 2 (common feature).
    std::size_t         e2;
    /// Vector indicating whether exponential term needs to be calculated.
    std::vector<bool>   calculateExp;
    /// Vector containing precalculated normalizing factor for each zeta.
    std::vector<double> factorNorm;
    /// Vector containing precalculated normalizing factor for derivatives.
    std::vector<double> factorDeriv;
    /// Vector containing values of all member symmetry functions.
    std::vector<bool>   useIntegerPow;
    /// Vector containing values of all member symmetry functions.
    std::vector<int>    zetaInt;
    /// Vector containing values of all member symmetry functions.
    std::vector<double> eta;
    /// Vector containing values of all member symmetry functions.
    std::vector<double> zeta;
    /// Vector containing values of all member symmetry functions.
    std::vector<double> lambda;
    /// Vector containing values of all member symmetry functions.
    std::vector<double> zetaLambda;
    /// Vector containing values of all member symmetry functions.
    std::vector<double> rs;
};

}

#endif
