// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
// Copyright (C) 2020 Martin P. Bircher
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

#ifndef SYMGRPBASECOMPANG_H
#define SYMGRPBASECOMPANG_H

#include "SymGrpBaseComp.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

class ElementMap;
class SymFncBaseCompAng;

class SymGrpBaseCompAng : public SymGrpBaseComp
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
    SymGrpBaseCompAng(std::size_t type, ElementMap const& elementMap);
    /** Get symmetry function members.
     *
     * @return Vector of pointers casted to base class.
     */
    virtual std::vector<SymFncBaseCompAng const*> getMembers() const = 0;

    /// Element index of neighbor atom 1 (common feature).
    std::size_t                e1;
    /// Element index of neighbor atom 2 (common feature).
    std::size_t                e2;
    /// Vector indicating whether compact function needs to be recalculated.
    std::vector<bool>          calculateComp;
    /// Member rl.
    std::vector<double>        mrl;
    /// Member rc.
    std::vector<double>        mrc;
    /// Member angleLeft.
    std::vector<double>        mal;
    /// Member angleRight.
    std::vector<double>        mar;
#ifndef NNP_NO_SF_CACHE
    /// Member cache indices for actual neighbor element.
    std::vector<std::vector<
    std::vector<std::size_t>>> mci;
#endif
};

}

#endif
