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

#ifndef SYMGRPBASECUTOFF_H
#define SYMGRPBASECUTOFF_H

#include "SymGrp.h"
#include "CutoffFunction.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

class SymGrpBaseCutoff : public SymGrp
{
public:
    /// Get private #rc member variable.
    double getRc() const;

protected:
    /// Cutoff radius @f$r_c@f$ (common feature).
    double                     rc;
    /// Cutoff function parameter @f$\alpha@f$ (common feature).
    double                     cutoffAlpha;
    /// Subtype string (specifies cutoff type) (common feature).
    std::string                subtype;
    /// Cutoff function used by this symmetry function group.
    CutoffFunction             fc;
    /// Cutoff type used by this symmetry function group (common feature).
    CutoffFunction::CutoffType cutoffType;

    /** Constructor, sets type.
     *
     * @param[in] type Type of symmetry functions grouped.
     * @param[in] elementMap Element Map used.
     */
    SymGrpBaseCutoff(std::size_t type, ElementMap const& elementMap);
};

}

#endif
