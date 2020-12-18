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

#ifndef SYMGRPBASECOMP_H
#define SYMGRPBASECOMP_H

#include "SymGrp.h"
#include "CutoffFunction.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

class SymGrpBaseComp : public SymGrp
{
public:
    /// Getter for #rmin.
    double getRmin() const;
    /// Getter for #rmax.
    double getRmax() const;

protected:
    /// Minimum radius within group.
    double rmin;
    /// Maximum radius within group.
    double rmax;

    /** Constructor, sets type.
     *
     * @param[in] type Type of symmetry functions grouped.
     * @param[in] elementMap Element Map used.
     */
    SymGrpBaseComp(std::size_t type, ElementMap const& elementMap);
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline double SymGrpBaseComp::getRmin() const
{
    return rmin;
}

inline double SymGrpBaseComp::getRmax() const
{
    return rmax;
}

}

#endif
