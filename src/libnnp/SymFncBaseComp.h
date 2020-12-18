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

#ifndef SYMFNCBASECOMP_H
#define SYMFNCBASECOMP_H

#include "SymFnc.h"
#include "CompactFunction.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/// Symmetry function base class for SFs with compact support.
class SymFncBaseComp : public SymFnc
{
public:
    /** Get description with parameter names and values.
     *
     * @return Vector of parameter description strings.
     */
    virtual
    std::vector<std::string> parameterInfo() const;
    /** Set radial compact function.
     *
     * @param[in] subtype Core function specification.
     *
     * The subtype argument determines which CoreFunction::Type is used to
     * generate the CompactFunction used by symmetry functions. The following
     * strings are accepted (without the apostrophes):
     * - "e": CoreFunction::Type::EXP
     * - "pN", where "N" is one of "1,2,3,4", corresponding to the
     *    polynomial CoreFunction::Type POLYN, respectively. "pN" can
     *    be followed by the letter "a", which will enable the asymmetric
     *    version of the compact function, i.e. @f$2x - x^2@f$ is used as
     *    argument. Examples: "p2", "p3a", "p4".
     */
    void                     setCompactFunction(std::string subtype);
    /** Get private #subtype member variable.
     */
    std::string              getSubtype() const;
    /** Get private #rl member variable.
     */
    double                   getRl() const;

protected:
    /// If asymmetric version of polynomials should be used.
    bool            asymmetric;
    /// Lower bound of compact function, @f$r_{l}@f$.
    double          rl;
    /// Subtype string (specifies e.g. polynom type).
    std::string     subtype;
    /// Compact function for radial part.
    CompactFunction cr;

    /** Constructor, initializes #type.
     */
    SymFncBaseComp(std::size_t type, ElementMap const&);
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline std::string SymFncBaseComp::getSubtype() const { return subtype; }
inline double SymFncBaseComp::getRl() const { return rl; }

}

#endif
