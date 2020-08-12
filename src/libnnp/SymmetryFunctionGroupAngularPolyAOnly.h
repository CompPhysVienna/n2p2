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

#ifndef SYMMETRYFUNCTIONGROUPANGULARPOLYAONLY_H
#define SYMMETRYFUNCTIONGROUPANGULARPOLYAONLY_H

#include "CompactFunction.h"
#include "SymmetryFunctionGroup.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymmetryFunction;
class SymmetryFunctionAngularPolyAOnly;

//TODO /** Angular symmetry function group (type 3)
//TODO  *
//TODO  * @f[
//TODO    G^9_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
//TODO            \left( 1 + \lambda \cos \theta_{ijk} \right)^\zeta
//TODO            \mathrm{e}^{-\eta( (r_{ij}-r_s)^2 + (r_{ik}-r_s)^2 ) }
//TODO            f_c(r_{ij}) f_c(r_{ik}) 
//TODO  * @f]
//TODO  * Common features:
//TODO  * - element of central atom
//TODO  * - element of neighbor atom 1
//TODO  * - element of neighbor atom 2
//TODO  * - cutoff type
//TODO  * - @f$r_c@f$
//TODO  * - @f$\alpha@f$
//TODO  */
class SymmetryFunctionGroupAngularPolyAOnly : public SymmetryFunctionGroup
{
public:
    /** Constructor, sets type = 89
     */
    SymmetryFunctionGroupAngularPolyAOnly(ElementMap const& elementMap);
    /** Overload == operator.
     */
    bool operator==(SymmetryFunctionGroup const& rhs) const;
    /** Overload != operator.
     */
    bool operator!=(SymmetryFunctionGroup const& rhs) const;
    /** Overload < operator.
     */
    bool operator<(SymmetryFunctionGroup const& rhs) const;
    /** Overload > operator.
     */
    bool operator>(SymmetryFunctionGroup const& rhs) const;
    /** Overload <= operator.
     */
    bool operator<=(SymmetryFunctionGroup const& rhs) const;
    /** Overload >= operator.
     */
    bool operator>=(SymmetryFunctionGroup const& rhs) const;
    /** Potentially add a member to group.
     *
     * @param[in] symmetryFunction Candidate symmetry function.
     * @return If addition was successful.
     *
     * If symmetry function is compatible with common feature list its pointer
     * will be added to #members.
     */
    bool addMember(SymmetryFunction const* const symmetryFunction);
    /** Sort member symmetry functions.
     *
     * Also allocate and precalculate additional stuff.
     */
    void sortMembers();
    /** Fill #scalingFactors with values from member symmetry functions.
     */
    void setScalingFactors();
    /** Calculate all symmetry functions of this group for one atom.
     *
     * @param[in,out] atom Atom for which symmetry functions are caluclated.
     * @param[in] derivatives If also symmetry function derivatives will be
     *                        calculated and saved.
     */
    void calculate(Atom& atom, bool const derivatives) const;
    /** Give symmetry function group parameters on multiple lines.
     *
     * @return Vector of string containing symmetry function parameters lines.
     */
    std::vector<std::string>
         parameterLines() const;

private:
    /// Element index of neighbor atom 1 (common feature).
    std::size_t                                     e1;
    /// Element index of neighbor atom 2 (common feature).
    std::size_t                                     e2;
    /// Vector of all group member pointers.
    std::vector<SymmetryFunctionAngularPolyAOnly const*> members;
    /// Smallest cutoff value within group.
    double                                          rl; 
    /// Largest cutoff value within group.
    double                                          rc;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool SymmetryFunctionGroupAngularPolyAOnly::
operator!=(SymmetryFunctionGroup const& rhs) const
{
    return !((*this) == rhs);
}

inline bool SymmetryFunctionGroupAngularPolyAOnly::
operator>(SymmetryFunctionGroup const& rhs) const
{
    return rhs < (*this);
}

inline bool SymmetryFunctionGroupAngularPolyAOnly::
operator<=(SymmetryFunctionGroup const& rhs) const
{
    return !((*this) > rhs);
}

inline bool SymmetryFunctionGroupAngularPolyAOnly::
operator>=(SymmetryFunctionGroup const& rhs) const
{
    return !((*this) < rhs);
}

}

#endif
