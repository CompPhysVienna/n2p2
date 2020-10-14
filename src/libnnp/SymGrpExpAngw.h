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

#ifndef SYMGRPEXPANGW_H
#define SYMGRPEXPANGW_H

#include "SymGrpBaseExpAng.h"
#include "SymFncExpAngw.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymFnc;

/** Angular symmetry function group (type 3)
 *
 * @f[
   G^9_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
           \left( 1 + \lambda \cos \theta_{ijk} \right)^\zeta
           \mathrm{e}^{-\eta( (r_{ij}-r_s)^2 + (r_{ik}-r_s)^2 ) }
           f_c(r_{ij}) f_c(r_{ik}) 
 * @f]
 * Common features:
 * - element of central atom
 * - element of neighbor atom 1
 * - element of neighbor atom 2
 * - cutoff type
 * - @f$r_c@f$
 * - @f$\alpha@f$
 */
class SymGrpExpAngw : public SymGrpBaseExpAng
{
public:
    /** Constructor, sets type = 3
     */
    SymGrpExpAngw(ElementMap const& elementMap);
    /** Overload == operator.
     */
    virtual bool operator==(SymGrp const& rhs) const;
    /** Overload < operator.
     */
    virtual bool operator<(SymGrp const& rhs) const;
    /** Potentially add a member to group.
     *
     * @param[in] symmetryFunction Candidate symmetry function.
     * @return If addition was successful.
     *
     * If symmetry function is compatible with common feature list its pointer
     * will be added to #members.
     */
    virtual bool addMember(SymFnc const* const symmetryFunction);
    /** Sort member symmetry functions.
     *
     * Also allocate and precalculate additional stuff.
     */
    virtual void sortMembers();
    /** Calculate all symmetry functions of this group for one atom.
     *
     * @param[in,out] atom Atom for which symmetry functions are caluclated.
     * @param[in] derivatives If also symmetry function derivatives will be
     *                        calculated and saved.
     */
    virtual void calculate(Atom& atom, bool const derivatives) const;

private:
    /** Get symmetry function members.
     *
     * @return Vector of pointers casted to base class.
     */
    virtual std::vector<SymFncBaseExpAng const*> getMembers() const;

    /// Vector of all group member pointers.
    std::vector<SymFncExpAngw const*> members;
};

inline std::vector<SymFncBaseExpAng const*> SymGrpExpAngw::getMembers() const
{
    std::vector<SymFncBaseExpAng const*> cast;

    for (auto p : members)
    {
        cast.push_back(dynamic_cast<SymFncBaseExpAng const*>(p));
    }

    return cast;
}

}

#endif
