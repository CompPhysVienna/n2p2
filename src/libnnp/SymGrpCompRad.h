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

#ifndef SYMGRPCOMPRAD_H
#define SYMGRPCOMPRAD_H

#include "SymGrpBaseComp.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymFnc;
class SymFncCompRad;

/** Radial symmetry function with compact support (type 20)
 *
 * @f[
   G^{20}_i = \sum_{\substack{j \neq i}}
              C(r_{ij}, r_l, r_c),
 * @f]
 * where @f$C(x, x_\text{low}, x_\text{high})@f$ is a function with compact
 * support @f$\left[x_\text{low}, x_\text{high}\right]@f$.
 *
 * Common features:
 * - element of central atom
 * - element of neighbor atom
 */
class SymGrpCompRad : public SymGrpBaseComp
{
public:
    /** Constructor, sets type = 20
     */
    SymGrpCompRad(ElementMap const& elementMap);
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
    /** Fill #scalingFactors with values from member symmetry functions.
     */
    virtual void setScalingFactors();
    /** Calculate all symmetry functions of this group for one atom.
     *
     * @param[in,out] atom Atom for which symmetry functions are caluclated.
     * @param[in] derivatives If also symmetry function derivatives will be
     *                        calculated and saved.
     */
    virtual void calculate(Atom& atom, bool const derivatives) const;
    /** Give symmetry function group parameters on multiple lines.
     *
     * @return Vector of string containing symmetry function parameters lines.
     */
    virtual std::vector<std::string>
                 parameterLines() const;

private:
    /// Element index of neighbor atom (common feature).
    std::size_t                           e1;
    /// Vector of all group member pointers.
    std::vector<SymFncCompRad const*>     members;
    /// Member rl.
    std::vector<double>                   mrl;
    /// Member rc.
    std::vector<double>                   mrc;
#ifndef NNP_NO_SF_CACHE
    /// Member cache indices for actual neighbor element.
    std::vector<std::vector<std::size_t>> mci;
#endif
};

}

#endif
