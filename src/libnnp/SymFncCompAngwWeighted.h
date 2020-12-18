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

#ifndef SYMFNCCOMPANGWWEIGHTED_H
#define SYMFNCCOMPANGWWEIGHTED_H

#include "SymFncBaseCompAngWeighted.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;

/** Weighted wide angular symmetry function with compact support (type 25)
 *
 * @f[
   G^{25}_i = \sum_{\substack{j,k\neq i \\ j < k}}
              Z_j Z_k
              C(r_{ij}, r_l, r_c)
              C(r_{ik}, r_l, r_c)
              C(\theta_{ijk}, \theta_l, \theta_r),
 * @f]
 * where @f$C(x, x_\text{low}, x_\text{high})@f$ is a function with compact
 * support @f$\left[x_\text{low}, x_\text{high}\right]@f$. @f$Z_j@f$ is defined
 * as the atomic number of the neighbor atom @f$j@f$.
 *
 * Parameter string:
 * ```
 * <element-central> 25 <rlow> <rcutoff> <left> <right> <subtype>
 * ```
 * where
 * - `<element-central> .....` element symbol of central atom
 * - `<rlow> ................` low radius boundary @f$r_{l}@f$
 * - `<rcutoff> .............` high radius boundary @f$r_{c}@f$
 * - `<left> ................` left angle boundary @f$\theta_l@f$
 * - `<right> ...............` right angle boundary @f$\theta_r@f$
 * - `<subtype> .............` compact function specifier
 *
 * See the description of SymFncBaseComp::setCompactFunction() for possible
 * values of the `<subtype>` argument.
 *
 * @note If `<subtype>` specifies an asymmetric version of a polynomial core
 * function the asymmetry only applies to the radial parts, i.e. "p2a" sets
 * a "p2a" radial compact function and a "p2" angular compact function.
 */
class SymFncCompAngwWeighted : public SymFncBaseCompAngWeighted
{
public:
    /** Constructor, sets type = 25
    */
    SymFncCompAngwWeighted(ElementMap const& elementMap);
    /** Overload == operator.
     */
    virtual bool operator==(SymFnc const& rhs) const;
    /** Overload < operator.
     */
    virtual bool operator<(SymFnc const& rhs) const;
    /** Calculate symmetry function for one atom.
     *
     * @param[in,out] atom Atom for which the symmetry function is caluclated.
     * @param[in] derivatives If also symmetry function derivatives will be
     *                        calculated and saved.
     */
    virtual void calculate(Atom& atom, bool const derivatives) const;
};

}

#endif
