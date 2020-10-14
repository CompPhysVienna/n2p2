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

#ifndef SYMFNCCOMPANGW_H
#define SYMFNCCOMPANGW_H

#include "SymFncBaseCompAng.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;

/** Angular symmetry function with polynomials (type 29)
 *
 * @f[
   G^{29}_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
              C_{\text{poly}}(\theta_{ijk})
              \mathrm{e}^{-\eta( (r_{ij}-r_s)^2 + (r_{ik}-r_s)^2 ) }
              f_c(r_{ij}) f_c(r_{ik}) 
 * @f]
 * Parameter string:
 * ```
 * <element-central> 89 <element-neighbor1> <element-neighbor2> <rlow> <left> <right> <rcutoff>
 * ```
 * where
 * - `<element-central> .....` element symbol of central atom
 * - `<element-neighbor1> ...` element symbol of neighbor atom 1
 * - `<element-neighbor2> ...` element symbol of neighbor atom 2
 * - `<rlow>.................` lower radial boundary
 * - `<left> ................` left angle boundary 
 * - `<right> ...............` right angle boundary 
 * - `<rcutoff> .............` upper radial boundary
 */
class SymFncCompAngw : public SymFncBaseCompAng
{
public:
    /** Constructor, sets type = 22
     */
    SymFncCompAngw(ElementMap const& elementMap);
    /** Overload == operator.
     */
    virtual bool        operator==(SymFnc const& rhs) const;
    /** Overload < operator.
     */
    virtual bool        operator<(SymFnc const& rhs) const;
    /** Calculate symmetry function for one atom.
     *
     * @param[in,out] atom Atom for which the symmetry function is caluclated.
     * @param[in] derivatives If also symmetry function derivatives will be
     *                        calculated and saved.
     */
    virtual void        calculate(Atom& atom, bool const derivatives) const;
};

}

#endif
