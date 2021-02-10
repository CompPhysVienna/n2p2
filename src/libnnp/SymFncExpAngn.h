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

#ifndef SYMFNCEXPANGN_H
#define SYMFNCEXPANGN_H

#include "SymFncBaseExpAng.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;

/** Angular symmetry function (type 3)
 *
 * @f[
   G^3_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
           \left( 1 + \lambda \cos \theta_{ijk} \right)^\zeta
           \mathrm{e}^{-\eta( (r_{ij}-r_s)^2 + (r_{ik}-r_s)^2
           + (r_{jk}-r_s)^2 ) }
           f_c(r_{ij}) f_c(r_{ik}) f_c(r_{jk}) 
 * @f]
 * Parameter string:
 * ```
 * <element-central> 3 <element-neighbor1> <element-neighbor2> <eta> <lambda> <zeta> <rcutoff> <<rshift>>
 * ```
 * where
 * - `<element-central> .....` element symbol of central atom
 * - `<element-neighbor1> ...` element symbol of neighbor atom 1
 * - `<element-neighbor2> ...` element symbol of neighbor atom 2
 * - `<eta> .................` @f$\eta@f$ 
 * - `<lambda> ..............` @f$\lambda@f$ 
 * - `<zeta> ................` @f$\zeta@f$ 
 * - `<rcutoff> .............` @f$r_c@f$
 * - `<<rshift>> ............` @f$r_s@f$ (optional, default @f$r_s = 0@f$)
 */
class SymFncExpAngn : public SymFncBaseExpAng
{
public:
    /** Constructor, sets type = 3
     */
    SymFncExpAngn(ElementMap const& elementMap);
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
