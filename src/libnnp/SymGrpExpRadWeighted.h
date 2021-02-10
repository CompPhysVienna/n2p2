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

#ifndef SYMGRPEXPRADWEIGHTED_H
#define SYMGRPEXPRADWEIGHTED_H

#include "SymGrpBaseCutoff.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymFnc;
class SymFncExpRadWeighted;

/** Weighted radial symmetry function group (type 12)
 *
 * @f[
 * G^{12}_i = \sum_{j \neq i} Z_j \mathrm{e}^{-\eta(r_{ij} - r_\mathrm{s})^2}
 *            f_c(r_{ij}) 
 * @f]
 * Common features:
 * - element of central atom
 * - cutoff type
 * - @f$r_c@f$
 * - @f$\alpha@f$
 */
class SymGrpExpRadWeighted : public SymGrpBaseCutoff
{
public:
    /** Constructor, sets type = 12
     */
    SymGrpExpRadWeighted(ElementMap const& elementMap);
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
    /// Vector of all group member pointers.
    std::vector<SymFncExpRadWeighted const*> members;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                      eta;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                      rs;
};

}

#endif
