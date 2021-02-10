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

#ifndef SYMGRPEXPANGNWEIGHTED_H
#define SYMGRPEXPANGNWEIGHTED_H

#include "SymGrpBaseCutoff.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymFnc;
class SymFncExpAngnWeighted;

/** Weighted angular symmetry function group (type 13)
 *
 * @f[
   G^{13}_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
              Z_j Z_k \,
              \left( 1 + \lambda \cos \theta_{ijk} \right)^\zeta
              \mathrm{e}^{-\eta \left[
              (r_{ij} - r_s)^2 + (r_{ik} - r_s)^2 + (r_{jk} - r_s)^2 \right] }
              f_c(r_{ij}) f_c(r_{ik}) f_c(r_{jk}) 
 * @f]
 * Common features:
 * - element of central atom
 * - cutoff type
 * - @f$r_c@f$
 * - @f$\alpha@f$
 */
class SymGrpExpAngnWeighted : public SymGrpBaseCutoff
{
public:
    /** Constructor, sets type = 13
     */
    SymGrpExpAngnWeighted(ElementMap const& elementMap);
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
    std::vector<SymFncExpAngnWeighted const*> members;
    /// Vector indicating whether exponential term needs to be calculated.
    std::vector<bool>                         calculateExp;
    /// Vector containing precalculated normalizing factor for each zeta.
    std::vector<double>                       factorNorm;
    /// Vector containing precalculated normalizing factor for derivatives.
    std::vector<double>                       factorDeriv;
    /// Vector containing values of all member symmetry functions.
    std::vector<bool>                         useIntegerPow;
    /// Vector containing values of all member symmetry functions.
    std::vector<int>                          zetaInt;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                       eta;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                       rs;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                       lambda;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                       zeta;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                       zetaLambda;
};

}

#endif
