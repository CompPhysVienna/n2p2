// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SYMMETRYFUNCTIONGROUPWEIGHTEDANGULAR_H
#define SYMMETRYFUNCTIONGROUPWEIGHTEDANGULAR_H

#include "SymmetryFunctionGroup.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymmetryFunction;
class SymmetryFunctionWeightedAngular;

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
class SymmetryFunctionGroupWeightedAngular : public SymmetryFunctionGroup
{
public:
    /** Constructor, sets type = 13
     */
    SymmetryFunctionGroupWeightedAngular(ElementMap const& elementMap);
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
     * @param[in,out] statistics Gathers statistics and extrapolation warnings.
     */
    void calculate(Atom&                       atom,
                   bool const                  derivatives,
                   SymmetryFunctionStatistics& statistics) const;
    /** Give symmetry function group parameters on multiple lines.
     *
     * @return Vector of string containing symmetry function parameters lines.
     */
    std::vector<std::string>
         parameterLines() const;

private:
    /// Vector of all group member pointers.
    std::vector<SymmetryFunctionWeightedAngular const*> members;
    /// Vector indicating whether exponential term needs to be calculated.
    std::vector<bool>                                   calculateExp;
    /// Vector containing precalculated normalizing factor for each zeta.
    std::vector<double>                                 factorNorm;
    /// Vector containing precalculated normalizing factor for derivatives.
    std::vector<double>                                 factorDeriv;
    /// Vector containing values of all member symmetry functions.
    std::vector<bool>                                   useIntegerPow;
    /// Vector containing values of all member symmetry functions.
    std::vector<size_t>                                 memberIndex;
    /// Vector containing values of all member symmetry functions.
    std::vector<int>                                    zetaInt;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                                 eta;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                                 rs;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                                 lambda;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                                 zeta;
    /// Vector containing values of all member symmetry functions.
    std::vector<double>                                 zetaLambda;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool SymmetryFunctionGroupWeightedAngular::
operator!=(SymmetryFunctionGroup const& rhs) const
{
    return !((*this) == rhs);
}

inline bool SymmetryFunctionGroupWeightedAngular::
operator>(SymmetryFunctionGroup const& rhs) const
{
    return rhs < (*this);
}

inline bool SymmetryFunctionGroupWeightedAngular::
operator<=(SymmetryFunctionGroup const& rhs) const
{
    return !((*this) > rhs);
}

inline bool SymmetryFunctionGroupWeightedAngular::
operator>=(SymmetryFunctionGroup const& rhs) const
{
    return !((*this) < rhs);
}

}

#endif
