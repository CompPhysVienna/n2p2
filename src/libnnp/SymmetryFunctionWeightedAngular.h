// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SYMMETRYFUNCTIONWEIGHTEDANGULAR_H
#define SYMMETRYFUNCTIONWEIGHTEDANGULAR_H

#include "SymmetryFunction.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymmetryFunctionStatistics;

/** Weighted angular symmetry function (type 13)
 *
 * @f[
   G^{13}_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
              Z_j Z_k \,
              \left( 1 + \lambda \cos \theta_{ijk} \right)^\zeta
              \mathrm{e}^{-\eta \left[
              (r_{ij} - r_s)^2 + (r_{ik} - r_s)^2 + (r_{jk} - r_s)^2 \right] }
              f_c(r_{ij}) f_c(r_{ik}) f_c(r_{jk}) 
 * @f]
 * Parameter string:
 * ```
 * <element-central> 13 <eta> <rshift> <lambda> <zeta> <rcutoff>
 * ```
 * where
 * - `<element-central> .....` element symbol of central atom
 * - `<eta> .................` @f$\eta@f$ 
 * - `<rshift> ..............` @f$r_s@f$ 
 * - `<lambda> ..............` @f$\lambda@f$ 
 * - `<zeta> ................` @f$\zeta@f$ 
 * - `<rcutoff> .............` @f$r_c@f$
 */
class SymmetryFunctionWeightedAngular : public SymmetryFunction
{
public:
    /** Constructor, sets type = 13
     */
    SymmetryFunctionWeightedAngular(ElementMap const& elementMap);
    /** Overload == operator.
     */
    bool         operator==(SymmetryFunction const& rhs) const;
    /** Overload != operator.
     */
    bool         operator!=(SymmetryFunction const& rhs) const;
    /** Overload < operator.
     */
    bool         operator<(SymmetryFunction const& rhs) const;
    /** Overload > operator.
     */
    bool         operator>(SymmetryFunction const& rhs) const;
    /** Overload <= operator.
     */
    bool         operator<=(SymmetryFunction const& rhs) const;
    /** Overload >= operator.
     */
    bool         operator>=(SymmetryFunction const& rhs) const;
    /** Set symmetry function parameters.
     *
     * @param[in] parameterString String containing weighted angular symmetry
     *                            function parameters.
     */
    void         setParameters(std::string const& parameterString);
    /** Change length unit.
     *
     * @param[in] convLength Multiplicative length unit conversion factor.
     */
    void         changeLengthUnit(double convLength);
    /** Get settings file line from currently set parameters.
     *
     * @param[in] convLength Apply length conversion factor if requested.
     *
     * @return Settings file string ("symfunction_short ...").
     */
    std::string  getSettingsLine(double const convLength = 1.0) const;
    /** Calculate symmetry function for one atom.
     *
     * @param[in,out] atom Atom for which the symmetry function is caluclated.
     * @param[in] derivatives If also symmetry function derivatives will be
     *                        calculated and saved.
     * @param[in,out] statistics Gathers statistics and extrapolation warnings.
     */
    void         calculate(Atom&                       atom,
                           bool const                  derivatives,
                           SymmetryFunctionStatistics& statistics) const;
    /** Give symmetry function parameters in one line.
     *
     * @return String containing symmetry function parameter values.
     */
    std::string  parameterLine() const;
    /** Get description with parameter names and values.
     *
     * @return Vector of parameter description strings.
     */
    std::vector<
    std::string> parameterInfo() const;
    /** Get private #useIntegerPow member variable.
     */
    bool         getUseIntegerPow() const;
    /** Get private #zetaInt member variable.
     */
    int          getZetaInt() const;
    /** Get private #eta member variable.
     */
    double       getEta() const;
    /** Get private #rs member variable.
     */
    double       getRs() const;
    /** Get private #lambda member variable.
     */
    double       getLambda() const;
    /** Get private #zeta member variable.
     */
    double       getZeta() const;
    /** Calculate (partial) symmetry function value for one given distance.
     *
     * @param[in] distance Distance between two atoms.
     * @return @f$\left(e^{-\eta (r - r_s)^2} f_c(r)\right)^3@f$
     */
    double       calculateRadialPart(double distance) const;
    /** Calculate (partial) symmetry function value for one given angle.
     *
     * @param[in] angle Angle between triplet of atoms (in radians).
     * @return @f$1@f$
     */
    double       calculateAngularPart(double angle) const;

private:
    /// Whether to use integer version of power function (faster).
    bool        useIntegerPow;
    /// Integer version of @f$\zeta@f$.
    int         zetaInt;
    /// Width @f$\eta@f$ of gaussian.
    double      eta;
    /// Shift @f$r_s@f$ of gaussian.
    double      rs;
    /// Cosine shift factor.
    double      lambda;
    /// Exponent @f$\zeta@f$ of cosine term.
    double      zeta;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool SymmetryFunctionWeightedAngular::
operator!=(SymmetryFunction const& rhs) const
{
    return !((*this) == rhs);
}

inline bool SymmetryFunctionWeightedAngular::
operator>(SymmetryFunction const& rhs) const
{
    return rhs < (*this);
}

inline bool SymmetryFunctionWeightedAngular::
operator<=(SymmetryFunction const& rhs) const
{
    return !((*this) > rhs);
}

inline bool SymmetryFunctionWeightedAngular::
operator>=(SymmetryFunction const& rhs) const
{
    return !((*this) < rhs);
}

inline bool SymmetryFunctionWeightedAngular::getUseIntegerPow() const
{
    return useIntegerPow;
}

inline int SymmetryFunctionWeightedAngular::getZetaInt() const
{
    return zetaInt;
}

inline double SymmetryFunctionWeightedAngular::getEta() const
{
    return eta;
}

inline double SymmetryFunctionWeightedAngular::getRs() const
{
    return rs;
}

inline double SymmetryFunctionWeightedAngular::getLambda() const
{
    return lambda;
}

inline double SymmetryFunctionWeightedAngular::getZeta() const
{
    return zeta;
}

}

#endif
