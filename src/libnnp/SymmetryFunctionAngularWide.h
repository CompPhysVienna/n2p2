// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SYMMETRYFUNCTIONANGULARWIDE_H
#define SYMMETRYFUNCTIONANGULARWIDE_H

#include "SymmetryFunction.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymmetryFunctionStatistics;

/** Angular symmetry function (type 9)
 *
 * @f[
   G^9_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
           \left( 1 + \lambda \cos \theta_{ijk} \right)^\zeta
           \mathrm{e}^{-\eta( r_{ij}^2 + r_{ik}^2 ) } f_c(r_{ij}) f_c(r_{ik}) 
 * @f]
 * Parameter string:
 * ```
 * <element-central> 9 <element-neighbor1> <element-neighbor2> <eta> <lambda> <zeta> <rcutoff>
 * ```
 * where
 * - `<element-central> .....` element symbol of central atom
 * - `<element-neighbor1> ...` element symbol of neighbor atom 1
 * - `<element-neighbor2> ...` element symbol of neighbor atom 2
 * - `<eta> .................` @f$\eta@f$ 
 * - `<lambda> ..............` @f$\lambda@f$ 
 * - `<zeta> ................` @f$\zeta@f$ 
 * - `<rcutoff> .............` @f$r_c@f$
 */
class SymmetryFunctionAngularWide : public SymmetryFunction
{
public:
    /** Constructor, sets type = 9
     */
    SymmetryFunctionAngularWide(ElementMap const& elementMap);
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
     * @param[in] parameterString String containing angular symmetry function
     *                            parameters.
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
    /** Get private #e1 member variable.
     */
    std::size_t  getE1() const;
    /** Get private #e2 member variable.
     */
    std::size_t  getE2() const;
    /** Get private #zetaInt member variable.
     */
    int          getZetaInt() const;
    /** Get private #lambda member variable.
     */
    double       getLambda() const;
    /** Get private #eta member variable.
     */
    double       getEta() const;
    /** Get private #zeta member variable.
     */
    double       getZeta() const;
    /** Calculate (partial) symmetry function value for one given distance.
     *
     * @param[in] distance Distance between two atoms.
     * @return @f$\left(e^{-\eta r^2} f_c(r)\right)^2@f$
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
    /// Element index of neighbor atom 1.
    std::size_t e1;
    /// Element index of neighbor atom 2.
    std::size_t e2;
    /// Integer version of @f$\zeta@f$.
    int         zetaInt;
    /// Cosine shift factor.
    double      lambda;
    /// Width @f$\eta@f$ of gaussian.
    double      eta;
    /// Exponent @f$\zeta@f$ of cosine term.
    double      zeta;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool SymmetryFunctionAngularWide::
operator!=(SymmetryFunction const& rhs) const
{
    return !((*this) == rhs);
}

inline bool SymmetryFunctionAngularWide::
operator>(SymmetryFunction const& rhs) const
{
    return rhs < (*this);
}

inline bool SymmetryFunctionAngularWide::
operator<=(SymmetryFunction const& rhs) const
{
    return !((*this) > rhs);
}

inline bool SymmetryFunctionAngularWide::
operator>=(SymmetryFunction const& rhs) const
{
    return !((*this) < rhs);
}

inline bool SymmetryFunctionAngularWide::getUseIntegerPow() const
{
    return useIntegerPow;
}

inline std::size_t SymmetryFunctionAngularWide::getE1() const
{
    return e1;
}

inline std::size_t SymmetryFunctionAngularWide::getE2() const
{
    return e2;
}

inline int SymmetryFunctionAngularWide::getZetaInt() const
{
    return zetaInt;
}

inline double SymmetryFunctionAngularWide::getLambda() const
{
    return lambda;
}

inline double SymmetryFunctionAngularWide::getEta() const
{
    return eta;
}

inline double SymmetryFunctionAngularWide::getZeta() const
{
    return zeta;
}

}

#endif
