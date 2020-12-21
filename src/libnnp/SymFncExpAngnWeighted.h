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

#ifndef SYMFNCEXPANGNWEIGHTED_H
#define SYMFNCEXPANGNWEIGHTED_H

#include "SymFncBaseCutoff.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;

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
class SymFncExpAngnWeighted : public SymFncBaseCutoff
{
public:
    /** Constructor, sets type = 13
     */
    SymFncExpAngnWeighted(ElementMap const& elementMap);
    /** Overload == operator.
     */
    virtual bool        operator==(SymFnc const& rhs) const;
    /** Overload < operator.
     */
    virtual bool        operator<(SymFnc const& rhs) const;
    /** Set symmetry function parameters.
     *
     * @param[in] parameterString String containing weighted angular symmetry
     *                            function parameters.
     */
    virtual void        setParameters(std::string const& parameterString);
    /** Change length unit.
     *
     * @param[in] convLength Multiplicative length unit conversion factor.
     */
    virtual void        changeLengthUnit(double convLength);
    /** Get settings file line from currently set parameters.
     *
     * @return Settings file string ("symfunction_short ...").
     */
    virtual std::string getSettingsLine() const;
    /** Calculate symmetry function for one atom.
     *
     * @param[in,out] atom Atom for which the symmetry function is caluclated.
     * @param[in] derivatives If also symmetry function derivatives will be
     *                        calculated and saved.
     */
    virtual void        calculate(Atom& atom, bool const derivatives) const;
    /** Give symmetry function parameters in one line.
     *
     * @return String containing symmetry function parameter values.
     */
    virtual std::string parameterLine() const;
    /** Get description with parameter names and values.
     *
     * @return Vector of parameter description strings.
     */
    virtual std::vector<
    std::string>        parameterInfo() const;
    /** Get private #useIntegerPow member variable.
     */
    bool                getUseIntegerPow() const;
    /** Get private #zetaInt member variable.
     */
    int                 getZetaInt() const;
    /** Get private #eta member variable.
     */
    double              getEta() const;
    /** Get private #rs member variable.
     */
    double              getRs() const;
    /** Get private #lambda member variable.
     */
    double              getLambda() const;
    /** Get private #zeta member variable.
     */
    double              getZeta() const;
    /** Calculate (partial) symmetry function value for one given distance.
     *
     * @param[in] distance Distance between two atoms.
     * @return @f$\left(e^{-\eta (r - r_s)^2} f_c(r)\right)^3@f$
     */
    virtual double      calculateRadialPart(double distance) const;
    /** Calculate (partial) symmetry function value for one given angle.
     *
     * @param[in] angle Angle between triplet of atoms (in radians).
     * @return @f$1@f$
     */
    virtual double      calculateAngularPart(double angle) const;
    /** Check whether symmetry function is relevant for given element.
     *
     * @param[in] index Index of given element.
     * @return True if symmetry function is sensitive to given element, false
     *         otherwise.
     */
    virtual bool        checkRelevantElement(std::size_t index) const;
#ifndef NNP_NO_SF_CACHE
    /** Get unique cache identifiers.
     *
     * @return Vector of string identifying the type of cache this symmetry
     *         function requires.
     */
    virtual std::vector<
    std::string>        getCacheIdentifiers() const;
#endif

private:
    /// Whether to use integer version of power function (faster).
    bool   useIntegerPow;
    /// Integer version of @f$\zeta@f$.
    int    zetaInt;
    /// Width @f$\eta@f$ of gaussian.
    double eta;
    /// Shift @f$r_s@f$ of gaussian.
    double rs;
    /// Cosine shift factor.
    double lambda;
    /// Exponent @f$\zeta@f$ of cosine term.
    double zeta;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool SymFncExpAngnWeighted::getUseIntegerPow() const
{
    return useIntegerPow;
}

inline int SymFncExpAngnWeighted::getZetaInt() const { return zetaInt; }
inline double SymFncExpAngnWeighted::getEta() const { return eta; }
inline double SymFncExpAngnWeighted::getRs() const { return rs; }
inline double SymFncExpAngnWeighted::getLambda() const { return lambda; }
inline double SymFncExpAngnWeighted::getZeta() const { return zeta; }

}

#endif
