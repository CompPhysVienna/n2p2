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

#ifndef SYMFNCCOMPRADWEIGHTED_H
#define SYMFNCCOMPRADWEIGHTED_H

#include "SymFncBaseComp.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;

/** Weighted radial symmetry function with compact support (type 23)
 *
 * @f[
   G^{23}_i = \sum_{\substack{j \neq i}}
              Z_j
              C(r_{ij}, r_l, r_c),
 * @f]
 * where @f$C(x, x_\text{low}, x_\text{high})@f$ is a function with compact
 * support @f$\left[x_\text{low}, x_\text{high}\right]@f$. @f$Z_j@f$ is defined
 * as the atomic number of the neighbor atom @f$j@f$.
 *
 * Parameter string:
 * ```
 * <element-central> 23 <rlow> <rcutoff> <subtype>
 * ```
 * where
 * - `<element-central> ....` element symbol of central atom
 * - `<rlow> ................` low radius boundary @f$r_{l}@f$
 * - `<rcutoff> .............` high radius boundary @f$r_{c}@f$
 * - `<subtype> ............` compact function specifier
 *
 * See the description of SymFncBaseComp::setCompactFunction() for possible
 * values of the `<subtype>` argument.
 */
class SymFncCompRadWeighted : public SymFncBaseComp
{
public:
    /** Constructor, sets type = 23
     */
    SymFncCompRadWeighted(ElementMap const& elementMap);
    /** Overload == operator.
     */
    virtual bool        operator==(SymFnc const& rhs) const;
    /** Overload < operator.
     */
    virtual bool        operator<(SymFnc const& rhs) const;
    /** Set symmetry function parameters.
     *
     * @param[in] parameterString String containing weighted radial symmetry
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
    // Access core fct. for groups
    void                getCompactOnly(double const x,
                                       double&      fx,
                                       double&      dfx) const;
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
    /** Calculate (partial) symmetry function value for one given distance.
     *
     * @param[in] distance Distance between two atoms.
     * @return @f$ e^{-\eta (r - r_s)^2} f_c(r)@f$
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
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline void SymFncCompRadWeighted::getCompactOnly(double const x,
                                                  double&      fx,
                                                  double&      dfx) const
{
    cr.fdf(x, fx, dfx);
    return;
}

}

#endif
