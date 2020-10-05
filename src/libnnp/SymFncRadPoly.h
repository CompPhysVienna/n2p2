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

#ifndef SYMFNCRADPOLY_H
#define SYMFNCRADPOLY_H

#include "SymFnc.h"
#include "CompactFunction.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymFncStatistics;

/** Radial symmetry function (type 28)
 *
 * @f[
 * G^2_i = \text{Polynomial goes here}
 * @f]
 * Parameter string:
 * ```
 * <element-central> 28 <element-neighbor> <rl> <rc>
 * ```
 * where
 * - `<element-central> ....` element symbol of central atom
 * - `<element-neighbor> ...` element symbol of neighbor atom
 * - `<rl>..................` @f$r_{l}@f$
 * - `<rc>..................` @f$r_{c}@f$
 */
class SymFncRadPoly : public SymFnc
{
public:
    /** Constructor, sets type = 2
     */
    SymFncRadPoly(ElementMap const& elementMap);
    /** Overload == operator.
     */
    bool         operator==(SymFnc const& rhs) const;
    /** Overload != operator.
     */
    bool         operator!=(SymFnc const& rhs) const;
    /** Overload < operator.
     */
    bool         operator<(SymFnc const& rhs) const;
    /** Overload > operator.
     */
    bool         operator>(SymFnc const& rhs) const;
    /** Overload <= operator.
     */
    bool         operator<=(SymFnc const& rhs) const;
    /** Overload >= operator.
     */
    bool         operator>=(SymFnc const& rhs) const;
    /** Set symmetry function parameters.
     *
     * @param[in] parameterString String containing radial symmetry function
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
     * @return Settings file string ("symfunction_short ...").
     */
    std::string  getSettingsLine() const;
    /** Calculate symmetry function for one atom.
     *
     * @param[in,out] atom Atom for which the symmetry function is caluclated.
     * @param[in] derivatives If also symmetry function derivatives will be
     *                        calculated and saved.
     */
    void         calculate(Atom& atom, bool const derivatives) const;
    // Access core fct. for groups
    bool         getCompactOnly(double x, double& fx, double& dfx) const;
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
    /** Get private #e1 member variable.
     */
    std::size_t  getE1() const;
    /** Get private #rl member variable.
     */
    double       getRl() const;
    /** Calculate (partial) symmetry function value for one given distance.
     *
     * @param[in] distance Distance between two atoms.
     * @return @f$ e^{-\eta (r - r_s)^2} f_c(r)@f$
     */
    double       calculateRadialPart(double distance) const;
    /** Calculate (partial) symmetry function value for one given angle.
     *
     * @param[in] angle Angle between triplet of atoms (in radians).
     * @return @f$1@f$
     */
    double       calculateAngularPart(double angle) const;
    /** Check whether symmetry function is relevant for given element.
     *
     * @param[in] index Index of given element.
     * @return True if symmetry function is sensitive to given element, false
     *         otherwise.
     */
    bool         checkRelevantElement(std::size_t index) const;

private:
    /// Element index of neighbor atom.
    std::size_t e1;
    /// Lower bound of polynomial, @f$r_{l}@f$.
    double      rl;
    /// Compact function member.
    CompactFunction c;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool SymFncRadPoly::operator!=(SymFnc const& rhs) const
{
    return !((*this) == rhs);
}

inline bool SymFncRadPoly::operator>(SymFnc const& rhs) const
{
    return rhs < (*this);
}

inline bool SymFncRadPoly::operator<=(SymFnc const& rhs) const
{
    return !((*this) > rhs);
}

inline bool SymFncRadPoly::operator>=(SymFnc const& rhs) const
{
    return !((*this) < rhs);
}

inline std::size_t SymFncRadPoly::getE1() const
{
    return e1;
}

inline double SymFncRadPoly::getRl() const
{
    return rl;
}

}

#endif
