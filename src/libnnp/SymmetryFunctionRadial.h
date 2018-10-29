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

#ifndef SYMMETRYFUNCTIONRADIAL_H
#define SYMMETRYFUNCTIONRADIAL_H

#include "SymmetryFunction.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymmetryFunctionStatistics;

/** Radial symmetry function (type 2)
 *
 * @f[
 * G^2_i = \sum_{j \neq i} \mathrm{e}^{-\eta(r_{ij} - r_\mathrm{s})^2}
 *         f_c(r_{ij}) 
 * @f]
 * Parameter string:
 * ```
 * <element-central> 2 <element-neighbor> <eta> <rshift> <rcutoff>
 * ```
 * where
 * - `<element-central> ....` element symbol of central atom
 * - `<element-neighbor> ...` element symbol of neighbor atom
 * - `<eta> ................` @f$\eta@f$ 
 * - `<rshift> .............` @f$r_\mathrm{s}@f$
 * - `<rcutoff> ............` @f$r_c@f$
 */
class SymmetryFunctionRadial : public SymmetryFunction
{
public:
    /** Constructor, sets type = 2
     */
    SymmetryFunctionRadial(ElementMap const& elementMap);
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
    /** Get private #eta member variable.
     */
    double       getEta() const;
    /** Get private #rs member variable.
     */
    double       getRs() const;
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

private:
    /// Element index of neighbor atom.
    std::size_t e1;
    /// Width @f$\eta@f$ of gaussian.
    double      eta;
    /// Shift @f$r_s@f$ of gaussian.
    double      rs;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline
bool SymmetryFunctionRadial::operator!=(SymmetryFunction const& rhs) const
{
    return !((*this) == rhs);
}

inline
bool SymmetryFunctionRadial::operator>(SymmetryFunction const& rhs) const
{
    return rhs < (*this);
}

inline
bool SymmetryFunctionRadial::operator<=(SymmetryFunction const& rhs) const
{
    return !((*this) > rhs);
}

inline
bool SymmetryFunctionRadial::operator>=(SymmetryFunction const& rhs) const
{
    return !((*this) < rhs);
}

inline std::size_t SymmetryFunctionRadial::getE1() const
{
    return e1;
}

inline double SymmetryFunctionRadial::getEta() const
{
    return eta;
}

inline double SymmetryFunctionRadial::getRs() const
{
    return rs;
}

}

#endif
