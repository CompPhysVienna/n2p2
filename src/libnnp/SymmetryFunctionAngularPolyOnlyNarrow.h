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

#ifndef SYMMETRYFUNCTIONANGULARPOLYONLYNARROW_H
#define SYMMETRYFUNCTIONANGULARPOLYONLYNARROW_H

#include "SymmetryFunction.h"
#include "CompactFunction.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymmetryFunctionStatistics;

/** Angular symmetry function with polynomials (type 29)
 *
 * @f[
   G^{99}_i = 2^{1-\zeta} \sum_{\substack{j,k\neq i \\ j < k}}
              C_{\text{poly}}(\theta_{ijk})
              \mathrm{e}^{-\eta( (r_{ij}-r_s)^2 + (r_{ik}-r_s)^2 ) }
              f_c(r_{ij}) f_c(r_{ik}) 
 * @f]
 * Parameter string:
 * ```
 * <element-central> 99 <element-neighbor1> <element-neighbor2> <rlow> <left> <right> <rcutoff>
 * ```
 * where
 * - `<element-central> .....` element symbol of central atom
 * - `<element-neighbor1> ...` element symbol of neighbor atom 1
 * - `<element-neighbor2> ...` element symbol of neighbor atom 2
 * - `<rlow>.................` lower radial boundary
 * - `<left> ................` left angle boundary 
 * - `<right> ...............` right angle boundary 
 * - `<rcutoff> .............` upper radial boundary
 */
class SymmetryFunctionAngularPolyOnlyNarrow : public SymmetryFunction
{
public:
    /** Constructor, sets type = 99
    */
    SymmetryFunctionAngularPolyOnlyNarrow(ElementMap const& elementMap);
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
    // Core fcts
    bool         getCompactAngle(double x, double& fx, double& dfx) const;
    bool         getCompactRadial(double x, double& fx, double& dfx) const;
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
    /** Get private #e2 member variable.
     */
    std::size_t  getE2() const;
    /** Get private #angleLeft member variable.
     */
    double       getAngleLeft() const;
    /** Get private #angleRight member variable.
     */
    double       getAngleRight() const;
    /** Get private #rl member variable.
     */
    double       getRl() const;
    /** Get private #rs member variable.
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
    /** Check whether symmetry function is relevant for given element.
     *
     * @param[in] index Index of given element.
     * @return True if symmetry function is sensitive to given element, false
     *         otherwise.
     */
    bool         checkRelevantElement(std::size_t index) const;

private:
    /// Element index of neighbor atom 1.
    std::size_t     e1;
    /// Element index of neighbor atom 2.
    std::size_t     e2;
    /// Lower boundary @f$r_l@f$ of radial component.
    double          rl;
    /// Left angle boundary.
    double          angleLeft;
    /// Right angle boundary.
    double          angleRight;
    /// Compact function member; radial.
    CompactFunction cr;
    /// Compact function member; angular.
    CompactFunction ca;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool SymmetryFunctionAngularPolyOnlyNarrow::
operator!=(SymmetryFunction const& rhs) const
{
    return !((*this) == rhs);
}

inline bool SymmetryFunctionAngularPolyOnlyNarrow::
operator>(SymmetryFunction const& rhs) const
{
    return rhs < (*this);
}

inline bool SymmetryFunctionAngularPolyOnlyNarrow::
operator<=(SymmetryFunction const& rhs) const
{
    return !((*this) > rhs);
}

inline bool SymmetryFunctionAngularPolyOnlyNarrow::
operator>=(SymmetryFunction const& rhs) const
{
    return !((*this) < rhs);
}

inline std::size_t SymmetryFunctionAngularPolyOnlyNarrow::getE1() const
{
    return e1;
}

inline std::size_t SymmetryFunctionAngularPolyOnlyNarrow::getE2() const
{
    return e2;
}

inline double SymmetryFunctionAngularPolyOnlyNarrow::getAngleLeft() const
{
    return angleLeft;
}

inline double SymmetryFunctionAngularPolyOnlyNarrow::getAngleRight() const
{
    return angleRight;
}

inline double SymmetryFunctionAngularPolyOnlyNarrow::getRl() const
{
    return rl;
}

}

#endif
