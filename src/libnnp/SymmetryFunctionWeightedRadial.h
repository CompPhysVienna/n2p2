// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SYMMETRYFUNCTIONWEIGHTEDRADIAL_H
#define SYMMETRYFUNCTIONWEIGHTEDRADIAL_H

#include "SymmetryFunction.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class ElementMap;
class SymmetryFunctionStatistics;

/** Weighted radial symmetry function (type 12)
 *
 * @f[
 * G^{12}_i = \sum_{j \neq i} Z_j \,
 *            \mathrm{e}^{-\eta(r_{ij} - r_\mathrm{s})^2}
 *            f_c(r_{ij}) 
 * @f]
 * Parameter string:
 * ```
 * <element-central> 12 <eta> <rshift> <rcutoff>
 * ```
 * where
 * - `<element-central> ....` element symbol of central atom
 * - `<eta> ................` @f$\eta@f$ 
 * - `<rshift> .............` @f$r_\mathrm{s}@f$
 * - `<rcutoff> ............` @f$r_c@f$
 */
class SymmetryFunctionWeightedRadial : public SymmetryFunction
{
public:
    /** Constructor, sets type = 12
     */
    SymmetryFunctionWeightedRadial(ElementMap const& elementMap);
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
     * @param[in] parameterString String containing weighted radial symmetry
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
    /// Width @f$\eta@f$ of gaussian.
    double      eta;
    /// Shift @f$r_s@f$ of gaussian.
    double      rs;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool SymmetryFunctionWeightedRadial::operator!=(
                                             SymmetryFunction const& rhs) const
{
    return !((*this) == rhs);
}

inline bool SymmetryFunctionWeightedRadial::operator>(
                                             SymmetryFunction const& rhs) const
{
    return rhs < (*this);
}

inline bool SymmetryFunctionWeightedRadial::operator<=(
                                             SymmetryFunction const& rhs) const
{
    return !((*this) > rhs);
}

inline bool SymmetryFunctionWeightedRadial::operator>=(
                                             SymmetryFunction const& rhs) const
{
    return !((*this) < rhs);
}

inline double SymmetryFunctionWeightedRadial::getEta() const
{
    return eta;
}

inline double SymmetryFunctionWeightedRadial::getRs() const
{
    return rs;
}

}

#endif
