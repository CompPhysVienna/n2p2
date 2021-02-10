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

#ifndef SYMFNCBASEEXPANG_H
#define SYMFNCBASEEXPANG_H

#include "SymFncBaseCutoff.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/// Intermediate class for angular SFs based on cutoffs and exponentials.
class SymFncBaseExpAng : public SymFncBaseCutoff
{
public:
    /** Set symmetry function parameters.
     *
     * @param[in] parameterString String containing angular symmetry function
     *                            parameters.
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
     * @return @f$\left(e^{-\eta r^2} f_c(r)\right)@f$
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
    /** Get private #useIntegerPow member variable.
     */
    bool                getUseIntegerPow() const;
    /** Get private #e1 member variable.
     */
    std::size_t         getE1() const;
    /** Get private #e2 member variable.
     */
    std::size_t         getE2() const;
    /** Get private #zetaInt member variable.
     */
    int                 getZetaInt() const;
    /** Get private #lambda member variable.
     */
    double              getLambda() const;
    /** Get private #eta member variable.
     */
    double              getEta() const;
    /** Get private #zeta member variable.
     */
    double              getZeta() const;
    /** Get private #rs member variable.
     */
    double              getRs() const;

protected:
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
    /// Shift @f$r_s@f$ of gaussian.
    double      rs;

    /** Constructor, initializes #type.
     */
    SymFncBaseExpAng(std::size_t type, ElementMap const&);
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool SymFncBaseExpAng::getUseIntegerPow() const { return useIntegerPow; }
inline std::size_t SymFncBaseExpAng::getE1() const { return e1; }
inline std::size_t SymFncBaseExpAng::getE2() const { return e2; }
inline int SymFncBaseExpAng::getZetaInt() const { return zetaInt; }
inline double SymFncBaseExpAng::getLambda() const { return lambda; }
inline double SymFncBaseExpAng::getEta() const { return eta; }
inline double SymFncBaseExpAng::getZeta() const { return zeta; }
inline double SymFncBaseExpAng::getRs() const { return rs; }

}

#endif
