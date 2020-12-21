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

#ifndef SYMFNCBASECOMPANGWEIGHTED_H
#define SYMFNCBASECOMPANGWEIGHTED_H

#include "SymFncBaseComp.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/// Intermediate symmetry function class for weighted angular compact SFs.
class SymFncBaseCompAngWeighted : public SymFncBaseComp
{
public:
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
    // Core fcts
    void                getCompactAngle(double const x,
                                        double&      fx,
                                        double&      dfx) const;
    void                getCompactRadial(double const x,
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
    /** Get private #angleLeft member variable.
     */
    double              getAngleLeft() const;
    /** Get private #angleRight member variable.
     */
    double              getAngleRight() const;
    /** Calculate (partial) symmetry function value for one given distance.
     *
     * @param[in] distance Distance between two atoms.
     * @return @f$\left(e^{-\eta r^2} f_c(r)\right)^2@f$
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

protected:
    /// Left angle boundary.
    double          angleLeft;
    /// Right angle boundary.
    double          angleRight;
    /// Left angle boundary in radians.
    double          angleLeftRadians;
    /// Right angle boundary in radians.
    double          angleRightRadians;
    /// Compact function member for angular part.
    CompactFunction ca;

    /** Constructor, initializes #type.
     */
    SymFncBaseCompAngWeighted(std::size_t type, ElementMap const&);
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline double SymFncBaseCompAngWeighted::getAngleLeft() const
{
    return angleLeft;
}

inline double SymFncBaseCompAngWeighted::getAngleRight() const
{
    return angleRight;
}

inline void SymFncBaseCompAngWeighted::getCompactAngle(double const x,
                                                       double&      fx,
                                                       double&      dfx) const
{
    ca.fdf(x, fx, dfx);
    return;
}

inline void SymFncBaseCompAngWeighted::getCompactRadial(double const x,
                                                        double&      fx,
                                                        double&      dfx) const
{
    cr.fdf(x, fx, dfx);
    return;
}

}

#endif
