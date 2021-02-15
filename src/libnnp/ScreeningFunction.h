// n2p2 - A neural network potential package
// Copyright (C) 2021 Andreas Singraber
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

#ifndef SCREENINGFUNCTION_H
#define SCREENINGFUNCTION_H

#include "CoreFunction.h"

namespace nnp
{

/// A screening functions for use with electrostatics.
class ScreeningFunction
{
public:
    /** Constructor, initializes to #CoreFunction::Type::POLY2.
     */
    ScreeningFunction();
    /** Set type.
     *
     * @param[in] type Type of core function to use.
     */
    void               setCoreFunction(CoreFunction::Type const type);
    /** Set inner and outer limit of transition region.
     *
     * @param[in] inner Inner radius where transition region begins.
     * @param[in] outer Outer radius where transition region ends.
     *
     */
    void               setInnerOuter(double inner, double outer);
    /** Getter for #type.
     *
     * @return Type used.
     */
    CoreFunction::Type getCoreFunctionType() const;
    /** Getter for #inner.
     *
     * @return Inner radius where transition region starts.
     */
    double             getInner() const;
    /** Getter for #outer.
     *
     * @return Outer radius where transition region ends.
     */
    double             getOuter() const;
    /** Screening function @f$f_\text{screen}@f$.
     *
     * @param[in] r Radius argument.
     *
     * @return Function value.
     */
    double             f(double r) const;
    /** Derivative of screening function @f$\frac{d f_\text{screen}}{d r}@f$.
     *
     * @param[in] r Radius argument.
     *
     * @return Value of function derivative.
     */
    double             df(double r) const;
    /** Calculate screening function and derivative at once.
     *
     * @param[in] r Radius argument.
     * @param[out] fr Screening function value.
     * @param[out] dfr Value of screening function derivative.
     */
    void               fdf(double r, double& fr, double& dfr) const;

private:
    /// Inner radius where transition region starts.
    double       inner;
    /// Outer radius where transition region ends.
    double       outer;
    /// Inverse width.
    double       scale;
    /// Core function to be used in the transition region.
    CoreFunction core;

};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline void ScreeningFunction::setCoreFunction(CoreFunction::Type const type)
{
    core.setType(type);

    return;
}

inline CoreFunction::Type ScreeningFunction::getCoreFunctionType() const
{
    return core.getType();
}

inline double ScreeningFunction::getInner() const { return inner; }
inline double ScreeningFunction::getOuter() const { return outer; }

inline double ScreeningFunction::f(double r) const
{
    if (r <= inner) return 0.0;
    if (r >= outer) return 1.0;
    r = 1 - (r - inner) * scale;
    return core.f(r);
}

inline double ScreeningFunction::df(double r) const
{
    if (r <= inner || r >= outer) return 0.0;
    r = 1 - (r - inner) * scale;
    return -scale * core.df(r);
}

inline void ScreeningFunction::fdf(double r, double& fr, double& dfr) const
{
    if (r <= inner)
    {
        fr = 0.0;
        dfr = 0.0;
        return;
    }
    if (r >= outer)
    {
        fr = 1.0;
        dfr = 0.0;
        return;
    }
    r = 1 - (r - inner) * scale;
    core.fdf(r, fr, dfr);
    dfr = -dfr;

    return;
}

}

#endif
