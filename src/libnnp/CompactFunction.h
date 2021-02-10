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

#ifndef COMPACTFUNCTION_H
#define COMPACTFUNCTION_H

#include "CoreFunction.h"

namespace nnp
{

/// A general function with compact support.
class CompactFunction
{
public:
    /** Constructor, initializes to #CoreFunction::Type::POLY2.
     */
    CompactFunction();
    /** Set type.
     *
     * @param[in] type Type of core function to use.
     */
    void               setCoreFunction(CoreFunction::Type const type);
    /** Set center and width.
     *
     * @param[in] center Center of compact function.
     * @param[in] width Width of compact function.
     *
     * @note Either use setCenterWidth() or setLeftRight() to initialize.
     */
    void               setCenterWidth(double center, double width);
    /** Set left and right boundary.
     *
     * @param[in] left Left boundary of compact function.
     * @param[in] right Right boundary of compact function.
     *
     * @note Either use setCenterWidth() or setLeftRight() to initialize.
     */
    void               setLeftRight(double left, double right);
    /** Getter for #type.
     *
     * @return Type used.
     */
    CoreFunction::Type getCoreFunctionType() const;
#ifndef NNP_NO_ASYM_POLY
    /** Set asymmetric property in core function.
     *
     * @param[in] asymmetric Whether asymmetry should be activated.
     */
    void               setAsymmetric(bool asymmetric);
    /** Check if asymmetry is enabled in core function.
     *
     * @return Whether asymmetry is activated.
     */
    bool               getAsymmetric() const;
#endif
    /** Getter for #center.
     *
     * @return Center of compact function.
     */
    double             getCenter() const;
    /** Getter for #width.
     *
     * @return Width of compact function.
     */
    double             getWidth() const;
    /** Getter for #left.
     *
     * @return Left boundary of compact function.
     */
    double             getLeft() const;
    /** Getter for #right.
     *
     * @return Right boundary of compact function.
     */
    double             getRight() const;
    /** Compact function @f$f_c@f$.
     *
     * @param[in] a Function argument.
     * @return Function value.
     */
    double             f(double a) const;
    /** Derivative of compact function @f$\frac{d f_c}{d a}@f$.
     *
     * @param[in] a Function argument.
     * @return Value of function derivative.
     */
    double             df(double a) const;
    /** Calculate compact function and derivative at once.
     *
     * @param[in] a Function argument.
     * @param[out] fa Cutoff function value.
     * @param[out] dfa Value of cutoff function derivative.
     */
    void               fdf(double a, double& fa, double& dfa) const;

private:
    /// Center of compact function.
    double       center;
    /// Width of compact function.
    double       width;
    /// Left boundary of compact function.
    double       left;
    /// Right boundary of compact function.
    double       right;
    /// Inverse width.
    double       scale;
    /// Core function to be used on either side of compact function.
    CoreFunction core;

};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline void CompactFunction::setCoreFunction(CoreFunction::Type const type)
{
    core.setType(type);

    return;
}

inline CoreFunction::Type CompactFunction::getCoreFunctionType() const
{
    return core.getType();
}

inline double CompactFunction::getCenter() const { return center; }
inline double CompactFunction::getWidth() const { return width; }
inline double CompactFunction::getLeft() const { return left; }
inline double CompactFunction::getRight() const { return right; }

inline double CompactFunction::f(double a) const
{
    a = (a - center) * scale;
    return core.f(std::abs(a));
}

inline double CompactFunction::df(double a) const
{
    a = (a - center) * scale;
    return copysign(scale * core.df(std::abs(a)), a);
}

inline void CompactFunction::fdf(double a, double& fa, double& dfa) const
{
    a = (a - center) * scale;
    core.fdf(std::abs(a), fa, dfa);
    dfa *= copysign(scale, a);

    return;
}


}

#endif
