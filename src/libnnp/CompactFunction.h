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

#ifndef WINDOWFUNCTION_H
#define WINDOWFUNCTION_H

#include "CoreFunction.h"

namespace nnp
{

/// A general function with compact support.
class CompactFunction
{
public:
    /// List of available function types with compact support.
    enum class Type
    {
        /** @f$f(x) = (2x - 3)x^2 + 1@f$
         */
        POLY1,
        /** @f$f(x) = ((15 - 6x)x - 10) x^3 + 1@f$
         */
        POLY2,
        /** @f$f(x) = (x(x(20x - 70) + 84) - 35)x^4 + 1@f$
         */
        POLY3,
        /** @f$f(x) = (x(x((315 - 70x)x - 540) + 420) - 126)x^5 + 1@f$
         */
        POLY4
    };

    /** Constructor, initializes to ´POLY2´.
     */
    CompactFunction();
    /** Set type.
     *
     * @param[in] type Type of compact function.
     */
    void setType(Type const type);
    /** Set center and width.
     *
     * @param[in] center Center of compact function.
     * @param[in] width Width of compact function.
     *
     * @note Either use setCenterWidth() or setLeftRight() to initialize.
     */
    void setCenterWidth(double center, double width);
    /** Set left and right boundary.
     *
     * @param[in] left Left boundary of compact function.
     * @param[in] right Right boundary of compact function.
     *
     * @note Either use setCenterWidth() or setLeftRight() to initialize.
     */
    void setLeftRight(double left, double right);
    /** Getter for #type.
     *
     * @return Type used.
     */
    Type getType() const;
    /** Getter for #center.
     *
     * @return Center of compact function.
     */
    double     getCenter() const;
    /** Getter for #width.
     *
     * @return Width of compact function.
     */
    double     getWidth() const;
    /** Getter for #left.
     *
     * @return Left boundary of compact function.
     */
    double     getLeft() const;
    /** Getter for #right.
     *
     * @return Right boundary of compact function.
     */
    double     getRight() const;
    /** Compact function @f$f_c@f$.
     *
     * @param[in] a Function argument.
     * @return Function value.
     */
    double     f(double a) const;
    /** Derivative of compact function @f$\frac{d f_c}{d a}@f$.
     *
     * @param[in] a Function argument.
     * @return Value of function derivative.
     */
    double     df(double a) const;
    /** Calculate compact function and derivative at once.
     *
     * @param[in] a Function argument.
     * @param[out] fa Cutoff function value.
     * @param[out] dfa Value of cutoff function derivative.
     */
    void       fdf(double a, double& fa, double& dfa) const;

private:
    /// Compact function type.
    Type         type;
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
    /// Core functions used by POLYN, if any.
    CoreFunction core;

};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline CompactFunction::Type CompactFunction::getType() const
{
    return type;
}

inline double CompactFunction::getCenter() const
{
    return center;
}

inline double CompactFunction::getWidth() const
{
    return width;
}

inline double CompactFunction::getLeft() const
{
    return left;
}

inline double CompactFunction::getRight() const
{
    return right;
}

}

#endif
