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

#ifndef CBN_CUTOFFFUNCTION_H
#define CBN_CUTOFFFUNCTION_H

namespace nnpCbn
{

class CutoffFunction
{
  public:
    /** List of available cutoff function types.
     *
     * Most cutoff types allow the definition of an inner cutoff
     * @f$ r_{ci} := \alpha \, r_c@f$. Then the cutoff is equal to @f$1@f$ up
     * to the inner cutoff:
     *
     * @f$ f_c(r) =
     * \begin{cases}
     * 1, & \text{for } 0 \le r < r_{ci} \\
     * f(x), & \text{for } r_{ci} \le r < r_c \text{ where }
     * x := \frac{r - r_{ci}}{r_c - r_{ci}} \\
     * 0 & \text{for } r \geq r_c
     * \end{cases} @f$
     */
    enum CutoffType
    {
        /** @f$f(x) = 1@f$
         */
        CT_HARD,
        /** @f$f(x) = \frac{1}{2} \left[ \cos (\pi x) + 1\right] @f$
         */
        CT_COS,
        /** @f$f_c(r) = \tanh^3 \left(1 - \frac{r}{r_c} \right) @f$
         */
        CT_TANHU,
        /** @f$f_c(r) = c \tanh^3 \left(1 - \frac{r}{r_c} \right),\,
         *  f(0) = 1 @f$
         */
        CT_TANH,
        /** @f$f(x) = e^{-\frac{1}{1 - x^2}}@f$
         */
        CT_EXP,
        /** @f$f(x) = (2x - 3)x^2 + 1@f$
         */
        CT_POLY1,
        /** @f$f(x) = ((15 - 6x)x - 10) x^3 + 1@f$
         */
        CT_POLY2,
        /** @f$f(x) = (x(x(20x - 70) + 84) - 35)x^4 + 1@f$
         */
        CT_POLY3,
        /** @f$f(x) = (x(x((315 - 70x)x - 540) + 420) - 126)x^5 + 1@f$
         */
        CT_POLY4
    };

    /** Constructor, initializes to ´CT_HARD´.
     */
    CutoffFunction();
    CutoffFunction( double _rc );

  private:
    static double const PI;
    static double const PI_2;
    static double const E;
    static double const TANH_PRE;

    /// Cutoff function type.
    CutoffType cutoffType;
    /// Outer cutoff radius @f$r_c@f$.
    double rc;
    /// Inverse cutoff radius @f$\frac{1}{r_c}@f$.
    double rcinv;
    /// Inner cutoff for cutoff function types which allow shifting.
    double rci;
    /// Cutoff function parameter for `CT_POLYn` and `CT_EXP` @f$\alpha@f$.
    double alpha;
    /// Inverse width of cutoff function @f$\frac{1}{r_c - r_{ci}}@f$.
    double iw;
};

} // namespace nnpCbn

#endif
