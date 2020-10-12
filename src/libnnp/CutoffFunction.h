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

#ifndef CUTOFFFUNCTION_H
#define CUTOFFFUNCTION_H

#include "CoreFunction.h"

namespace nnp
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
        /** @f$f(x) = e^{1 - \frac{1}{1 - x^2}}@f$
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
    /** Set cutoff type.
     *
     * @param[in] cutoffType Type of cutoff used.
     */
    void       setCutoffType(CutoffType const cutoffType);
    /** Getter for #cutoffType.
     *
     * @return CutoffType used.
     */
    CutoffType getCutoffType() const;
    /** Set cutoff radius.
     *
     * @param[in] cutoffRadius Cutoff radius @f$r_c@f$.
     */
    void       setCutoffRadius(double const cutoffRadius);
    /** Getter for #rc.
     *
     * @return Cutoff radius used.
     */
    double     getCutoffRadius() const;
    /** Set parameter for polynomial cutoff function (CT_POLY).
     *
     * @param[in] alpha Width parameter @f$\alpha@f$.
     */
    void       setCutoffParameter(double const alpha);
    /** Getter for #alpha.
     *
     * @return Cutoff parameter used.
     */
    double     getCutoffParameter() const;
    /** Cutoff function @f$f_c@f$.
     *
     * @param[in] r Distance.
     * @return Cutoff function value.
     */
    double     f(double r) const;
    /** Derivative of cutoff function @f$\frac{d f_c}{d r}@f$.
     *
     * @param[in] r Distance.
     * @return Value of cutoff function derivative.
     */
    double     df(double r) const;
    /** Calculate cutoff function @f$f_c@f$ and derivative
     * @f$\frac{d f_c}{d r}@f$.
     *
     * @param[in] r Distance.
     * @param[out] fc Cutoff function value.
     * @param[out] dfc Value of cutoff function derivative.
     */
    void       fdf(double r, double& fc, double& dfc) const;

private:
    static double const PI;
    static double const PI_2;
    static double const E;
    static double const TANH_PRE;

    /// Cutoff function type.
    CutoffType   cutoffType;
    /// Outer cutoff radius @f$r_c@f$.
    double       rc;
    /// Inverse cutoff radius @f$\frac{1}{r_c}@f$.
    double       rcinv;
    /// Inner cutoff for cutoff function types which allow shifting.
    double       rci;
    /// Cutoff function parameter for `CT_POLYn` and `CT_EXP` @f$\alpha@f$.
    double       alpha;
    /// Inverse width of cutoff function @f$\frac{1}{r_c - r_{ci}}@f$.
    double       iw;
    /// Core functions used by POLYN, if any.
    CoreFunction core;
    /// Function pointer to f.
    double       (CutoffFunction::*fPtr)(double r) const;
    /// Function pointer to df.
    double       (CutoffFunction::*dfPtr)(double r) const;
    /// Function pointer to fdf.
    void         (CutoffFunction::*fdfPtr)(double  r,
                                           double& fc,
                                           double& dfc) const;

    // Individual cutoff functions.
    double   fHARD (double r) const;
    double  dfHARD (double r) const;
    void   fdfHARD (double r, double& fc, double& dfc) const;

    double   fCOS  (double r) const;
    double  dfCOS  (double r) const;
    void   fdfCOS  (double r, double& fc, double& dfc) const;

    double   fTANHU(double r) const;
    double  dfTANHU(double r) const;
    void   fdfTANHU(double r, double& fc, double& dfc) const;

    double   fTANH (double r) const;
    double  dfTANH (double r) const;
    void   fdfTANH (double r, double& fc, double& dfc) const;

    double   fCORE (double r) const;
    double  dfCORE (double r) const;
    void   fdfCORE (double r, double& fc, double& dfc) const;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline CutoffFunction::CutoffType CutoffFunction::getCutoffType() const
{
    return cutoffType;
}

inline double CutoffFunction::getCutoffRadius() const
{
    return rc;
}

inline double CutoffFunction::getCutoffParameter() const
{
    return alpha;
}

inline double CutoffFunction::f(double r) const
{
    if (r >= rc) return 0.0;
    return (this->*fPtr)(r);
}

inline double CutoffFunction::df(double r) const
{
    if (r >= rc) return 0.0;
    return (this->*dfPtr)(r);
}

inline void CutoffFunction::fdf(double r, double& fc, double& dfc) const
{
    if (r >= rc)
    {
        fc = 0.0;
        dfc = 0.0;
        return;
    }
    (this->*fdfPtr)(r, fc, dfc);
    return;
}

inline double CutoffFunction::fHARD(double /*r*/) const
{
    return 1.0;
}

inline double CutoffFunction::dfHARD(double /*r*/) const
{
    return 0.0;
}

inline void CutoffFunction::fdfHARD(double /*r*/,
                                    double& fc,
                                    double& dfc) const
{
    fc = 1.0;
    dfc = 0.0;
    return;
}

}

#endif
