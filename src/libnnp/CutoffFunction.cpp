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

#include "CutoffFunction.h"
#include <stdexcept>
#include <cmath>  // cos, sin, tanh, exp, pow
#include <limits> // std::numeric_limits

using namespace std;
using namespace nnp;

double const CutoffFunction::PI = 4.0 * atan(1.0);
double const CutoffFunction::PI_2 = 2.0 * atan(1.0);
double const CutoffFunction::E = exp(1.0);
double const CutoffFunction::TANH_PRE = pow((E + 1 / E) / (E - 1 / E), 3);

CutoffFunction::CutoffFunction() : cutoffType(CT_HARD                 ),
                                   rc        (0.0                     ),
                                   rcinv     (0.0                     ),
                                   rci       (0.0                     ),
                                   alpha     (0.0                     ),
                                   iw        (0.0                     ),
                                   fPtr      (&CutoffFunction::  fHARD),
                                   dfPtr     (&CutoffFunction:: dfHARD),
                                   fdfPtr    (&CutoffFunction::fdfHARD)
{
}

void CutoffFunction::setCutoffType(CutoffType const cutoffType)
{
    this->cutoffType = cutoffType;

    if (cutoffType == CT_HARD)
    {
          fPtr = &CutoffFunction::  fHARD;
         dfPtr = &CutoffFunction:: dfHARD;
        fdfPtr = &CutoffFunction::fdfHARD;
    }
    else if (cutoffType == CT_COS)
    {
          fPtr = &CutoffFunction::  fCOS;
         dfPtr = &CutoffFunction:: dfCOS;
        fdfPtr = &CutoffFunction::fdfCOS;
    }
    else if (cutoffType == CT_TANHU)
    {
          fPtr = &CutoffFunction::  fTANHU;
         dfPtr = &CutoffFunction:: dfTANHU;
        fdfPtr = &CutoffFunction::fdfTANHU;
    }
    else if (cutoffType == CT_TANH)
    {
          fPtr = &CutoffFunction::  fTANH;
         dfPtr = &CutoffFunction:: dfTANH;
        fdfPtr = &CutoffFunction::fdfTANH;
    }
    else if (cutoffType == CT_POLY1 ||
             cutoffType == CT_POLY2 ||
             cutoffType == CT_POLY3 ||
             cutoffType == CT_POLY4 ||
             cutoffType == CT_EXP)
    {
        using CFT = CoreFunction::Type;
        if      (cutoffType == CT_POLY1) core.setType(CFT::POLY1);
        else if (cutoffType == CT_POLY2) core.setType(CFT::POLY2);
        else if (cutoffType == CT_POLY3) core.setType(CFT::POLY3);
        else if (cutoffType == CT_POLY4) core.setType(CFT::POLY4);
        else if (cutoffType == CT_EXP)   core.setType(CFT::EXP);

          fPtr = &CutoffFunction::  fCORE;
         dfPtr = &CutoffFunction:: dfCORE;
        fdfPtr = &CutoffFunction::fdfCORE;
    }
    else
    {
        throw invalid_argument("ERROR: Unknown cutoff type.\n");
    }

    return;
}

void CutoffFunction::setCutoffRadius(double const cutoffRadius)
{
    rc = cutoffRadius;
    rcinv = 1.0 / cutoffRadius;
    return;
}

void CutoffFunction::setCutoffParameter(double const alpha)
{
    if (alpha < 0.0 || alpha >= 1.0)
    {
        throw invalid_argument("ERROR: 0 <= alpha < 1.0 is required.\n");
    }
    this->alpha = alpha;
    rci = rc * alpha;
    iw = 1.0 / (rc - rci);
    return;
}

double CutoffFunction::fCOS(double r) const
{
    if (r < rci) return 1.0;
    double const x = (r - rci) * iw;
    return 0.5 * (cos(PI * x) + 1.0);
}

double CutoffFunction::dfCOS(double r) const
{
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    return -PI_2 * iw * sin(PI * x);
}

void CutoffFunction::fdfCOS(double r, double& fc, double& dfc) const
{
    if (r < rci)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - rci) * iw;
    double const temp = cos(PI * x);
    fc = 0.5 * (temp + 1.0);
    dfc = -0.5 * iw * PI * sqrt(1.0 - temp * temp);
    return;
}

double CutoffFunction::fTANHU(double r) const
{
    double const temp = tanh(1.0 - r * rcinv);
    return temp * temp * temp;
}

double CutoffFunction::dfTANHU(double r) const
{
    double temp = tanh(1.0 - r * rcinv);
    temp *= temp;
    return 3.0 * temp * (temp - 1.0) * rcinv;
}

void CutoffFunction::fdfTANHU(double r, double& fc, double& dfc) const
{
    double const temp = tanh(1.0 - r * rcinv);
    double const temp2 = temp * temp;
    fc = temp * temp2;
    dfc = 3.0 * temp2 * (temp2 - 1.0) * rcinv;
    return;
}

double CutoffFunction::fTANH(double r) const
{
    double const temp = tanh(1.0 - r * rcinv);
    return TANH_PRE * temp * temp * temp;
}

double CutoffFunction::dfTANH(double r) const
{
    double temp = tanh(1.0 - r * rcinv);
    temp *= temp;
    return 3.0 * TANH_PRE * temp * (temp - 1.0) * rcinv;
}

void CutoffFunction::fdfTANH(double r, double& fc, double& dfc) const
{
    double const temp = tanh(1.0 - r * rcinv);
    double const temp2 = temp * temp;
    fc = TANH_PRE * temp * temp2;
    dfc = 3.0 * TANH_PRE * temp2 * (temp2 - 1.0) * rcinv;
    return;
}

double CutoffFunction::fCORE(double r) const
{
    if (r < rci) return 1.0;
    double const x = (r - rci) * iw;
    return core.f(x);
}

double CutoffFunction::dfCORE(double r) const
{
    if (r < rci) return 0.0;
    double const x = (r - rci) * iw;
    return iw * core.df(x);
}

void CutoffFunction::fdfCORE(double r, double& fc, double& dfc) const
{
    if (r < rci)
    {
        fc = 1.0;
        dfc = 0.0;
        return;
    }
    double const x = (r - rci) * iw;
    core.fdf(x, fc, dfc);
    dfc *= iw;
    return;
}
