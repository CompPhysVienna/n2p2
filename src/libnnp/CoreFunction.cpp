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

#include "CoreFunction.h"
#include <stdexcept>

using namespace std;
using namespace nnp;

CoreFunction::CoreFunction() : type  (Type::POLY2            ),
                               fPtr  (&CoreFunction::  fPOLY2),
                               dfPtr (&CoreFunction:: dfPOLY2),
                               fdfPtr(&CoreFunction::fdfPOLY2)
{
}

void CoreFunction::setType(Type const type)
{
    this->type = type;

    if (type == Type::POLY1)
    {
          fPtr = &CoreFunction::  fPOLY1;
         dfPtr = &CoreFunction:: dfPOLY1;
        fdfPtr = &CoreFunction::fdfPOLY1;
    }
    else if (type == Type::POLY2)
    {
          fPtr = &CoreFunction::  fPOLY2;
         dfPtr = &CoreFunction:: dfPOLY2;
        fdfPtr = &CoreFunction::fdfPOLY2;
    }
    else if (type == Type::POLY3)
    {
          fPtr = &CoreFunction::  fPOLY3;
         dfPtr = &CoreFunction:: dfPOLY3;
        fdfPtr = &CoreFunction::fdfPOLY3;
    }
    else if (type == Type::POLY4)
    {
          fPtr = &CoreFunction::  fPOLY4;
         dfPtr = &CoreFunction:: dfPOLY4;
        fdfPtr = &CoreFunction::fdfPOLY4;
    }
    else
    {
        throw invalid_argument("ERROR: Unknown function.\n");
    }

    return;
}

double CoreFunction::fPOLY1(double x) const
{
    return (2.0 * x - 3.0) * x * x + 1.0;
}

double CoreFunction::dfPOLY1(double x) const
{
    return x * (6.0 * x - 6.0);
}

void CoreFunction::fdfPOLY1(double x, double& fx, double& dfx) const
{
    fx = (2.0 * x - 3.0) * x * x + 1.0;
    dfx = x * (6.0 * x - 6.0);
    return;
}

double CoreFunction::fPOLY2(double x) const
{
    return ((15.0 - 6.0 * x) * x - 10.0) * x * x * x + 1.0;
}

double CoreFunction::dfPOLY2(double x) const
{
    return x * x * ((60.0 - 30.0 * x) * x - 30.0);
}

void CoreFunction::fdfPOLY2(double x, double& fx, double& dfx) const
{
    double const x2 = x * x;
    fx = ((15.0 - 6.0 * x) * x - 10.0) * x * x2 + 1.0;
    dfx = x2 * ((60.0 - 30.0 * x) * x - 30.0);
    return;
}

double CoreFunction::fPOLY3(double x) const
{
    double const x2 = x * x;
    return (x * (x * (20.0 * x - 70.0) + 84.0) - 35.0) * x2 * x2 + 1.0;
}

double CoreFunction::dfPOLY3(double x) const
{
    return x * x * x * (x * (x * (140.0 * x - 420.0) + 420.0) - 140.0);
}

void CoreFunction::fdfPOLY3(double x, double& fx, double& dfx) const
{
    double const x2 = x * x;
    fx = (x * (x * (20.0 * x - 70.0) + 84.0) - 35.0) * x2 * x2 + 1.0;
    dfx = x2 * x * (x * (x * (140.0 * x - 420.0) + 420.0) - 140.0);
    return;
}

double CoreFunction::fPOLY4(double x) const
{
    double const x2 = x * x;
    return (x * (x * ((315.0 - 70.0 * x) * x - 540.0) + 420.0) - 126.0) *
           x2 * x2 * x + 1.0;
}

double CoreFunction::dfPOLY4(double x) const
{
    double const x2 = x * x;
    return x2 * x2 *
           (x * (x * ((2520.0 - 630.0 * x) * x - 3780.0) + 2520.0) - 630.0);
}

void CoreFunction::fdfPOLY4(double x, double& fx, double& dfx) const
{
    double x4 = x * x;
    x4 *= x4;
    fx = (x * (x * ((315.0 - 70.0 * x) * x - 540.0) + 420.0) - 126.0) *
         x * x4 + 1.0;
    dfx = x4 *
          (x * (x * ((2520.0 - 630.0 * x) * x - 3780.0) + 2520.0) - 630.0);
    return;
}
