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

#include "CoreFunction.h"
#include "utility.h"
#include <stdexcept>

using namespace std;
using namespace nnp;

double const CoreFunction::PI = 4.0 * atan(1.0);
double const CoreFunction::PI_2 = 2.0 * atan(1.0);
double const CoreFunction::E = exp(1.0);

CoreFunction::CoreFunction() : type      (Type::POLY2            ),
#ifndef N2P2_NO_ASYM_POLY
                               asymmetric(false                  ),
#endif
                               fPtr      (&CoreFunction::  fPOLY2),
                               dfPtr     (&CoreFunction:: dfPOLY2),
                               fdfPtr    (&CoreFunction::fdfPOLY2)
{
}

void CoreFunction::setType(Type const type)
{
    this->type = type;

    if (type == Type::COS)
    {
          fPtr = &CoreFunction::  fCOS;
         dfPtr = &CoreFunction:: dfCOS;
        fdfPtr = &CoreFunction::fdfCOS;
    }
    else if (type == Type::POLY1)
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
    else if (type == Type::EXP)
    {
          fPtr = &CoreFunction::  fEXP;
         dfPtr = &CoreFunction:: dfEXP;
        fdfPtr = &CoreFunction::fdfEXP;
    }
    else
    {
        throw invalid_argument("ERROR: Unknown core function.\n");
    }

    return;
}

void CoreFunction::setType(string const typeString)
{
    CoreFunction::Type t;

    if      (typeString == "c")  t = CoreFunction::Type::COS;
    else if (typeString == "p1") t = CoreFunction::Type::POLY1;
    else if (typeString == "p2") t = CoreFunction::Type::POLY2;
    else if (typeString == "p3") t = CoreFunction::Type::POLY3;
    else if (typeString == "p4") t = CoreFunction::Type::POLY4;
    else if (typeString == "e")  t = CoreFunction::Type::EXP;
    else
    {
        throw invalid_argument("ERROR: Unknown CoreFunction type.\n");
    }

    setType(t);

    return;
}

vector<string> CoreFunction::info() const
{
    vector<string> v;

    if (type == Type::COS)
    {
        v.push_back(strpr("CoreFunction::Type::COS (%d):\n", type));
        v.push_back("f(x) := 1/2 * (cos(pi*x) + 1)\n");
    }
    else if (type == Type::POLY1)
    {
        v.push_back(strpr("CoreFunction::Type::POLY1 (%d):\n", type));
        v.push_back("f(x) := (2x - 3)x^2 + 1\n");
    }
    else if (type == Type::POLY2)
    {
        v.push_back(strpr("CoreFunction::Type::POLY2 (%d):\n", type));
        v.push_back("f(x) := ((15 - 6x)x - 10)x^3 + 1\n");
    }
    else if (type == Type::POLY3)
    {
        v.push_back(strpr("CoreFunction::Type::POLY3 (%d):\n", type));
        v.push_back("f(x) := (x(x(20x - 70) + 84) - 35)x^4 + 1\n");
    }
    else if (type == Type::POLY4)
    {
        v.push_back(strpr("CoreFunction::Type::POLY4 (%d):\n", type));
        v.push_back("f(x) := (x(x((315 - 70x)x - 540) + 420) - 126)x^5 + 1\n");
    }
    else if (type == Type::EXP)
    {
        v.push_back(strpr("CoreFunction::Type::EXP (%d):\n", type));
        v.push_back("f(x) := exp(-1 / 1 - x^2)\n");
    }

    return v;
}
