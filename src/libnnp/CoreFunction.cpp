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
#include <stdexcept>

using namespace std;
using namespace nnp;

double const CoreFunction::E = exp(1.0);

CoreFunction::CoreFunction() : type      (Type::POLY2            ),
#ifndef NNP_NO_ASYM_POLY
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
