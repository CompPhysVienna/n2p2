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

#include "CompactFunction.h"
#include <stdexcept>

using namespace std;
using namespace nnp;

CompactFunction::CompactFunction() : type  (Type::POLY2),
                                     center(0.0        ),
                                     width (0.0        ),
                                     left  (0.0        ),
                                     right (0.0        ),
                                     scale (0.0        )
{
    core.setType(CoreFunction::Type::POLY2);
}

void CompactFunction::setType(Type const type)
{
    this->type = type;
    
    if      (type == Type::POLY1) core.setType(CoreFunction::Type::POLY1);
    else if (type == Type::POLY2) core.setType(CoreFunction::Type::POLY2);
    else if (type == Type::POLY3) core.setType(CoreFunction::Type::POLY3);
    else if (type == Type::POLY4) core.setType(CoreFunction::Type::POLY4);
    else
    {
        throw invalid_argument("ERROR: Unknown compact function type.\n");
    }

    return;
}

void CompactFunction::setCenterWidth(double center, double width)
{
    if (width <= 0.0)
    {
        throw invalid_argument("ERROR: Width <= 0.\n");
    }

    this->center = center;
    this->width  = width;
    this->left   = center - width;
    this->right  = center + width;
    this->scale  = 1.0 / width;

    return;
}

void CompactFunction::setLeftRight(double left, double right)
{
    if (left >= right)
    {
        throw invalid_argument("ERROR: Left >= Right.\n");
    }

    this->left   = left;
    this->right  = right;
    this->center = (right + left) / 2.0;
    this->width  = (right - left) / 2.0;
    this->scale  = 1.0 / width;

    return;
}

double CompactFunction::f(double a) const
{
    if (a <= left || a >= right) return 0.0;
    if (a == center) return 1.0;
    a = (a - center) * scale;
    if (a < 0) return core.f(-a);
    else       return core.f(a);
}

double CompactFunction::df(double a) const
{
    if (a <= left || a >= right || a == center) return 0.0;
    a = (a - center) * scale;
    if (a < 0) return -scale * core.df(-a);
    else       return  scale * core.df(a);
}

void CompactFunction::fdf(double a, double& fa, double& dfa) const
{
    if (a <= left || a >= right)
    {
        fa  = 0.0;
        dfa = 0.0;
        return;
    }
    else if (a == center)
    {
        fa  = 1.0;
        dfa = 0.0;
        return;
    }

    a = (a - center) * scale;
    if (a < 0)
    {
       core.fdf(-a, fa, dfa);
       dfa *= -scale;
    }
    else
    {
       core.fdf(a, fa, dfa);
       dfa *= scale;
    }

    return;
}

