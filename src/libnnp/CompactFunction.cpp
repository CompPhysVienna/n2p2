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

#include "CompactFunction.h"
#include <stdexcept>

using namespace std;
using namespace nnp;

CompactFunction::CompactFunction() : center    (0.0  ),
                                     width     (0.0  ),
                                     left      (0.0  ),
                                     right     (0.0  ),
                                     scale     (0.0  )
{
    core.setType(CoreFunction::Type::POLY2);
}

#ifndef NNP_NO_ASYM_POLY
bool CompactFunction::getAsymmetric() const
{
    return core.getAsymmetric();
}

void CompactFunction::setAsymmetric(bool asymmetric)
{
    core.setAsymmetric(asymmetric);

    return;
}
#endif

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
