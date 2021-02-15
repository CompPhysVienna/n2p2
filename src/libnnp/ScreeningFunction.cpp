// n2p2 - A neural network potential package
// Copyright (C) 2021 Andreas Singraber
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

#include "ScreeningFunction.h"
#include <stdexcept>

using namespace std;
using namespace nnp;

ScreeningFunction::ScreeningFunction() : inner(0.0),
                                         outer(0.0),
                                         scale(0.0)
{
    core.setType(CoreFunction::Type::POLY2);
}

void ScreeningFunction::setInnerOuter(double inner, double outer)
{
    if (inner >= outer)
    {
        throw invalid_argument("ERROR: Inner radius of transition region >= "
                               "outer radius.\n");
    }

    this->inner  = inner;
    this->outer  = outer;
    this->scale  = 1.0 / (outer - inner);

    return;
}
