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

#include "SymGrpBaseComp.h"
#include "utility.h"

using namespace std;
using namespace nnp;

SymGrpBaseComp::SymGrpBaseComp(size_t type, ElementMap const& elementMap) :
    SymGrp(type, elementMap),
    rmin(0.0),
    rmax(0.0)
{
    parametersCommon.insert("rs/rl");
    parametersCommon.insert("rc");

    parametersMember.insert("subtype");
    parametersMember.insert("rs/rl");
    parametersMember.insert("rc");
}
