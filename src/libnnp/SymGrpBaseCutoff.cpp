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

#include "SymGrpBaseCutoff.h"
#include "utility.h"

using namespace std;
using namespace nnp;

SymGrpBaseCutoff::SymGrpBaseCutoff(size_t type, ElementMap const& elementMap) :
    SymGrp(type, elementMap),
    rc         (0.0                    ),
    cutoffAlpha(0.0                    ),
    subtype    ("ct0"                  ),
    cutoffType (CutoffFunction::CT_HARD)
{
    // Add standard common parameter IDs to set.
    parametersCommon.insert("subtype");
    parametersCommon.insert("rc");
    parametersCommon.insert("alpha");
}
