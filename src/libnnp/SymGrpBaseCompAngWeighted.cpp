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

#include "SymGrpBaseCompAngWeighted.h"
#include "SymFncBaseCompAngWeighted.h"
#include "Atom.h"
#include "SymFnc.h"
#include "Vec3D.h"
#include "utility.h"
#include <algorithm> // std::sort
#include <cmath>     // exp
#include <stdexcept> // std::runtime_error
using namespace std;
using namespace nnp;

SymGrpBaseCompAngWeighted::
SymGrpBaseCompAngWeighted(size_t type, ElementMap const& elementMap) :
    SymGrpBaseComp(type, elementMap)
{
    parametersMember.insert("angleLeft");
    parametersMember.insert("angleRight");
    parametersMember.insert("calcexp");
}

void SymGrpBaseCompAngWeighted::setScalingFactors()
{
    vector<SymFncBaseCompAngWeighted const*> members = getMembers();
    scalingFactors.resize(members.size(), 0.0);
    for (size_t i = 0; i < members.size(); i++)
    {
        scalingFactors.at(i) = members[i]->getScalingFactor();
    }

    return;
}

vector<string> SymGrpBaseCompAngWeighted::parameterLines() const
{
    vector<string> v;
    vector<SymFncBaseCompAngWeighted const*> members = getMembers();

    v.push_back(strpr(getPrintFormatCommon().c_str(),
                      index + 1,
                      elementMap[ec].c_str(),
                      type,
                      rmin / convLength,
                      rmax / convLength));

    for (size_t i = 0; i < members.size(); ++i)
    {
        v.push_back(strpr(getPrintFormatMember().c_str(),
                          members[i]->getSubtype().c_str(),
                          members[i]->getRl() / convLength,
                          members[i]->getRc() / convLength,
                          members[i]->getAngleLeft(),
                          members[i]->getAngleRight(),
                          members[i]->getLineNumber() + 1,
                          i + 1,
                          members[i]->getIndex() + 1,
                          calculateComp[i]));
    }

    return v;
}
