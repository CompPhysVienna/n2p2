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

#include "SymGrpBaseExpAng.h"
#include "Atom.h"
#include "SymFnc.h"
#include "SymFncBaseExpAng.h"
#include "Vec3D.h"
#include "utility.h"
#include <algorithm> // std::sort
#include <cmath>     // exp
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

SymGrpBaseExpAng::SymGrpBaseExpAng(size_t type, ElementMap const& elementMap) :
    SymGrpBaseCutoff(type, elementMap),
    e1(0),
    e2(0)
{
    parametersCommon.insert("e1");
    parametersCommon.insert("e2");

    parametersMember.insert("eta");
    parametersMember.insert("rs/rl");
    parametersMember.insert("lambda");
    parametersMember.insert("zeta");
    parametersMember.insert("calcexp");
}

void SymGrpBaseExpAng::setScalingFactors()
{
    vector<SymFncBaseExpAng const*> members = getMembers();

    scalingFactors.resize(members.size(), 0.0);
    for (size_t i = 0; i < members.size(); i++)
    {
        scalingFactors.at(i) = members[i]->getScalingFactor();
        factorNorm.at(i) *= scalingFactors.at(i);
    }

    return;
}

vector<string> SymGrpBaseExpAng::parameterLines() const
{
    vector<string> v;
    vector<SymFncBaseExpAng const*> members = getMembers();

    v.push_back(strpr(getPrintFormatCommon().c_str(),
                      index + 1,
                      elementMap[ec].c_str(),
                      type,
                      subtype.c_str(),
                      elementMap[e1].c_str(),
                      elementMap[e2].c_str(),
                      rc / convLength,
                      cutoffAlpha));

    for (size_t i = 0; i < members.size(); ++i)
    {
        v.push_back(strpr(getPrintFormatMember().c_str(),
                          members[i]->getEta() * convLength * convLength,
                          members[i]->getRs() / convLength,
                          members[i]->getLambda(),
                          members[i]->getZeta(),
                          members[i]->getLineNumber() + 1,
                          i + 1,
                          members[i]->getIndex() + 1,
                          (int)calculateExp[i]));
    }

    return v;
}
