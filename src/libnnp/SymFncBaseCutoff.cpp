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

#include "SymFncBaseCutoff.h"
#include "utility.h"
#include <string>

using namespace std;
using namespace nnp;

vector<string> SymFncBaseCutoff::parameterInfo() const
{
    vector<string> v = SymFnc::parameterInfo();
    string s;
    size_t w = sfinfoWidth;

    s = "subtype";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), subtype.c_str()));
    s = "alpha";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), cutoffAlpha));

    return v;
}

void SymFncBaseCutoff::setCutoffFunction(
                                        CutoffFunction::CutoffType cutoffType,
                                        double                     cutoffAlpha)
{
    this->cutoffType = cutoffType;
    this->cutoffAlpha = cutoffAlpha;
    fc.setCutoffType(cutoffType);
    fc.setCutoffParameter(cutoffAlpha);
    this->subtype = string("ct") + strpr("%d", static_cast<int>(cutoffType));

    return;
}

SymFncBaseCutoff::SymFncBaseCutoff(size_t type,
                                     ElementMap const& elementMap) :
    SymFnc(type, elementMap),
    cutoffAlpha  (0.0                    ),
    subtype      ("ct0"                  ),
    cutoffType   (CutoffFunction::CT_HARD)
{
    // Add cutoff-related parameter IDs to set.
    parameters.insert("subtype");
    parameters.insert("alpha");
}
