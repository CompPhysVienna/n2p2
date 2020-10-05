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

#include "SymFncCutoffBased.h"
#include "utility.h"

using namespace std;
using namespace nnp;

vector<string> SymFncCutoffBased::parameterInfo() const
{
    vector<string> v = SymFnc::parameterInfo();
    string s;
    size_t w = sfinfoWidth;

    s = "cutoffType";
    v.push_back(strpr((pad(s, w) + "%d"    ).c_str(), (int)cutoffType));
    s = "cutoffAlpha";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), cutoffAlpha));

    return v;
}

void SymFncCutoffBased::setCutoffFunction(
                                        CutoffFunction::CutoffType cutoffType,
                                        double                     cutoffAlpha)
{
    this->cutoffType = cutoffType;
    this->cutoffAlpha = cutoffAlpha;
    fc.setCutoffType(cutoffType);
    fc.setCutoffParameter(cutoffAlpha);

    return;
}

SymFncCutoffBased::SymFncCutoffBased(size_t type,
                                     ElementMap const& elementMap) :
    SymFnc(type, elementMap),
    cutoffAlpha  (0.0                    ),
    cutoffType   (CutoffFunction::CT_HARD)
{
    // Add cutoff-related parameter IDs to set.
    parameters.insert("cutoffType");
    parameters.insert("cutoffAlpha");
}
