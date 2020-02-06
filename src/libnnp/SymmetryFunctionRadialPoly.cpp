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

#include "SymmetryFunctionRadialPoly.h"
#include "Atom.h"
#include "ElementMap.h"
#include "utility.h"
#include "Vec3D.h"
#include <cstdlib>   // atof, atoi
#include <cmath>     // exp
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

SymmetryFunctionRadialPoly::SymmetryFunctionRadialPoly(ElementMap const& elementMap) :
    SymmetryFunction(28, elementMap),
    e1              (0  ),
    rl              (0.0)
{
    minNeighbors = 1;
    parameters.insert("e1");
    parameters.insert("rl");
}

bool SymmetryFunctionRadialPoly::operator==(SymmetryFunction const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymmetryFunctionRadialPoly const& c =
        dynamic_cast<SymmetryFunctionRadialPoly const&>(rhs);
    if (rl          != c.rl         ) return false;
    if (rc          != c.rc         ) return false;
    if (e1          != c.e1         ) return false;
    return true;
}

bool SymmetryFunctionRadialPoly::operator<(SymmetryFunction const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymmetryFunctionRadialPoly const& c =
        dynamic_cast<SymmetryFunctionRadialPoly const&>(rhs);
    // TODO: We don't need these! if      (cutoffType  < c.cutoffType ) return true;
    // TODO: We don't need these! else if (cutoffType  > c.cutoffType ) return false;
    // TODO: We don't need these! if      (cutoffAlpha < c.cutoffAlpha) return true;
    // TODO: We don't need these! else if (cutoffAlpha > c.cutoffAlpha) return false;
    if      (rl          < c.rl         ) return true;
    else if (rl          > c.rl         ) return false;
    if      (rc          < c.rc         ) return true;
    else if (rc          > c.rc         ) return false;
    if      (e1          < c.e1         ) return true;
    else if (e1          > c.e1         ) return false;
    return false;
}

void SymmetryFunctionRadialPoly::setParameters(string const& parameterString)
{
    vector<string> splitLine = split(reduce(parameterString));

    if (type != (size_t)atoi(splitLine.at(1).c_str()))
    {
        throw runtime_error("ERROR: Incorrect symmetry function type.\n");
    }

    ec   = elementMap[splitLine.at(0)];
    e1   = elementMap[splitLine.at(2)];
    rl   = atof(splitLine.at(3).c_str());
    rc   = atof(splitLine.at(4).c_str());

    if (rl > rc)
    {
        throw runtime_error("ERROR: Lower radial boundary >= upper radial boundary.\n");
    }
    else if (rl < 0.0 && (rl + rc) != 0.0)
    {
        throw runtime_error("ERROR: Radial function not symmetric w.r.t origin.\n");
    }
   
    c.setType(CompactFunction::Type::POLY2);
    c.setLeftRight(rl,rc);

    // TODO: Get rid of fc at all, or replace it by its compact counterpart
    // fc.setCutoffRadius(rc);
    // fc.setCutoffParameter(cutoffAlpha);
    // TODO: For now, just make sure we never use FC!

    return;
}

void SymmetryFunctionRadialPoly::changeLengthUnit(double convLength)
{
    this->convLength = convLength;
    rl *= convLength;
    rc *= convLength;
   
    c.setType(CompactFunction::Type::POLY2);
    c.setLeftRight(rl,rc);

    return;
}

string SymmetryFunctionRadialPoly::getSettingsLine() const
{
    string s = strpr("symfunction_short %2s %2zu %2s %16.8E %16.8E\n",
                     elementMap[ec].c_str(),
                     type,
                     elementMap[e1].c_str(),
                     rl / convLength,
                     rc / convLength);

    return s;
}

bool SymmetryFunctionRadialPoly::getCompactOnly(double x, double& fx, double& dfx) const
{
    bool const stat = c.fdf(x, fx, dfx);
    return stat;
}

void SymmetryFunctionRadialPoly::calculate(Atom&  atom,
                                       bool const derivatives) const
{
    double result = 0.0;

    for (size_t j = 0; j < atom.numNeighbors; ++j)
    {
        Atom::Neighbor& n = atom.neighbors[j];
        if (e1 == n.element && n.d < rc)
        {
            // Energy calculation.
            double const rij = n.d;

            double rad;
            double drad;
            c.fdf(rij, rad, drad);
            result += rad;

            // Force calculation.
            if (!derivatives) continue;
            double const p1 = scalingFactor * drad / rij;
            Vec3D dij = p1 * n.dr;
            // Save force contributions in Atom storage.
            atom.dGdr[index] += dij;
#ifdef IMPROVED_SFD_MEMORY
            n.dGdr[indexPerElement[e1]] -= dij;
#else
            n.dGdr[index] -= dij;
#endif
        }
    }

    atom.G[index] = scale(result);

    return;
}

string SymmetryFunctionRadialPoly::parameterLine() const
{
    return strpr(getPrintFormat().c_str(),
                 index + 1,
                 elementMap[ec].c_str(),
                 type,
                 elementMap[e1].c_str(),
                 rl / convLength,
                 rc / convLength,
                 // TODO (int)cutoffType,
                 // TODO cutoffAlpha,
                 lineNumber + 1);
}

vector<string> SymmetryFunctionRadialPoly::parameterInfo() const
{
    vector<string> v = SymmetryFunction::parameterInfo();
    string s;
    size_t w = sfinfoWidth;

    s = "e1";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), elementMap[e1].c_str()));
    s = "rl";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), rl / convLength));
    s = "rc";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), rc / convLength));

    return v;
}

double SymmetryFunctionRadialPoly::calculateRadialPart(double distance) const
{
    double const& r = distance * convLength;

    return c.f(r);
}

double SymmetryFunctionRadialPoly::calculateAngularPart(double /* angle */) const
{
    return 1.0;
}

bool SymmetryFunctionRadialPoly::checkRelevantElement(size_t index) const
{
    if (index == e1) return true;
    else return false;
}
