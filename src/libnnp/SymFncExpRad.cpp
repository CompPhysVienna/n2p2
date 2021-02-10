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

#include "SymFncExpRad.h"
#include "Atom.h"
#include "ElementMap.h"
#include "utility.h"
#include "Vec3D.h"
#include <cstdlib>   // atof, atoi
#include <cmath>     // exp
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

SymFncExpRad::SymFncExpRad(ElementMap const& elementMap) :
    SymFncBaseCutoff(2, elementMap),
    e1              (0  ),
    eta             (0.0),
    rs              (0.0)
{
    minNeighbors = 1;
    parameters.insert("e1");
    parameters.insert("eta");
    parameters.insert("rs/rl");
}

bool SymFncExpRad::operator==(SymFnc const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymFncExpRad const& c = dynamic_cast<SymFncExpRad const&>(rhs);
    if (cutoffType  != c.cutoffType ) return false;
    if (cutoffAlpha != c.cutoffAlpha) return false;
    if (rc          != c.rc         ) return false;
    if (eta         != c.eta        ) return false;
    if (rs          != c.rs         ) return false;
    if (e1          != c.e1         ) return false;
    return true;
}

bool SymFncExpRad::operator<(SymFnc const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymFncExpRad const& c = dynamic_cast<SymFncExpRad const&>(rhs);
    if      (cutoffType  < c.cutoffType ) return true;
    else if (cutoffType  > c.cutoffType ) return false;
    if      (cutoffAlpha < c.cutoffAlpha) return true;
    else if (cutoffAlpha > c.cutoffAlpha) return false;
    if      (rc          < c.rc         ) return true;
    else if (rc          > c.rc         ) return false;
    if      (eta         < c.eta        ) return true;
    else if (eta         > c.eta        ) return false;
    if      (rs          < c.rs         ) return true;
    else if (rs          > c.rs         ) return false;
    if      (e1          < c.e1         ) return true;
    else if (e1          > c.e1         ) return false;
    return false;
}

void SymFncExpRad::setParameters(string const& parameterString)
{
    vector<string> splitLine = split(reduce(parameterString));

    if (type != (size_t)atoi(splitLine.at(1).c_str()))
    {
        throw runtime_error("ERROR: Incorrect symmetry function type.\n");
    }

    ec   = elementMap[splitLine.at(0)];
    e1   = elementMap[splitLine.at(2)];
    eta  = atof(splitLine.at(3).c_str());
    rs   = atof(splitLine.at(4).c_str());
    rc   = atof(splitLine.at(5).c_str());

    fc.setCutoffRadius(rc);
    fc.setCutoffParameter(cutoffAlpha);

    return;
}

void SymFncExpRad::changeLengthUnit(double convLength)
{
    this->convLength = convLength;
    eta /= convLength * convLength;
    rs *= convLength;
    rc *= convLength;

    fc.setCutoffRadius(rc);
    fc.setCutoffParameter(cutoffAlpha);

    return;
}

string SymFncExpRad::getSettingsLine() const
{
    string s = strpr("symfunction_short %2s %2zu %2s %16.8E %16.8E %16.8E\n",
                     elementMap[ec].c_str(),
                     type,
                     elementMap[e1].c_str(),
                     eta * convLength * convLength,
                     rs / convLength,
                     rc / convLength);

    return s;
}

void SymFncExpRad::calculate(Atom& atom, bool const derivatives) const
{
    double result = 0.0;
#ifndef NNP_NO_SF_CACHE
    bool unique = true;
    size_t c0 = 0;
    size_t c1 = 0;
    if (cacheIndices.at(e1).size() > 0)
    {
        unique = false;
        c0 = cacheIndices.at(e1).at(0);
        c1 = cacheIndices.at(e1).at(1);
    }
#endif

    for (size_t j = 0; j < atom.numNeighbors; ++j)
    {
        Atom::Neighbor& n = atom.neighbors[j];
        if (e1 == n.element && n.d < rc)
        {
            // Energy calculation.
            double const rij = n.d;
            double const pexp = exp(-eta * (rij - rs) * (rij - rs));

            // Calculate cutoff function and derivative.
            double pfc;
            double pdfc;
#ifndef NNP_NO_SF_CACHE
            if (unique) fc.fdf(rij, pfc, pdfc);
            else
            {
                double& cfc = n.cache[c0];
                double& cdfc = n.cache[c1];
                if (cfc < 0) fc.fdf(rij, cfc, cdfc);
                pfc = cfc;
                pdfc = cdfc;
            }
#else
            fc.fdf(rij, pfc, pdfc);
#endif
            result += pexp * pfc;
            // Force calculation.
            if (!derivatives) continue;
            double const p1 = scalingFactor
                * (pdfc - 2.0 * eta * (rij - rs)
                * pfc) * pexp / rij;
            Vec3D dij = p1 * n.dr;
            // Save force contributions in Atom storage.
            atom.dGdr[index] += dij;
#ifndef NNP_FULL_SFD_MEMORY
            n.dGdr[indexPerElement[e1]] -= dij;
#else
            n.dGdr[index] -= dij;
#endif
        }
    }

    atom.G[index] = scale(result);

    return;
}

string SymFncExpRad::parameterLine() const
{
    return strpr(getPrintFormat().c_str(),
                 index + 1,
                 elementMap[ec].c_str(),
                 type,
                 subtype.c_str(),
                 elementMap[e1].c_str(),
                 eta * convLength * convLength,
                 rs / convLength,
                 rc / convLength,
                 cutoffAlpha,
                 lineNumber + 1);
}

vector<string> SymFncExpRad::parameterInfo() const
{
    vector<string> v = SymFncBaseCutoff::parameterInfo();
    string s;
    size_t w = sfinfoWidth;

    s = "e1";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), elementMap[e1].c_str()));
    s = "eta";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(),
                      eta * convLength * convLength));
    s = "rs";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), rs / convLength));

    return v;
}

double SymFncExpRad::calculateRadialPart(double distance) const
{
    double const& r = distance * convLength;

    return exp(-eta * (r - rs) * (r - rs)) * fc.f(r);
}

double SymFncExpRad::calculateAngularPart(double /* angle */) const
{
    return 1.0;
}

bool SymFncExpRad::checkRelevantElement(size_t index) const
{
    if (index == e1) return true;
    else return false;
}

#ifndef NNP_NO_SF_CACHE
vector<string> SymFncExpRad::getCacheIdentifiers() const
{
    vector<string> v;
    string s("");

    s += subtype;
    s += " ";
    s += strpr("alpha = %16.8E", cutoffAlpha);
    s += " ";
    s += strpr("rc = %16.8E", rc / convLength);

    v.push_back(strpr("%zu f ", e1) + s);
    v.push_back(strpr("%zu df ", e1) + s);

    return v;
}
#endif
