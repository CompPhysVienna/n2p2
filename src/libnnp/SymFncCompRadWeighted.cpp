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

#include "SymFncCompRadWeighted.h"
#include "Atom.h"
#include "ElementMap.h"
#include "utility.h"
#include "Vec3D.h"
#include <cstdlib>   // atof, atoi
#include <cmath>     // exp
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

SymFncCompRadWeighted::SymFncCompRadWeighted(ElementMap const& elementMap) :
    SymFncBaseComp(23, elementMap)
{
    minNeighbors = 1;
}

bool SymFncCompRadWeighted::operator==(SymFnc const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymFncCompRadWeighted const& c
        = dynamic_cast<SymFncCompRadWeighted const&>(rhs);
    if (subtype     != c.getSubtype()) return false;
    if (rc          != c.rc          ) return false;
    if (rl          != c.rl          ) return false;
    return true;
}

bool SymFncCompRadWeighted::operator<(SymFnc const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymFncCompRadWeighted const& c
        = dynamic_cast<SymFncCompRadWeighted const&>(rhs);
    if      (subtype     < c.getSubtype()) return true;
    else if (subtype     > c.getSubtype()) return false;
    if      (rc          < c.rc          ) return true;
    else if (rc          > c.rc          ) return false;
    if      (rl          < c.rl          ) return true;
    else if (rl          > c.rl          ) return false;
    return false;
}

void SymFncCompRadWeighted::setParameters(string const& parameterString)
{
    vector<string> splitLine = split(reduce(parameterString));

    if (type != (size_t)atoi(splitLine.at(1).c_str()))
    {
        throw runtime_error("ERROR: Incorrect symmetry function type.\n");
    }

    ec      = elementMap[splitLine.at(0)];
    rl      = atof(splitLine.at(2).c_str());
    rc      = atof(splitLine.at(3).c_str());
    subtype = splitLine.at(4);

    if (rl > rc)
    {
        throw runtime_error("ERROR: Lower radial boundary >= upper "
                            "radial boundary.\n");
    }
    //if (rl < 0.0 && abs(rl + rc) > numeric_limits<double>::epsilon())
    //{
    //    throw runtime_error("ERROR: Radial function not symmetric "
    //                        "w.r.t. origin.\n");
    //}

    setCompactFunction(subtype);
    cr.setLeftRight(rl,rc);

    return;
}

void SymFncCompRadWeighted::changeLengthUnit(double convLength)
{
    this->convLength = convLength;
    rl *= convLength;
    rc *= convLength;

    cr.setLeftRight(rl, rc);

    return;
}

string SymFncCompRadWeighted::getSettingsLine() const
{
    string s = strpr("symfunction_short %2s %2zu %16.8E %16.8E %s\n",
                     elementMap[ec].c_str(),
                     type,
                     rl / convLength,
                     rc / convLength,
                     subtype.c_str());

    return s;
}

void SymFncCompRadWeighted::calculate(Atom& atom, bool const derivatives) const
{
    double result = 0.0;

    for (size_t j = 0; j < atom.numNeighbors; ++j)
    {
        Atom::Neighbor& n = atom.neighbors[j];
        if (n.d > rl && n.d < rc)
        {
            // Energy calculation.
            size_t const ne = n.element;
            double const rij = n.d;

            double rad;
            double drad;
            double r = rij;
#ifndef NNP_NO_SF_CACHE
            if (cacheIndices[ne].size() == 0) cr.fdf(r, rad, drad);
            else
            {
                double& crad = n.cache[cacheIndices[ne][0]];
                double& cdrad = n.cache[cacheIndices[ne][1]];
                if (crad < 0) cr.fdf(r, crad, cdrad);
                rad = crad;
                drad = cdrad;
            }
#else
            cr.fdf(r, rad, drad);
#endif
            result += rad * elementMap.atomicNumber(ne);

            // Force calculation.
            if (!derivatives) continue;
            double const p1 = scalingFactor * elementMap.atomicNumber(ne)
                            * drad / rij;
            Vec3D dij = p1 * n.dr;
            // Save force contributions in Atom storage.
            atom.dGdr[index] += dij;
#ifndef NNP_FULL_SFD_MEMORY
            n.dGdr[indexPerElement[ne]] -= dij;
#else
            n.dGdr[index] -= dij;
#endif
        }
    }

    atom.G[index] = scale(result);

    return;
}

string SymFncCompRadWeighted::parameterLine() const
{
    return strpr(getPrintFormat().c_str(),
                 index + 1,
                 elementMap[ec].c_str(),
                 type,
                 subtype.c_str(),
                 rl / convLength,
                 rc / convLength,
                 lineNumber + 1);
}

vector<string> SymFncCompRadWeighted::parameterInfo() const
{
    vector<string> v = SymFncBaseComp::parameterInfo();

    return v;
}

double SymFncCompRadWeighted::calculateRadialPart(double distance) const
{
    double const& r = distance * convLength;

    return cr.f(r);
}

double SymFncCompRadWeighted::calculateAngularPart(double /* angle */) const
{
    return 1.0;
}

bool SymFncCompRadWeighted::checkRelevantElement(size_t /*index*/) const
{
    return true;
}

#ifndef NNP_NO_SF_CACHE
vector<string> SymFncCompRadWeighted::getCacheIdentifiers() const
{
    vector<string> v;
    string s("");

    s += subtype;
    s += " ";
    s += strpr("rl = %16.8E", rl / convLength);
    s += " ";
    s += strpr("rc = %16.8E", rc / convLength);

    for (size_t i = 0; i < elementMap.size(); ++i)
    {
        v.push_back(strpr("%zu f ", i) + s);
        v.push_back(strpr("%zu df ", i) + s);
    }

    return v;
}
#endif
