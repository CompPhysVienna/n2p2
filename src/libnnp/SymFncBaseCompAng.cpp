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

#include "SymFncBaseCompAng.h"
#include "Atom.h"
#include "ElementMap.h"
#include "utility.h"
#include "Vec3D.h"
#include <cstdlib>   // atof, atoi
#include <cmath>     // exp, pow, cos
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

SymFncBaseCompAng::
SymFncBaseCompAng(size_t type, ElementMap const& elementMap) :
    SymFncBaseComp(type, elementMap),
    e1               (0  ),
    e2               (0  ),
    angleLeft        (0.0),
    angleRight       (0.0),
    angleLeftRadians (0.0),
    angleRightRadians(0.0)
{
    minNeighbors = 2;
    parameters.insert("e1");
    parameters.insert("e2");
    parameters.insert("angleLeft");
    parameters.insert("angleRight");
}

void SymFncBaseCompAng::setParameters(string const& parameterString)
{
    vector<string> splitLine = split(reduce(parameterString));

    if (type != (size_t)atoi(splitLine.at(1).c_str()))
    {
        throw runtime_error("ERROR: Incorrect symmetry function type.\n");
    }

    ec         = elementMap[splitLine.at(0)];
    e1         = elementMap[splitLine.at(2)];
    e2         = elementMap[splitLine.at(3)];
    rl         = atof(splitLine.at(4).c_str());
    rc         = atof(splitLine.at(5).c_str());
    angleLeft  = atof(splitLine.at(6).c_str());
    angleRight = atof(splitLine.at(7).c_str());
    subtype    = splitLine.at(8);

    if (e1 > e2)
    {
        size_t tmp = e1;
        e1         = e2;
        e2         = tmp;
    }

    // Radial part.
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

    // Angular part.
    if (angleLeft >= angleRight)
    {
        throw runtime_error("ERROR: Left angle boundary right of or equal to "
                            "right angle boundary.\n");
    }

    double const center = (angleLeft + angleRight) / 2.0;
    if ( (angleLeft  <   0.0 && center != 0.0) ||
         (angleRight > 180.0 && center != 180.0) )
    {
        throw runtime_error("ERROR: Angle boundary out of [0,180] and "
                            "center of angular function /= 0 or /= 180.\n");
    }
    if (angleRight - angleLeft > 360.0)
    {
        throw runtime_error("ERROR: Periodic symmetry function cannot spread "
                            "over domain > 360 degrees\n");
    }

    ca.setCoreFunction(cr.getCoreFunctionType());
    angleLeftRadians = angleLeft * M_PI / 180.0;
    angleRightRadians = angleRight * M_PI / 180.0;
    ca.setLeftRight(angleLeftRadians, angleRightRadians);

    return;
}

void SymFncBaseCompAng::changeLengthUnit(double convLength)
{
    this->convLength = convLength;
    rc *= convLength;
    rl *= convLength;

    cr.setLeftRight(rl,rc);

    return;
}

string SymFncBaseCompAng::getSettingsLine() const
{
    string s = strpr("symfunction_short %2s %2zu %2s %2s %16.8E %16.8E "
                     "%16.8E %16.8E %s\n",
                     elementMap[ec].c_str(),
                     type,
                     elementMap[e1].c_str(),
                     elementMap[e2].c_str(),
                     rl / convLength,
                     rc / convLength,
                     angleLeft,
                     angleRight,
                     subtype.c_str());

    return s;
}

string SymFncBaseCompAng::parameterLine() const
{
    return strpr(getPrintFormat().c_str(),
                 index + 1,
                 elementMap[ec].c_str(),
                 type,
                 subtype.c_str(),
                 elementMap[e1].c_str(),
                 elementMap[e2].c_str(),
                 rl / convLength,
                 rc / convLength,
                 angleLeft,
                 angleRight,
                 lineNumber + 1);
}

vector<string> SymFncBaseCompAng::parameterInfo() const
{
    vector<string> v = SymFncBaseComp::parameterInfo();
    string s;
    size_t w = sfinfoWidth;

    s = "e1";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), elementMap[e1].c_str()));
    s = "e2";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), elementMap[e2].c_str()));
    s = "angleLeft";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), angleLeft));
    s = "angleRight";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), angleRight));

    return v;
}

double SymFncBaseCompAng::calculateRadialPart(double distance) const
{
    double const& r = distance * convLength;

    return cr.f(r);
}

double SymFncBaseCompAng::calculateAngularPart(double angle) const
{
    return ca.f(angle);
}

bool SymFncBaseCompAng::checkRelevantElement(size_t index) const
{
    if (index == e1 || index == e2) return true;
    else return false;
}

#ifndef NNP_NO_SF_CACHE
vector<string> SymFncBaseCompAng::getCacheIdentifiers() const
{
    vector<string> v;
    string s("");

    s += subtype;
    s += " ";
    s += strpr("rl = %16.8E", rl / convLength);
    s += " ";
    s += strpr("rc = %16.8E", rc / convLength);

    v.push_back(strpr("%zu f ", e1) + s);
    v.push_back(strpr("%zu df ", e1) + s);
    if (e1 != e2) v.push_back(strpr("%zu f ", e2) + s);
    if (e1 != e2) v.push_back(strpr("%zu df ", e2) + s);

    return v;
}
#endif
