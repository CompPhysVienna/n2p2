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

#include "SymFncAngwComp.h"
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

SymFncAngwComp::
SymFncAngwComp(ElementMap const& elementMap) :
    SymFncBaseComp(22, elementMap),
    e1           (0  ),
    e2           (0  ),
    angleLeft    (0.0),
    angleRight   (0.0)
{
    minNeighbors = 2;
    parameters.insert("e1");
    parameters.insert("e2");
    parameters.insert("angleLeft");
    parameters.insert("angleRight");
}

bool SymFncAngwComp::operator==(SymFnc const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymFncAngwComp const& c = dynamic_cast<SymFncAngwComp const&>(rhs);
    if (subtype     != c.getSubtype()) return false;
    if (e1          != c.e1          ) return false;
    if (e2          != c.e2          ) return false;
    if (rc          != c.rc          ) return false;
    if (rl          != c.rl          ) return false;
    if (angleLeft   != c.angleLeft   ) return false;
    if (angleRight  != c.angleRight  ) return false;
    return true;
}

bool SymFncAngwComp::operator<(SymFnc const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymFncAngwComp const& c = dynamic_cast<SymFncAngwComp const&>(rhs);
    if      (subtype     < c.getSubtype()) return true;
    else if (subtype     > c.getSubtype()) return false;
    if      (e1          < c.e1          ) return true;
    else if (e1          > c.e1          ) return false;
    if      (e2          < c.e2          ) return true;
    else if (e2          > c.e2          ) return false;
    if      (rc          < c.rc          ) return true;
    else if (rc          > c.rc          ) return false;
    if      (rl          < c.rl          ) return true;
    else if (rl          > c.rl          ) return false;
    if      (angleLeft   < c.angleLeft   ) return true;
    else if (angleLeft   > c.angleLeft   ) return false;
    if      (angleRight  < c.angleRight  ) return true;
    else if (angleRight  > c.angleRight  ) return false;
    return false;
}

void SymFncAngwComp::setParameters(string const& parameterString)
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
    ca.setLeftRight(angleLeft * M_PI / 180.0, angleRight * M_PI / 180.0);

    return;
}

void SymFncAngwComp::changeLengthUnit(double convLength)
{
    this->convLength = convLength;
    rc *= convLength;
    rl *= convLength;

    cr.setLeftRight(rl,rc);

    return;
}

string SymFncAngwComp::getSettingsLine() const
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

bool SymFncAngwComp::getCompactAngle(double x, double& fx, double& dfx) const
{
    bool const stat = ca.fdf(x, fx, dfx);
    return stat;
}

bool SymFncAngwComp::getCompactRadial(double x, double& fx, double& dfx) const
{
    bool const stat = cr.fdf(x, fx, dfx);
    return stat;
}

void SymFncAngwComp::calculate(Atom& atom, bool const derivatives) const
{
    // TODO double const pnorm  = pow(2.0, 1.0 - zeta);
    // TODO double const pzl    = zeta * lambda;
    double       result = 0.0;

    size_t numNeighbors = atom.numNeighbors;
    // Prevent problematic condition in loop test below (j < numNeighbors - 1).
    if (numNeighbors == 0) numNeighbors = 1;

    for (size_t j = 0; j < numNeighbors - 1; j++)
    {
        Atom::Neighbor& nj = atom.neighbors[j];
        size_t const nej = nj.element;
        double const rij = nj.d;
        if ((e1 == nej || e2 == nej) && rij < rc && rij > rl)
        {

            // Is one part of the product == zero?
            double radij;
            double dradij;
            if (!cr.fdf(rij, radij, dradij)) continue;

            for (size_t k = j + 1; k < numNeighbors; k++)
            {
                Atom::Neighbor& nk = atom.neighbors[k];
                size_t const nek = nk.element;
                if ((e1 == nej && e2 == nek) ||
                    (e2 == nej && e1 == nek))
                {
                    double const rik = nk.d;
                    if (rik < rc && rik > rl)
                    {
                    // Energy calculation.

                        double radik;
                        double dradik;
                        if (!cr.fdf(rik, radik, dradik)) continue;

                        Vec3D drij = nj.dr;
                        Vec3D drik = nk.dr;
                        Vec3D drjk = drik - drij;
                        double costijk = drij * drik;
                        double rinvijik = 1.0 / rij / rik;
                        costijk *= rinvijik;

                        // By definition, our polynomial is zero at 0 and 180 deg.
                        // Therefore, skip the whole rest which might yield some NaN
                        if (costijk <= -1.0 || costijk >= 1.0) continue;
 
                        // Regroup later: Get acos(cos)
                        double const acostijk = acos(costijk);
                        // Only go on if we are within our compact support
                        double ang  = 0.0;
                        double dang = 0.0;
                        if (!ca.fdf(acostijk, ang, dang)) continue;

                        double const rad  = radij * radik; // product of cutoff fcts
                        result += rad*ang;

                        // Force calculation.
                        if (!derivatives) continue;

                        double const dacostijk = -1.0 / sqrt(1.0 - costijk*costijk);
                        dang *= dacostijk;

                        double const rinvij  = rinvijik * rik;
                        double const rinvik  = rinvijik * rij;
                        double phiijik = rinvij * (rinvik - rinvij*costijk);
                        double phiikij = rinvik * (rinvij - rinvik*costijk);
                        double psiijik = rinvijik; // careful: sign flip w.r.t. notes due to nj.dGd...
                        phiijik *= dang;
                        phiikij *= dang;
                        psiijik *= dang;

                        // Cutoff function might be a divide by zero, but we screen that case before
                        double const chiij = rinvij * radik * dradij;
                        double const chiik = rinvik * radij * dradik;

                        // rijs/rij due to the shifted radial part of the Gaussian                        
                        double const p1 = rad * phiijik +  ang * chiij;
                        double const p2 = rad * phiikij +  ang * chiik;
                        double const p3 = rad * psiijik;
                        drij *= p1 * scalingFactor;
                        drik *= p2 * scalingFactor;
                        drjk *= p3 * scalingFactor;

                        // Save force contributions in Atom storage.
                        atom.dGdr[index] += drij + drik;
#ifdef IMPROVED_SFD_MEMORY
                        nj.dGdr[indexPerElement[nej]] -= drij + drjk;
                        nk.dGdr[indexPerElement[nek]] -= drik - drjk;
#else
                        nj.dGdr[index] -= drij + drjk;
                        nk.dGdr[index] -= drik - drjk;
#endif
                    } // rik <= rc
                } // elem
            } // k
        } // rij <= rc
    } // j

    // TODO result *= pnorm;
    // supposed to disappear

    atom.G[index] = scale(result);

    return;
}

string SymFncAngwComp::parameterLine() const
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

vector<string> SymFncAngwComp::parameterInfo() const
{
    vector<string> v = SymFnc::parameterInfo();
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

double SymFncAngwComp::calculateRadialPart(double distance) const
{
    double const& r = distance * convLength;

    return cr.f(r);
}

double SymFncAngwComp::calculateAngularPart(double angle) const
{
    return ca.f(angle);
}

bool SymFncAngwComp::checkRelevantElement(size_t index) const
{
    if (index == e1 || index == e2) return true;
    else return false;
}
