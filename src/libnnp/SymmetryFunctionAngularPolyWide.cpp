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

#include "SymmetryFunctionAngularPolyWide.h"
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

SymmetryFunctionAngularPolyWide::
SymmetryFunctionAngularPolyWide(ElementMap const& elementMap) :
    SymmetryFunction(29, elementMap),
    e1           (0    ),
    e2           (0    ),
    angleLeft    (0    ),
    angleRight   (0.0  ),
    eta          (0.0  ),
    rs           (0.0  )
{
    minNeighbors = 2;
    parameters.insert("e1");
    parameters.insert("e2");
    parameters.insert("eta");
    parameters.insert("angleLeft");
    parameters.insert("angleRight");
    parameters.insert("rs");
}

bool SymmetryFunctionAngularPolyWide::
operator==(SymmetryFunction const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymmetryFunctionAngularPolyWide const& c =
        dynamic_cast<SymmetryFunctionAngularPolyWide const&>(rhs);
    if (cutoffType  != c.cutoffType ) return false;
    if (cutoffAlpha != c.cutoffAlpha) return false;
    if (rc          != c.rc         ) return false;
    if (eta         != c.eta        ) return false;
    if (angleLeft   != c.angleLeft  ) return false;
    if (angleRight  != c.angleRight ) return false;
    if (e1          != c.e1         ) return false;
    if (e2          != c.e2         ) return false;
    return true;
}

bool SymmetryFunctionAngularPolyWide::
operator<(SymmetryFunction const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymmetryFunctionAngularPolyWide const& c =
        dynamic_cast<SymmetryFunctionAngularPolyWide const&>(rhs);
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
    if      (angleLeft   < c.angleLeft  ) return true;
    else if (angleLeft   > c.angleLeft  ) return false;
    if      (angleRight  < c.angleRight ) return true;
    else if (angleRight  > c.angleRight ) return false;
    if      (e1          < c.e1         ) return true;
    else if (e1          > c.e1         ) return false;
    if      (e2          < c.e2         ) return true;
    else if (e2          > c.e2         ) return false;
    return false;
}

void SymmetryFunctionAngularPolyWide::
setParameters(string const& parameterString)
{
    vector<string> splitLine = split(reduce(parameterString));

    if (type != (size_t)atoi(splitLine.at(1).c_str()))
    {
        throw runtime_error("ERROR: Incorrect symmetry function type.\n");
    }

    ec         = elementMap[splitLine.at(0)];
    e1         = elementMap[splitLine.at(2)];
    e2         = elementMap[splitLine.at(3)];
    eta        = atof(splitLine.at(4).c_str());
    angleLeft  = atof(splitLine.at(5).c_str());
    angleRight = atof(splitLine.at(6).c_str());
    rc         = atof(splitLine.at(7).c_str());
    // Shift parameter is optional.
    if (splitLine.size() > 8)
    {
        rs = atof(splitLine.at(8).c_str());
    }

    fc.setCutoffRadius(rc);
    fc.setCutoffParameter(cutoffAlpha);

    if (e1 > e2)
    {
        size_t tmp = e1;
        e1         = e2;
        e2         = tmp;
    }

    if (angleLeft >= angleRight)
    {
        throw runtime_error("ERROR: Left angle boundary right of or equal to "
                            "right angle boundary.\n");
    }

    c.setType(CompactFunction::Type::POLY2);
    c.setLeftRight(angleLeft * M_PI / 180.0, angleRight * M_PI / 180.0);

    return;
}

void SymmetryFunctionAngularPolyWide::changeLengthUnit(double convLength)
{
    this->convLength = convLength;
    eta /= convLength * convLength;
    rc *= convLength;
    rs *= convLength;

    fc.setCutoffRadius(rc);
    fc.setCutoffParameter(cutoffAlpha);

    return;
}

string SymmetryFunctionAngularPolyWide::getSettingsLine() const
{
    string s = strpr("symfunction_short %2s %2zu %2s %2s %16.8E %16.8E "
                     "%16.8E %16.8E %16.8E\n",
                     elementMap[ec].c_str(),
                     type,
                     elementMap[e1].c_str(),
                     elementMap[e2].c_str(),
                     eta * convLength * convLength,
                     angleLeft,
                     angleRight,
                     rc / convLength,
                     rs / convLength);

    return s;
}

void SymmetryFunctionAngularPolyWide::calculate(Atom&      atom,
                                                bool const derivatives) const
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
        if ((e1 == nej || e2 == nej) && rij < rc)
        {
            // TODO double const r2ij   = rij * rij;

            // Calculate cutoff function and derivative.
#ifdef NOCFCACHE
            double pfcij;
            double pdfcij;
            fc.fdf(rij, pfcij, pdfcij);
#else
            // If cutoff radius matches with the one in the neighbor storage
            // we can use the previously calculated value.
            double& pfcij = nj.fc;
            double& pdfcij = nj.dfc;
            if (nj.cutoffType != cutoffType ||
                nj.rc != rc ||
                nj.cutoffAlpha != cutoffAlpha)
            {
                fc.fdf(rij, pfcij, pdfcij);
                nj.rc = rc;
                nj.cutoffType = cutoffType;
                nj.cutoffAlpha = cutoffAlpha;
            }
#endif
            for (size_t k = j + 1; k < numNeighbors; k++)
            {
                Atom::Neighbor& nk = atom.neighbors[k];
                size_t const nek = nk.element;
                if ((e1 == nej && e2 == nek) ||
                    (e2 == nej && e1 == nek))
                {
                    double const rik = nk.d;
                    if (rik < rc)
                    {
                    // Energy calculation.
#ifdef NOCFCACHE
                        double pfcik;
                        double pdfcik;
                        fc.fdf(rik, pfcik, pdfcik);
#else
                        double& pfcik = nk.fc;
                        double& pdfcik = nk.dfc;
                        if (nk.cutoffType != cutoffType ||
                            nk.rc != rc ||
                            nk.cutoffAlpha != cutoffAlpha)
                        {
                            fc.fdf(rik, pfcik, pdfcik);
                            nk.rc = rc;
                            nk.cutoffType = cutoffType;
                            nk.cutoffAlpha = cutoffAlpha;
                        }
#endif
                        Vec3D drij = nj.dr;
                        Vec3D drik = nk.dr;
                        Vec3D drjk = drik - drij;
                        double costijk = drij * drik;
                        double rinvijik = 1.0 / rij / rik;
                        costijk *= rinvijik;
                        // Regroup later: Get acos(cos)
                        double const acostijk = acos(costijk);
                        // TODO checks for acos
                        double const pfc = pfcij * pfcik; // product of cutoff fcts
                        // TODO double const r2ik = rik * rik;
                        double const rijs = rij - rs;
                        double const riks = rik - rs;
                        double const pexp = exp(-eta * (rijs * rijs
                                                      + riks * riks));

                        // TODO: Carefully check conditions, can we do this?
                        double const poly = c.f(acostijk);
                        double       fg   = pexp;
                        // Should be able to leave that condition in
                        fg *= poly;
                        double const fgp = fg*pfc; // fgp = BCD

                        result += fgp;

                        // Force calculation.
                        if (!derivatives) continue;

                        double const dacostijk = -1.0 / sqrt(1.0 - costijk*costijk);
                        double dpoly = c.df(acostijk);
                        dpoly *= dacostijk;
                        dpoly /= poly; // Keep B out of the whole equation (keep fg out)
                        // dpoly now contains everything but \nabla costijk
                        // dpoly = B'' from derviation

                        double const rinvij  = rinvijik * rik;
                        double const rinvik  = rinvijik * rij;
                        double phiijik = rinvij * (rinvik - rinvij*costijk);
                        double phiikij = rinvik * (rinvij - rinvik*costijk);
                        double psiijik = rinvijik; // careful: sign flip w.r.t. notes due to nj.dGd...
                        phiijik *= dpoly;
                        phiikij *= dpoly;
                        psiijik *= dpoly;

                        double const p2eta = 2.0*eta;
                        double const chiij = rinvij / pfcij * pdfcij;
                        double const chiik = rinvik / pfcik * pdfcik;

                        // rijs/rij due to the shifted radial part of the Gaussian                        
                        double const p1 = fgp * (  phiijik - p2eta*rijs*rinvij + chiij );
                        double const p2 = fgp * (  phiikij - p2eta*riks*rinvik + chiik );
                        double const p3 = fgp * psiijik;
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

string SymmetryFunctionAngularPolyWide::parameterLine() const
{
    return strpr(getPrintFormat().c_str(),
                 index + 1,
                 elementMap[ec].c_str(),
                 type,
                 elementMap[e1].c_str(),
                 elementMap[e2].c_str(),
                 eta * convLength * convLength,
                 rs / convLength,
                 angleLeft,
                 angleRight,
                 rc / convLength,
                 (int)cutoffType,
                 cutoffAlpha,
                 lineNumber + 1);
}

vector<string> SymmetryFunctionAngularPolyWide::parameterInfo() const
{
    vector<string> v = SymmetryFunction::parameterInfo();
    string s;
    size_t w = sfinfoWidth;

    s = "e1";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), elementMap[e1].c_str()));
    s = "e2";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), elementMap[e2].c_str()));
    s = "eta";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(),
                      eta * convLength * convLength));
    s = "angleLeft";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), angleLeft));
    s = "angleRight";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), angleRight));
    s = "rs";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), rs / convLength));

    return v;
}

double SymmetryFunctionAngularPolyWide::calculateRadialPart(double distance) const
{
    double const& r = distance * convLength;
    double const p = exp(-eta * (r - rs) * (r -rs)) * fc.f(r);

    return p * p;
}

double SymmetryFunctionAngularPolyWide::calculateAngularPart(double angle) const
{
    return c.f(angle);
}

bool SymmetryFunctionAngularPolyWide::checkRelevantElement(size_t index) const
{
    if (index == e1 || index == e2) return true;
    else return false;
}
