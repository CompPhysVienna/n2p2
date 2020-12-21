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

#include "SymFncCompAngn.h"
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

SymFncCompAngn::
SymFncCompAngn(ElementMap const& elementMap) :
    SymFncBaseCompAng(21, elementMap)
{
}

bool SymFncCompAngn::operator==(SymFnc const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymFncCompAngn const& c = dynamic_cast<SymFncCompAngn const&>(rhs);
    if (subtype     != c.getSubtype()) return false;
    if (e1          != c.e1          ) return false;
    if (e2          != c.e2          ) return false;
    if (rc          != c.rc          ) return false;
    if (rl          != c.rl          ) return false;
    if (angleLeft   != c.angleLeft   ) return false;
    if (angleRight  != c.angleRight  ) return false;
    return true;
}

bool SymFncCompAngn::operator<(SymFnc const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymFncCompAngn const& c = dynamic_cast<SymFncCompAngn const&>(rhs);
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

void SymFncCompAngn::calculate(Atom& atom, bool const derivatives) const
{
    double r2l = 0.0;
    if (rl > 0.0) r2l = rl * rl;
    double r2c = rc * rc;
    double result = 0.0;

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
            double radij;
            double dradij;
#ifndef NNP_NO_SF_CACHE
            if (cacheIndices[nej].size() == 0) cr.fdf(rij, radij, dradij);
            else
            {
                double& crad = nj.cache[cacheIndices[nej][0]];
                double& cdrad = nj.cache[cacheIndices[nej][1]];
                if (crad < 0) cr.fdf(rij, crad, cdrad);
                radij = crad;
                dradij = cdrad;
            }
#else
            cr.fdf(rij, radij, dradij);
#endif
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
                        Vec3D drij = nj.dr;
                        Vec3D drik = nk.dr;
                        Vec3D drjk = nk.dr - nj.dr;
                        double rjk = drjk.norm2();
                        if (rjk >= r2c || rjk <= r2l) continue;
                        rjk = sqrt(rjk);
 
                        double radik;
                        double dradik;
#ifndef NNP_NO_SF_CACHE
                        if (cacheIndices[nek].size() == 0)
                        {
                            cr.fdf(rik, radik, dradik);
                        }
                        else
                        {
                            double& crad = nk.cache[cacheIndices[nek][0]];
                            double& cdrad = nk.cache[cacheIndices[nek][1]];
                            if (crad < 0) cr.fdf(rik, crad, cdrad);
                            radik = crad;
                            dradik = cdrad;
                        }
#else
                        cr.fdf(rik, radik, dradik);
#endif
                        double radjk;
                        double dradjk;
                        cr.fdf(rjk, radjk, dradjk);

                        double costijk = drij * drik;
                        double rinvijik = 1.0 / rij / rik;
                        costijk *= rinvijik;

                        // By definition, our polynomial is zero at 0 and 180 deg.
                        // Therefore, skip the whole rest which might yield some NaN
                        if (costijk <= -1.0 || costijk >= 1.0) continue;
 
                        // Regroup later: Get acos(cos)
                        double const acostijk = acos(costijk);
                        // Only go on if we are within our compact support
                        if (acostijk < angleLeftRadians ||
                            acostijk > angleRightRadians) continue;
                        double ang  = 0.0;
                        double dang = 0.0;
                        ca.fdf(acostijk, ang, dang);

                        double const rad  = radij * radik * radjk; // product of cutoff fcts
                        result += rad * ang;

                        // Force calculation.
                        if (!derivatives) continue;

                        double const dacostijk = -1.0
                                               / sqrt(1.0 - costijk * costijk);
                        dang *= dacostijk;
                        double const rinvij = rinvijik * rik;
                        double const rinvik = rinvijik * rij;
                        double const rinvjk = 1.0 / rjk;
                        double phiijik = rinvij * (rinvik - rinvij*costijk);
                        double phiikij = rinvik * (rinvij - rinvik*costijk);
                        double psiijik = rinvijik; // careful: sign flip w.r.t. notes due to nj.dGd...
                        phiijik *= dang;
                        phiikij *= dang;
                        psiijik *= dang;

                        // Cutoff function might be a divide by zero, but we screen that case before
                        double const chiij =  rinvij * dradij *  radik *  radjk;
                        double const chiik =  rinvik *  radij * dradik *  radjk;
                        double const chijk = -rinvjk *  radij *  radik * dradjk;

                        // rijs/rij due to the shifted radial part of the Gaussian                        
                        double const p1 = rad * phiijik +  ang * chiij;
                        double const p2 = rad * phiikij +  ang * chiik;
                        double const p3 = rad * psiijik +  ang * chijk;
                        drij *= p1 * scalingFactor;
                        drik *= p2 * scalingFactor;
                        drjk *= p3 * scalingFactor;

                        // Save force contributions in Atom storage.
                        atom.dGdr[index] += drij + drik;
#ifndef NNP_FULL_SFD_MEMORY
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

    atom.G[index] = scale(result);

    return;
}
