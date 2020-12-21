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

#include "SymFncExpAngn.h"
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

SymFncExpAngn::SymFncExpAngn(ElementMap const& elementMap) :
    SymFncBaseExpAng(3, elementMap)
{
}

bool SymFncExpAngn::operator==(SymFnc const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymFncExpAngn const& c = dynamic_cast<SymFncExpAngn const&>(rhs);
    if (cutoffType  != c.cutoffType ) return false;
    if (cutoffAlpha != c.cutoffAlpha) return false;
    if (rc          != c.rc         ) return false;
    if (eta         != c.eta        ) return false;
    if (zeta        != c.zeta       ) return false;
    if (lambda      != c.lambda     ) return false;
    if (e1          != c.e1         ) return false;
    if (e2          != c.e2         ) return false;
    return true;
}

bool SymFncExpAngn::operator<(SymFnc const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymFncExpAngn const& c = dynamic_cast<SymFncExpAngn const&>(rhs);
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
    if      (zeta        < c.zeta       ) return true;
    else if (zeta        > c.zeta       ) return false;
    if      (lambda      < c.lambda     ) return true;
    else if (lambda      > c.lambda     ) return false;
    if      (e1          < c.e1         ) return true;
    else if (e1          > c.e1         ) return false;
    if      (e2          < c.e2         ) return true;
    else if (e2          > c.e2         ) return false;
    return false;
}

void SymFncExpAngn::calculate(Atom& atom, bool const derivatives) const
{
    double const pnorm  = pow(2.0, 1.0 - zeta);
    double const pzl    = zeta * lambda;
    double const rc2    = rc * rc;
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
            double const r2ij   = rij * rij;

            // Calculate cutoff function and derivative.
            double pfcij;
            double pdfcij;
#ifndef NNP_NO_SF_CACHE
            if (cacheIndices[nej].size() == 0) fc.fdf(rij, pfcij, pdfcij);
            else
            {
                double& cfc = nj.cache[cacheIndices[nej][0]];
                double& cdfc = nj.cache[cacheIndices[nej][1]];
                if (cfc < 0) fc.fdf(rij, cfc, cdfc);
                pfcij = cfc;
                pdfcij = cdfc;
            }
#else
            fc.fdf(rij, pfcij, pdfcij);
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
                        Vec3D drjk = nk.dr - nj.dr;
                        double rjk = drjk.norm2();;
                        if (rjk < rc2)
                        {
                            // Energy calculation.
                            double pfcik;
                            double pdfcik;
#ifndef NNP_NO_SF_CACHE
                            if (cacheIndices[nej].size() == 0)
                            {
                                fc.fdf(rik, pfcik, pdfcik);
                            }
                            else
                            {
                                double& cfc = nk.cache[cacheIndices[nek][0]];
                                double& cdfc = nk.cache[cacheIndices[nek][1]];
                                if (cfc < 0) fc.fdf(rik, cfc, cdfc);
                                pfcik = cfc;
                                pdfcik = cdfc;
                            }
#else
                            fc.fdf(rik, pfcik, pdfcik);
#endif
                            rjk = sqrt(rjk);
                            double pfcjk;
                            double pdfcjk;
                            fc.fdf(rjk, pfcjk, pdfcjk);

                            Vec3D drij = nj.dr;
                            Vec3D drik = nk.dr;
                            double costijk = drij * drik;;
                            double rinvijik = 1.0 / rij / rik;
                            costijk *= rinvijik;

                            double const pfc = pfcij * pfcik * pfcjk;
                            double const r2ik = rik * rik;
                            double const rijs = rij - rs;
                            double const riks = rik - rs;
                            double const rjks = rjk - rs;
                            double const pexp = exp(-eta * (rijs * rijs +
                                                            riks * riks +
                                                            rjks * rjks));
                            double const plambda = 1.0 + lambda * costijk;
                            double       fg = pexp;
                            if (plambda <= 0.0) fg = 0.0;
                            else
                            {
                                if (useIntegerPow)
                                {
                                    fg *= pow_int(plambda, zetaInt - 1);
                                }
                                else
                                {
                                    fg *= pow(plambda, zeta - 1.0);
                                }
                            }
                            result += fg * plambda * pfc;

                            // Force calculation.
                            if (!derivatives) continue;
                            fg       *= pnorm;
                            rinvijik *= pzl;
                            costijk  *= pzl;
                            double const p2etapl = 2.0 * eta * plambda;
                            double const p1 = fg * (pfc * (rinvijik - costijk
                                            / r2ij - p2etapl * rijs / rij)
                                            + pfcik * pfcjk * pdfcij * plambda
                                            / rij);
                            double const p2 = fg * (pfc * (rinvijik - costijk
                                            / r2ik - p2etapl * riks / rik)
                                            + pfcij * pfcjk * pdfcik * plambda
                                            / rik);
                            double const p3 = fg * (pfc * (rinvijik + p2etapl
                                            * rjks / rjk) - pfcij * pfcik
                                            * pdfcjk * plambda / rjk);
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
                        } // rjk <= rc
                    } // rik <= rc
                } // elem
            } // k
        } // rij <= rc
    } // j
    result *= pnorm;

    atom.G[index] = scale(result);

    return;
}
