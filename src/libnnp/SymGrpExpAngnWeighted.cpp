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

#include "SymGrpExpAngnWeighted.h"
#include "Atom.h"
#include "SymFnc.h"
#include "SymFncExpAngnWeighted.h"
#include "Vec3D.h"
#include "utility.h"
#include <algorithm> // std::sort
#include <cmath>     // exp
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

SymGrpExpAngnWeighted::SymGrpExpAngnWeighted(ElementMap const& elementMap) :
    SymGrpBaseCutoff(13, elementMap)
{
    parametersMember.insert("eta");
    parametersMember.insert("rs/rl");
    parametersMember.insert("lambda");
    parametersMember.insert("zeta");
    parametersMember.insert("mindex");
    parametersMember.insert("sfindex");
    parametersMember.insert("calcexp");
}

bool SymGrpExpAngnWeighted::operator==(SymGrp const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymGrpExpAngnWeighted const& c =
        dynamic_cast<SymGrpExpAngnWeighted const&>(rhs);
    if (cutoffType  != c.cutoffType ) return false;
    if (cutoffAlpha != c.cutoffAlpha) return false;
    if (rc          != c.rc         ) return false;
    return true;
}

bool SymGrpExpAngnWeighted::operator<(SymGrp const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymGrpExpAngnWeighted const& c =
        dynamic_cast<SymGrpExpAngnWeighted const&>(rhs);
    if      (cutoffType  < c.cutoffType ) return true;
    else if (cutoffType  > c.cutoffType ) return false;
    if      (cutoffAlpha < c.cutoffAlpha) return true;
    else if (cutoffAlpha > c.cutoffAlpha) return false;
    if      (rc          < c.rc         ) return true;
    else if (rc          > c.rc         ) return false;
    return false;
}

bool SymGrpExpAngnWeighted::addMember(SymFnc const* const symmetryFunction)
{
    if (symmetryFunction->getType() != type) return false;

    SymFncExpAngnWeighted const* sf =
        dynamic_cast<SymFncExpAngnWeighted const*>(symmetryFunction);

    if (members.empty())
    {
        cutoffType  = sf->getCutoffType();
        subtype     = sf->getSubtype();
        cutoffAlpha = sf->getCutoffAlpha();
        ec          = sf->getEc();
        rc          = sf->getRc();
        convLength  = sf->getConvLength();

        fc.setCutoffType(cutoffType);
        fc.setCutoffRadius(rc);
        fc.setCutoffParameter(cutoffAlpha);
    }

    if (sf->getCutoffType()  != cutoffType ) return false;
    if (sf->getCutoffAlpha() != cutoffAlpha) return false;
    if (sf->getEc()          != ec         ) return false;
    if (sf->getRc()          != rc         ) return false;
    if (sf->getConvLength()  != convLength )
    {
        throw runtime_error("ERROR: Unable to add symmetry function members "
                            "with different conversion factors.\n");
    }

    members.push_back(sf);

    return true;
}

void SymGrpExpAngnWeighted::sortMembers()
{
    sort(members.begin(),
         members.end(),
         comparePointerTargets<SymFncExpAngnWeighted const>);

    // Members are now sorted with eta changing the slowest.
    for (size_t i = 0; i < members.size(); i++)
    {
        factorNorm.push_back(pow(2.0, 1.0 - members[i]->getZeta()));
        factorDeriv.push_back(2.0 * members[i]->getEta() /
                              members[i]->getZeta() / members[i]->getLambda());
        if (i == 0)
        {
            calculateExp.push_back(true);
        }
        else
        {
            if ( members[i - 1]->getEta() != members[i]->getEta() ||
                 members[i - 1]->getRs()  != members[i]->getRs() )
            {
                calculateExp.push_back(true);
            }
            else
            {
                calculateExp.push_back(false);
            }
        }
        useIntegerPow.push_back(members[i]->getUseIntegerPow());
        memberIndex.push_back(members[i]->getIndex());
        zetaInt.push_back(members[i]->getZetaInt());
        eta.push_back(members[i]->getEta());
        rs.push_back(members[i]->getRs());
        lambda.push_back(members[i]->getLambda());
        zeta.push_back(members[i]->getZeta());
        zetaLambda.push_back(members[i]->getZeta() * members[i]->getLambda());
        memberIndexPerElement.push_back(members[i]->getIndexPerElement());
    }

    return;
}

void SymGrpExpAngnWeighted::setScalingFactors()
{
    scalingFactors.resize(members.size(), 0.0);
    for (size_t i = 0; i < members.size(); i++)
    {
        scalingFactors.at(i) = members[i]->getScalingFactor();
        factorNorm.at(i) *= scalingFactors.at(i);
    }

    return;
}

// Depending on chosen symmetry functions this function may be very
// time-critical when predicting new structures (e.g. in MD simulations). Thus,
// lots of optimizations were used sacrificing some readablity. Vec3D
// operations have been rewritten in simple C array style and the use of
// temporary objects has been minimized. Some of the originally coded
// expressions are kept in comments marked with "SIMPLE EXPRESSIONS:".
void SymGrpExpAngnWeighted::calculate(Atom& atom, bool const derivatives) const
{
#ifndef NNP_NO_SF_CACHE
    // Can use cache indices of any member because this group is defined via
    // identical symmetry function type and cutoff functions.
    auto cacheIndices = members.at(0)->getCacheIndices();
#endif
    double* result = new double[members.size()];
    for (size_t l = 0; l < members.size(); ++l)
    {
        result[l] = 0.0;
    }

    double const rc2 = rc * rc;
    size_t numNeighbors = atom.numNeighbors;
    // Prevent problematic condition in loop test below (j < numNeighbors - 1).
    if (numNeighbors == 0) numNeighbors = 1;

    for (size_t j = 0; j < numNeighbors - 1; j++)
    {
        Atom::Neighbor& nj = atom.neighbors[j];
        size_t const nej = nj.element;
        double const rij = nj.d;
        if (rij < rc)
        {
            double const r2ij = rij * rij;

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
            // SIMPLE EXPRESSIONS:
            //Vec3D const drij(atom.neighbors[j].dr);
            //double const* const dr1 = drij.r;
            double const* const dr1 = nj.dr.r;

            for (size_t k = j + 1; k < numNeighbors; k++)
            {
                Atom::Neighbor& nk = atom.neighbors[k];
                size_t const nek = nk.element;
                double const rik = nk.d;
                if (rik < rc)
                {
                    // SIMPLE EXPRESSIONS:
                    //Vec3D const drjk(atom.neighbors[k].dr
                    //               - atom.neighbors[j].dr);
                    //double rjk = drjk.norm2();
                    double const* const dr2 = nk.dr.r;
                    double dr3[3];
                    dr3[0] = dr2[0] - dr1[0];
                    dr3[1] = dr2[1] - dr1[1];
                    dr3[2] = dr2[2] - dr1[2];
                    double rjk = dr3[0] * dr3[0]
                               + dr3[1] * dr3[1]
                               + dr3[2] * dr3[2];
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

                        // SIMPLE EXPRESSIONS:
                        //Vec3D const drik(atom.neighbors[k].dr);
                        //double const* const dr2 = drik.r;
                        //double const* const dr3 = drjk.r;
                        double const rinvijik = 1.0 / rij / rik;
                        // SIMPLE EXPRESSIONS:
                        //double const costijk = (drij * drik) * rinvijik;
                        double const costijk = (dr1[0] * dr2[0] +
                                                dr1[1] * dr2[1] +
                                                dr1[2] * dr2[2]) * rinvijik;
                        double const z = elementMap.atomicNumber(nej)
                                       * elementMap.atomicNumber(nek);
                        double const pfc = z * pfcij * pfcik * pfcjk;
                        double const r2ik = rik * rik;
                        double const pr1 = z * pfcik * pfcjk * pdfcij / rij;
                        double const pr2 = z * pfcij * pfcjk * pdfcik / rik;
                        double const pr3 = z * pfcij * pfcik * pdfcjk / rjk;
                        double vexp = 0.0;
                        double rijs = 0.0;
                        double riks = 0.0;
                        double rjks = 0.0;

                        for (size_t l = 0; l < members.size(); ++l)
                        {
                            if (calculateExp[l])
                            {
                                rijs = rij - rs[l];
                                riks = rik - rs[l];
                                rjks = rjk - rs[l];
                                double const r2sum = rijs * rijs
                                                   + riks * riks
                                                   + rjks * rjks;
                                vexp = exp(-eta[l] * r2sum);
                            }
                            double const plambda = 1.0
                                                 + lambda[l] * costijk;
                            double fg = vexp;
                            if (plambda <= 0.0) fg = 0.0;
                            else
                            {
                                if (useIntegerPow[l])
                                {
                                    fg *= pow_int(plambda, zetaInt[l] - 1);
                                }
                                else
                                {
                                    fg *= pow(plambda, zeta[l] - 1.0);
                                }
                            }
                            result[l] += fg * plambda * pfc;

                            // Force calculation.
                            if (!derivatives) continue;
                            fg *= factorNorm[l];
                            double const pfczl = pfc * zetaLambda[l];
                            double const p2etapl = plambda * factorDeriv[l];
                            double const p1 = fg * (pfczl * (rinvijik - costijk
                                            / r2ij - p2etapl * rijs / rij)
                                            + pr1 * plambda);
                            double const p2 = fg * (pfczl * (rinvijik - costijk
                                            / r2ik - p2etapl * riks / rik)
                                            + pr2 * plambda);
                            double const p3 = fg * (pfczl * (rinvijik + p2etapl
                                            * rjks / rjk) - pr3 * plambda);

                            // Save force contributions in Atom storage.
                            //
                            // SIMPLE EXPRESSIONS:
                            //size_t const li = members[l]->getIndex();
                            //atom.dGdr[li] += p1 * drij + p2 * drik;
                            //atom.neighbors[j].dGdr[li] -= p1 * drij
                            //                            + p3 * drjk;
                            //atom.neighbors[k].dGdr[li] -= p2 * drik
                            //                            - p3 * drjk;

                            double const p1drijx = p1 * dr1[0];
                            double const p1drijy = p1 * dr1[1];
                            double const p1drijz = p1 * dr1[2];

                            double const p2drikx = p2 * dr2[0];
                            double const p2driky = p2 * dr2[1];
                            double const p2drikz = p2 * dr2[2];

                            double const p3drjkx = p3 * dr3[0];
                            double const p3drjky = p3 * dr3[1];
                            double const p3drjkz = p3 * dr3[2];

#ifndef NNP_FULL_SFD_MEMORY
                            size_t li = memberIndex[l];
#else
                            size_t const li = memberIndex[l];
#endif
                            double* dGdr = atom.dGdr[li].r;
                            dGdr[0] += p1drijx + p2drikx;
                            dGdr[1] += p1drijy + p2driky;
                            dGdr[2] += p1drijz + p2drikz;

#ifndef NNP_FULL_SFD_MEMORY
                            li = memberIndexPerElement[l][nej];
#endif
                            dGdr = nj.dGdr[li].r;
                            dGdr[0] -= p1drijx + p3drjkx;
                            dGdr[1] -= p1drijy + p3drjky;
                            dGdr[2] -= p1drijz + p3drjkz;

#ifndef NNP_FULL_SFD_MEMORY
                            li = memberIndexPerElement[l][nek];
#endif
                            dGdr = nk.dGdr[li].r;
                            dGdr[0] -= p2drikx - p3drjkx;
                            dGdr[1] -= p2driky - p3drjky;
                            dGdr[2] -= p2drikz - p3drjkz;
                        } // l
                    } // rjk <= rc
                } // rik <= rc
            } // k
        } // rij <= rc
    } // j

    for (size_t l = 0; l < members.size(); ++l)
    {
        result[l] *= factorNorm[l] / scalingFactors[l];
        atom.G[memberIndex[l]] = members[l]->scale(result[l]);
    }

    delete[] result;

    return;
}

vector<string> SymGrpExpAngnWeighted::parameterLines() const
{
    vector<string> v;

    v.push_back(strpr(getPrintFormatCommon().c_str(),
                      index + 1,
                      elementMap[ec].c_str(),
                      type,
                      subtype.c_str(),
                      rc / convLength,
                      cutoffAlpha));

    for (size_t i = 0; i < members.size(); ++i)
    {
        v.push_back(strpr(getPrintFormatMember().c_str(),
                          members[i]->getEta() * convLength * convLength,
                          members[i]->getRs() / convLength,
                          members[i]->getLambda(),
                          members[i]->getZeta(),
                          members[i]->getLineNumber() + 1,
                          i + 1,
                          members[i]->getIndex() + 1,
                          (int)calculateExp[i]));
    }

    return v;
}
