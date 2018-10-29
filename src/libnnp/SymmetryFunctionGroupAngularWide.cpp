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

#include "SymmetryFunctionGroupAngularWide.h"
#include "Atom.h"
#include "SymmetryFunction.h"
#include "SymmetryFunctionAngularWide.h"
#include "Vec3D.h"
#include "utility.h"
#include <algorithm> // std::sort
#include <cmath>     // exp
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

SymmetryFunctionGroupAngularWide::
SymmetryFunctionGroupAngularWide(ElementMap const& elementMap) :
    SymmetryFunctionGroup(9, elementMap),
    e1(0),
    e2(0)
{
    parametersCommon.insert("e1");
    parametersCommon.insert("e2");

    parametersMember.insert("eta");
    parametersMember.insert("rs");
    parametersMember.insert("lambda");
    parametersMember.insert("zeta");
    parametersMember.insert("mindex");
    parametersMember.insert("sfindex");
    parametersMember.insert("calcexp");
}

bool SymmetryFunctionGroupAngularWide::
operator==(SymmetryFunctionGroup const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymmetryFunctionGroupAngularWide const& c =
        dynamic_cast<SymmetryFunctionGroupAngularWide const&>(rhs);
    if (cutoffType  != c.cutoffType ) return false;
    if (cutoffAlpha != c.cutoffAlpha) return false;
    if (rc          != c.rc         ) return false;
    if (e1          != c.e1         ) return false;
    if (e2          != c.e2         ) return false;
    return true;
}

bool SymmetryFunctionGroupAngularWide::
operator<(SymmetryFunctionGroup const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymmetryFunctionGroupAngularWide const& c =
        dynamic_cast<SymmetryFunctionGroupAngularWide const&>(rhs);
    if      (cutoffType  < c.cutoffType ) return true;
    else if (cutoffType  > c.cutoffType ) return false;
    if      (cutoffAlpha < c.cutoffAlpha) return true;
    else if (cutoffAlpha > c.cutoffAlpha) return false;
    if      (rc          < c.rc         ) return true;
    else if (rc          > c.rc         ) return false;
    if      (e1          < c.e1         ) return true;
    else if (e1          > c.e1         ) return false;
    if      (e2          < c.e2         ) return true;
    else if (e2          > c.e2         ) return false;
    return false;
}

bool SymmetryFunctionGroupAngularWide::
addMember(SymmetryFunction const* const symmetryFunction)
{
    if (symmetryFunction->getType() != type) return false;

    SymmetryFunctionAngularWide const* sf =
        dynamic_cast<SymmetryFunctionAngularWide const*>(
        symmetryFunction);

    if (members.empty())
    {
        cutoffType  = sf->getCutoffType();
        cutoffAlpha = sf->getCutoffAlpha();
        ec          = sf->getEc();
        rc          = sf->getRc();
        e1          = sf->getE1();
        e2          = sf->getE2();
        convLength  = sf->getConvLength();

        fc.setCutoffType(cutoffType);
        fc.setCutoffRadius(rc);
        fc.setCutoffParameter(cutoffAlpha);
    }

    if (sf->getCutoffType()  != cutoffType ) return false;
    if (sf->getCutoffAlpha() != cutoffAlpha) return false;
    if (sf->getEc()          != ec         ) return false;
    if (sf->getRc()          != rc         ) return false;
    if (sf->getE1()          != e1         ) return false;
    if (sf->getE2()          != e2         ) return false;
    if (sf->getConvLength()  != convLength )
    {
        throw runtime_error("ERROR: Unable to add symmetry function members "
                            "with different conversion factors.\n");
    }

    members.push_back(sf);

    return true;
}

void SymmetryFunctionGroupAngularWide::sortMembers()
{
    sort(members.begin(),
         members.end(),
         comparePointerTargets<SymmetryFunctionAngularWide const>);

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
        zeta.push_back(members[i]->getZeta());
        lambda.push_back(members[i]->getLambda());
        zetaLambda.push_back(members[i]->getZeta() * members[i]->getLambda());
    }

    return;
}

void SymmetryFunctionGroupAngularWide::setScalingFactors()
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
void SymmetryFunctionGroupAngularWide::calculate(Atom&      atom,
                                                 bool const derivatives) const
{
    double* result = new double[members.size()];
    for (size_t l = 0; l < members.size(); ++l)
    {
        result[l] = 0.0;
    }

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
            double const r2ij = rij * rij;

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
            // SIMPLE EXPRESSIONS:
            //Vec3D drij(atom.neighbors[j].dr);
            double const* const dr1 = nj.dr.r;

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
                        // SIMPLE EXPRESSIONS:
                        //Vec3D drik(atom.neighbors[k].dr);
                        //Vec3D drjk = drik - drij;
                        double const* const dr2 = nk.dr.r;
                        double const dr30 = dr2[0] - dr1[0];
                        double const dr31 = dr2[1] - dr1[1];
                        double const dr32 = dr2[2] - dr1[2];

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
                        double const rinvijik = 1.0 / rij / rik;
                        // SIMPLE EXPRESSIONS:
                        //double const costijk = drij * drik * rinvijik;
                        double const costijk = (dr1[0] * dr2[0] +
                                                dr1[1] * dr2[1] +
                                                dr1[2] * dr2[2]) * rinvijik;
                        double const pfc = pfcij * pfcik;
                        double const r2ik = rik * rik;
                        double const r2sum = r2ij + r2ik;
                        double const pr1 = pfcik * pdfcij / rij;
                        double const pr2 = pfcij * pdfcik / rik;
                        double vexp = 0.0;
                        double rijs = 0.0;
                        double riks = 0.0;

                        for (size_t l = 0; l < members.size(); ++l)
                        {
                            if (calculateExp[l])
                            {
                                if (rs[l] > 0.0)
                                {
                                    rijs = rij - rs[l];
                                    riks = rik - rs[l];
                                    vexp = exp(-eta[l] * (rijs * rijs
                                                        + riks * riks));
                                }
                                else
                                {
                                    vexp = exp(-eta[l] * r2sum);
                                }
                            }
                            double const plambda = 1.0 + lambda[l] * costijk;
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
                            double p1;
                            double p2;
                            if (rs[l] > 0)
                            {
                                p1 = fg * (pfczl * (rinvijik
                                   - costijk / r2ij - p2etapl
                                   * rijs / rij) + pr1 * plambda);
                                p2 = fg * (pfczl * (rinvijik
                                   - costijk / r2ik - p2etapl
                                   * riks / rik) + pr2 * plambda);
                            }
                            else
                            {
                                p1 = fg * (pfczl * (rinvijik - costijk / r2ij
                                   - p2etapl) + pr1 * plambda);
                                p2 = fg * (pfczl * (rinvijik - costijk / r2ik
                                   - p2etapl) + pr2 * plambda);

                            }
                            double const p3 = fg * pfczl * rinvijik;

                            // SIMPLE EXPRESSIONS:
                            // Save force contributions in Atom storage.
                            //atom.dGdr[memberIndex[l]] += p1 * drij
                            //                           + p2 * drik;
                            //atom.neighbors[j].
                            //    dGdr[memberIndex[l]] -= p1 * drij
                            //                          + p3 * drjk;
                            //atom.neighbors[k].
                            //    dGdr[memberIndex[l]] -= p2 * drik
                            //                          - p3 * drjk;

                            double const p1drijx = p1 * dr1[0];
                            double const p1drijy = p1 * dr1[1];
                            double const p1drijz = p1 * dr1[2];

                            double const p2drikx = p2 * dr2[0];
                            double const p2driky = p2 * dr2[1];
                            double const p2drikz = p2 * dr2[2];

                            double const p3drjkx = p3 * dr30;
                            double const p3drjky = p3 * dr31;
                            double const p3drjkz = p3 * dr32;

                            size_t const li = memberIndex[l];
                            double* dGdr = atom.dGdr[li].r;
                            dGdr[0] += p1drijx + p2drikx;
                            dGdr[1] += p1drijy + p2driky;
                            dGdr[2] += p1drijz + p2drikz;

                            dGdr = nj.dGdr[li].r;
                            dGdr[0] -= p1drijx + p3drjkx;
                            dGdr[1] -= p1drijy + p3drjky;
                            dGdr[2] -= p1drijz + p3drjkz;

                            dGdr = nk.dGdr[li].r;
                            dGdr[0] -= p2drikx - p3drjkx;
                            dGdr[1] -= p2driky - p3drjky;
                            dGdr[2] -= p2drikz - p3drjkz;
                        } // l
                    } // rik <= rc
                } // elem
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

vector<string> SymmetryFunctionGroupAngularWide::parameterLines() const
{
    vector<string> v;

    v.push_back(strpr(getPrintFormatCommon().c_str(),
                      index + 1,
                      elementMap[ec].c_str(),
                      type,
                      elementMap[e1].c_str(),
                      elementMap[e2].c_str(),
                      rc / convLength,
                      (int)cutoffType,
                      cutoffAlpha));

    for (size_t i = 0; i < members.size(); ++i)
    {
        v.push_back(strpr(getPrintFormatMember().c_str(),
                          members[i]->getEta() * convLength * convLength,
                          members[i]->getRs() / convLength,
                          members[i]->getLambda(),
                          members[i]->getZeta(),
                          members[i]->getLineNumber(),
                          i + 1,
                          members[i]->getIndex() + 1,
                          (int)calculateExp[i]));
    }

    return v;
}
