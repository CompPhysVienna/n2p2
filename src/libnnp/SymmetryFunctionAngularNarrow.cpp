// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "SymmetryFunctionAngularNarrow.h"
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

SymmetryFunctionAngularNarrow::
SymmetryFunctionAngularNarrow(ElementMap const& elementMap) :
    SymmetryFunction(3, elementMap),
    useIntegerPow(false),
    e1           (0    ),
    e2           (0    ),
    zetaInt      (0    ),
    lambda       (0.0  ),
    eta          (0.0  ),
    zeta         (0.0  )
{
    minNeighbors = 2;
    parameters.insert("e1");
    parameters.insert("e2");
    parameters.insert("eta");
    parameters.insert("zeta");
    parameters.insert("lambda");
}

bool SymmetryFunctionAngularNarrow::
operator==(SymmetryFunction const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymmetryFunctionAngularNarrow const& c =
        dynamic_cast<SymmetryFunctionAngularNarrow const&>(rhs);
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

bool SymmetryFunctionAngularNarrow::
operator<(SymmetryFunction const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymmetryFunctionAngularNarrow const& c =
        dynamic_cast<SymmetryFunctionAngularNarrow const&>(rhs);
    if      (cutoffType  < c.cutoffType ) return true;
    else if (cutoffType  > c.cutoffType ) return false;
    if      (cutoffAlpha < c.cutoffAlpha) return true;
    else if (cutoffAlpha > c.cutoffAlpha) return false;
    if      (rc          < c.rc         ) return true;
    else if (rc          > c.rc         ) return false;
    if      (eta         < c.eta        ) return true;
    else if (eta         > c.eta        ) return false;
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

void SymmetryFunctionAngularNarrow::
     setParameters(string const& parameterString)
{
    vector<string> splitLine = split(reduce(parameterString));

    if (type != (size_t)atoi(splitLine.at(1).c_str()))
    {
        throw runtime_error("ERROR: Incorrect symmetry function type.\n");
    }

    ec     = elementMap[splitLine.at(0)];
    e1     = elementMap[splitLine.at(2)];
    e2     = elementMap[splitLine.at(3)];
    eta    = atof(splitLine.at(4).c_str());
    lambda = atof(splitLine.at(5).c_str());
    zeta   = atof(splitLine.at(6).c_str());
    rc     = atof(splitLine.at(7).c_str());

    fc.setCutoffRadius(rc);

    if (e1 > e2)
    {
        size_t tmp = e1;
        e1         = e2;
        e2         = tmp;
    }

    zetaInt = round(zeta);
    if (fabs(zeta - zetaInt) <= numeric_limits<double>::min())
    {
        useIntegerPow = true;
    }
    else
    {
        useIntegerPow = false;
    }

    return;
}

void SymmetryFunctionAngularNarrow::changeLengthUnit(double convLength)
{
    eta /= convLength * convLength;
    rc *= convLength;

    return;
}

string SymmetryFunctionAngularNarrow::getSettingsLine(
                                                 double const convLength) const
{
    string s = strpr("symfunction_short %2s %2zu %2s %2s %16.8E %16.8E "
                     "%16.8E %16.8E\n",
                     elementMap[ec].c_str(),
                     type,
                     elementMap[e1].c_str(),
                     elementMap[e2].c_str(),
                     eta / (convLength * convLength),
                     lambda,
                     zeta,
                     rc * convLength);

    return s;
}

void SymmetryFunctionAngularNarrow::calculate(
                                  Atom&                       atom,
                                  bool const                  derivatives,
                                  SymmetryFunctionStatistics& statistics) const
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
                        Vec3D drjk = nk.dr - nj.dr;
                        double rjk = drjk.norm2();;
                        if (rjk < rc2)
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
                            double const pexp = exp(-eta *
                                                (r2ij + r2ik + rjk * rjk));
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
                                            / r2ij - p2etapl) + pfcik * pfcjk
                                            * pdfcij * plambda / rij);
                            double const p2 = fg * (pfc * (rinvijik - costijk
                                            / r2ik - p2etapl) + pfcij * pfcjk
                                            * pdfcik * plambda / rik);
                            double const p3 = fg * (pfc * (rinvijik + p2etapl)
                                            - pfcij * pfcik * pdfcjk
                                            * plambda / rjk);
                            drij *= p1 * scalingFactor;
                            drik *= p2 * scalingFactor;
                            drjk *= p3 * scalingFactor;

                            // Save force contributions in Atom storage.
                            atom.dGdr[index] += drij + drik;
                            nj.dGdr[index]   -= drij + drjk;
                            nk.dGdr[index]   -= drik - drjk;
                        } // rjk <= rc
                    } // rik <= rc
                } // elem
            } // k
        } // rij <= rc
    } // j
    result *= pnorm;

    updateStatisticsGeneric(result, atom, statistics);
    atom.G[index] = scale(result);

    return;
}

string SymmetryFunctionAngularNarrow::parameterLine() const
{
    return strpr(getPrintFormat().c_str(),
                 index + 1,
                 elementMap[ec].c_str(),
                 type,
                 elementMap[e1].c_str(),
                 elementMap[e2].c_str(),
                 eta,
                 lambda,
                 zeta,
                 rc,
                 (int)cutoffType,
                 cutoffAlpha,
                 lineNumber + 1);
}

vector<string> SymmetryFunctionAngularNarrow::parameterInfo() const
{
    vector<string> v = SymmetryFunction::parameterInfo();
    string s;
    size_t w = sfinfoWidth;

    s = "e1";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), elementMap[e1].c_str()));
    s = "e2";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), elementMap[e2].c_str()));
    s = "eta";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), eta));
    s = "lambda";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), lambda));
    s = "zeta";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), zeta));

    return v;
}

double SymmetryFunctionAngularNarrow::calculateRadialPart(
                                                         double distance) const
{
    double const& r = distance;
    double const p = exp(-eta * r * r) * fc.f(r);

    return p * p * p;
}

double SymmetryFunctionAngularNarrow::calculateAngularPart(double angle) const
{
    return 2.0 * pow((1.0 + lambda * cos(angle)) / 2.0, zeta);
}
