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

#include "SymFncExpAngnWeighted.h"
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

SymFncExpAngnWeighted::SymFncExpAngnWeighted(ElementMap const& elementMap) :
    SymFncBaseCutoff(13, elementMap),
    useIntegerPow(false),
    zetaInt      (0    ),
    eta          (0.0  ),
    rs           (0.0  ),
    lambda       (0.0  ),
    zeta         (0.0  )
{
    minNeighbors = 2;
    parameters.insert("rs/rl");
    parameters.insert("eta");
    parameters.insert("zeta");
    parameters.insert("lambda");
}

bool SymFncExpAngnWeighted::operator==(SymFnc const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymFncExpAngnWeighted const& c =
        dynamic_cast<SymFncExpAngnWeighted const&>(rhs);
    if (cutoffType  != c.cutoffType ) return false;
    if (cutoffAlpha != c.cutoffAlpha) return false;
    if (rc          != c.rc         ) return false;
    if (eta         != c.eta        ) return false;
    if (rs          != c.rs         ) return false;
    if (zeta        != c.zeta       ) return false;
    if (lambda      != c.lambda     ) return false;
    return true;
}

bool SymFncExpAngnWeighted::operator<(SymFnc const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymFncExpAngnWeighted const& c =
        dynamic_cast<SymFncExpAngnWeighted const&>(rhs);
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
    return false;
}

void SymFncExpAngnWeighted::setParameters(string const& parameterString)
{
    vector<string> splitLine = split(reduce(parameterString));

    if (type != (size_t)atoi(splitLine.at(1).c_str()))
    {
        throw runtime_error("ERROR: Incorrect symmetry function type.\n");
    }

    ec     = elementMap[splitLine.at(0)];
    eta    = atof(splitLine.at(2).c_str());
    rs     = atof(splitLine.at(3).c_str());
    lambda = atof(splitLine.at(4).c_str());
    zeta   = atof(splitLine.at(5).c_str());
    rc     = atof(splitLine.at(6).c_str());

    fc.setCutoffRadius(rc);
    fc.setCutoffParameter(cutoffAlpha);

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

void SymFncExpAngnWeighted::changeLengthUnit(double convLength)
{
    this->convLength = convLength;
    eta /= convLength * convLength;
    rs *= convLength;
    rc *= convLength;

    fc.setCutoffRadius(rc);
    fc.setCutoffParameter(cutoffAlpha);

    return;
}

string SymFncExpAngnWeighted::getSettingsLine() const
{
    string s = strpr("symfunction_short %2s %2zu %16.8E %16.8E %16.8E "
                     "%16.8E %16.8E\n",
                     elementMap[ec].c_str(),
                     type,
                     eta * convLength * convLength,
                     rs / convLength,
                     lambda,
                     zeta,
                     rc / convLength);

    return s;
}

void SymFncExpAngnWeighted::calculate(Atom& atom, bool const derivatives) const
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
        if (rij < rc)
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
                        double const pexp = elementMap.atomicNumber(nej)
                                          * elementMap.atomicNumber(nek)
                                          * exp(-eta * (rijs * rijs +
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
                                        / r2ij - p2etapl * rijs / rij) + pfcik
                                        * pfcjk * pdfcij * plambda / rij);
                        double const p2 = fg * (pfc * (rinvijik - costijk
                                        / r2ik - p2etapl * riks / rik) + pfcij
                                        * pfcjk * pdfcik * plambda / rik);
                        double const p3 = fg * (pfc * (rinvijik + p2etapl
                                        * rjks / rjk) - pfcij * pfcik * pdfcjk
                                        * plambda / rjk);
                        drij *= p1 * scalingFactor;
                        drik *= p2 * scalingFactor;
                        drjk *= p3 * scalingFactor;

                        // Save force contributions in Atom storage.
                        atom.dGdr[index] += drij + drik;
#ifndef NNP_FULL_SFD_MEMORY
                        nj.dGdr[indexPerElement[nj.element]] -= drij + drjk;
                        nk.dGdr[indexPerElement[nk.element]] -= drik - drjk;
#else
                        nj.dGdr[index] -= drij + drjk;
                        nk.dGdr[index] -= drik - drjk;
#endif
                    } // rjk <= rc
                } // rik <= rc
            } // k
        } // rij <= rc
    } // j
    result *= pnorm;

    atom.G[index] = scale(result);

    return;
}

string SymFncExpAngnWeighted::parameterLine() const
{
    return strpr(getPrintFormat().c_str(),
                 index + 1,
                 elementMap[ec].c_str(),
                 type,
                 subtype.c_str(),
                 eta * convLength * convLength,
                 rs / convLength,
                 rc / convLength,
                 lambda,
                 zeta,
                 cutoffAlpha,
                 lineNumber + 1);
}

vector<string> SymFncExpAngnWeighted::parameterInfo() const
{
    vector<string> v = SymFncBaseCutoff::parameterInfo();
    string s;
    size_t w = sfinfoWidth;

    s = "eta";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(),
                      eta * convLength * convLength));
    s = "lambda";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), lambda));
    s = "zeta";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), zeta));
    s = "rs";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), rs / convLength));

    return v;
}

double SymFncExpAngnWeighted::calculateRadialPart(double distance) const
{
    double const& r = distance * convLength;
    double const p = exp(-eta * (r - rs) * (r - rs)) * fc.f(r);

    return p * p * p;
}

double SymFncExpAngnWeighted::calculateAngularPart(double angle) const
{
    return 2.0 * pow((1.0 + lambda * cos(angle)) / 2.0, zeta);
}

bool SymFncExpAngnWeighted::checkRelevantElement(size_t /*index*/) const
{
    return true;
}

#ifndef NNP_NO_SF_CACHE
vector<string> SymFncExpAngnWeighted::getCacheIdentifiers() const
{
    vector<string> v;
    string s("");

    s += subtype;
    s += " ";
    s += strpr("alpha = %16.8E", cutoffAlpha);
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
