// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "SymmetryFunctionWeightedRadial.h"
#include "Atom.h"
#include "ElementMap.h"
#include "utility.h"
#include "Vec3D.h"
#include <cstdlib>   // atof, atoi
#include <cmath>     // exp
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

SymmetryFunctionWeightedRadial::
SymmetryFunctionWeightedRadial(ElementMap const& elementMap) :
    SymmetryFunction(12, elementMap),
    eta             (0.0),
    rs              (0.0)
{
    minNeighbors = 1;
    parameters.insert("eta");
    parameters.insert("rs");
}

bool SymmetryFunctionWeightedRadial::operator==(
                                             SymmetryFunction const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymmetryFunctionWeightedRadial const& c =
        dynamic_cast<SymmetryFunctionWeightedRadial const&>(rhs);
    if (cutoffType  != c.cutoffType ) return false;
    if (cutoffAlpha != c.cutoffAlpha) return false;
    if (rc          != c.rc         ) return false;
    if (eta         != c.eta        ) return false;
    if (rs          != c.rs         ) return false;
    return true;
}

bool SymmetryFunctionWeightedRadial::operator<(
                                             SymmetryFunction const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymmetryFunctionWeightedRadial const& c =
        dynamic_cast<SymmetryFunctionWeightedRadial const&>(rhs);
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
    return false;
}

void SymmetryFunctionWeightedRadial::setParameters(
                                                 string const& parameterString)
{
    vector<string> splitLine = split(reduce(parameterString));

    if (type != (size_t)atoi(splitLine.at(1).c_str()))
    {
        throw runtime_error("ERROR: Incorrect symmetry function type.\n");
    }

    ec   = elementMap[splitLine.at(0)];
    eta  = atof(splitLine.at(2).c_str());
    rs   = atof(splitLine.at(3).c_str());
    rc   = atof(splitLine.at(4).c_str());

    fc.setCutoffRadius(rc);

    return;
}

void SymmetryFunctionWeightedRadial::changeLengthUnit(double convLength)
{
    eta /= convLength * convLength;
    rs *= convLength;
    rc *= convLength;

    return;
}

string SymmetryFunctionWeightedRadial::getSettingsLine(
                                                 double const convLength) const
{
    string s = strpr("symfunction_short %2s %2zu %16.8E %16.8E %16.8E\n",
                     elementMap[ec].c_str(),
                     type,
                     eta / (convLength * convLength),
                     rs * convLength,
                     rc * convLength);

    return s;
}

void SymmetryFunctionWeightedRadial::calculate(
                                  Atom&                       atom,
                                  bool const                  derivatives,
                                  SymmetryFunctionStatistics& statistics) const
{
    double result = 0.0;

    for (size_t j = 0; j < atom.numNeighbors; ++j)
    {
        Atom::Neighbor& n = atom.neighbors[j];
        if (n.d < rc)
        {
            // Energy calculation.
            double const rij = n.d;
            double const pexp = elementMap.atomicNumber(n.element)
                              * exp(-eta * (rij - rs) * (rij - rs));

            // Calculate cutoff function and derivative.
#ifdef NOCFCACHE
            double pfc;
            double pdfc;
            fc.fdf(rij, pfc, pdfc);
#else
            // If cutoff radius matches with the one in the neighbor storage
            // we can use the previously calculated value.
            double& pfc = n.fc;
            double& pdfc = n.dfc;
            if (n.cutoffType != cutoffType ||
                n.rc != rc ||
                n.cutoffAlpha != cutoffAlpha)
            {
                fc.fdf(rij, pfc, pdfc);
                n.rc = rc;
                n.cutoffType = cutoffType;
                n.cutoffAlpha = cutoffAlpha;
            }
#endif

            result += pexp * pfc;
            // Force calculation.
            if (!derivatives) continue;
            double const p1 = scalingFactor
                * (pdfc - 2.0 * eta * (rij - rs)
                * pfc) * pexp / rij;
            Vec3D dij = p1 * n.dr;
            // Save force contributions in Atom storage.
            atom.dGdr[index] += dij;
            n.dGdr[index]    -= dij;
        }
    }

    updateStatisticsGeneric(result, atom, statistics);
    atom.G[index] = scale(result);

    return;
}

string SymmetryFunctionWeightedRadial::parameterLine() const
{
    return strpr(getPrintFormat().c_str(),
                 index + 1,
                 elementMap[ec].c_str(),
                 type,
                 eta,
                 rs,
                 rc,
                 (int)cutoffType,
                 cutoffAlpha,
                 lineNumber + 1);
}

vector<string> SymmetryFunctionWeightedRadial::parameterInfo() const
{
    vector<string> v = SymmetryFunction::parameterInfo();
    string s;
    size_t w = sfinfoWidth;

    s = "eta";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), eta));
    s = "rs";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), rs));

    return v;
}

double SymmetryFunctionWeightedRadial::calculateRadialPart(
                                                         double distance) const
{
    double const& r = distance;

    return exp(-eta * (r - rs) * (r - rs)) * fc.f(r);
}

double SymmetryFunctionWeightedRadial::calculateAngularPart(
                                                      double /* angle */) const
{
    return 1.0;
}
