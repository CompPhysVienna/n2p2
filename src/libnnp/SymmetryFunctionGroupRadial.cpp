// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "SymmetryFunctionGroupRadial.h"
#include "Atom.h"
#include "SymmetryFunction.h"
#include "SymmetryFunctionRadial.h"
#include "Vec3D.h"
#include "utility.h"
#include <algorithm> // std::sort
#include <cmath>     // exp

using namespace std;
using namespace nnp;

SymmetryFunctionGroupRadial::
SymmetryFunctionGroupRadial(ElementMap const& elementMap) :
    SymmetryFunctionGroup(2, elementMap),
    e1(0)
{
    parametersCommon.insert("e1");

    parametersMember.insert("eta");
    parametersMember.insert("rs");
    parametersMember.insert("mindex");
    parametersMember.insert("sfindex");
}

bool SymmetryFunctionGroupRadial::
operator==(SymmetryFunctionGroup const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymmetryFunctionGroupRadial const& c =
        dynamic_cast<SymmetryFunctionGroupRadial const&>(rhs);
    if (cutoffType  != c.cutoffType ) return false;
    if (cutoffAlpha != c.cutoffAlpha) return false;
    if (rc          != c.rc         ) return false;
    if (e1          != c.e1         ) return false;
    return true;
}

bool SymmetryFunctionGroupRadial::
operator<(SymmetryFunctionGroup const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymmetryFunctionGroupRadial const& c =
        dynamic_cast<SymmetryFunctionGroupRadial const&>(rhs);
    if      (cutoffType  < c.cutoffType ) return true;
    else if (cutoffType  > c.cutoffType ) return false;
    if      (cutoffAlpha < c.cutoffAlpha) return true;
    else if (cutoffAlpha > c.cutoffAlpha) return false;
    if      (rc          < c.rc         ) return true;
    else if (rc          > c.rc         ) return false;
    if      (e1          < c.e1         ) return true;
    else if (e1          > c.e1         ) return false;
    return false;
}

bool SymmetryFunctionGroupRadial::
addMember(SymmetryFunction const* const symmetryFunction)
{
    if (symmetryFunction->getType() != type) return false;

    SymmetryFunctionRadial const* sf =
        dynamic_cast<SymmetryFunctionRadial const*>(symmetryFunction);

    if (members.empty())
    {
        cutoffType  = sf->getCutoffType();
        cutoffAlpha = sf->getCutoffAlpha();
        ec          = sf->getEc();
        rc          = sf->getRc();
        e1          = sf->getE1();

        fc.setCutoffType(cutoffType);
        fc.setCutoffRadius(rc);
        fc.setCutoffParameter(cutoffAlpha);
    }

    if (sf->getCutoffType()  != cutoffType ) return false;
    if (sf->getCutoffAlpha() != cutoffAlpha) return false;
    if (sf->getEc()          != ec         ) return false;
    if (sf->getRc()          != rc         ) return false;
    if (sf->getE1()          != e1         ) return false;

    members.push_back(sf);

    return true;
}

void SymmetryFunctionGroupRadial::sortMembers()
{
    sort(members.begin(),
         members.end(),
         comparePointerTargets<SymmetryFunctionRadial const>);

    for (size_t i = 0; i < members.size(); i++)
    {
        memberIndex.push_back(members[i]->getIndex());
        eta.push_back(members[i]->getEta());
        rs.push_back(members[i]->getRs());
    }

    return;
}

void SymmetryFunctionGroupRadial::setScalingFactors()
{
    scalingFactors.resize(members.size(), 0.0);
    for (size_t i = 0; i < members.size(); i++)
    {
        scalingFactors.at(i) = members[i]->getScalingFactor();
    }

    return;
}

// Depending on chosen symmetry functions this function may be very
// time-critical when predicting new structures (e.g. in MD simulations). Thus,
// lots of optimizations were used sacrificing some readablity. Vec3D
// operations have been rewritten in simple C array style and the use of
// temporary objects has been minmized. Some of the originally coded
// expressions are kept in comments marked with "SIMPLE EXPRESSIONS:".
void SymmetryFunctionGroupRadial::calculate(
                                  Atom&                       atom,
                                  bool const                  derivatives,
                                  SymmetryFunctionStatistics& statistics) const
{
    double* result = new double[members.size()];
    for (size_t k = 0; k < members.size(); ++k)
    {
        result[k] = 0.0;
    }

    for (size_t j = 0; j < atom.numNeighbors; ++j)
    {
        Atom::Neighbor& n = atom.neighbors[j];
        if (e1 == n.element && n.d < rc)
        {
            // Energy calculation.
            double const rij = n.d;

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
            double const* const d1 = n.dr.r;
            for (size_t k = 0; k < members.size(); ++k)
            {
                double pexp = exp(-eta[k] * (rij - rs[k]) * (rij - rs[k]));
                result[k] += pexp * pfc;
                // Force calculation.
                if (!derivatives) continue;
                double const p1 = scalingFactors[k] * (pdfc - 2.0 * eta[k]
                                * (rij - rs[k]) * pfc) * pexp / rij;
                // SIMPLE EXPRESSIONS:
                //Vec3D const dij = p1 * atom.neighbors[j].dr;
                double const p1drijx = p1 * d1[0];
                double const p1drijy = p1 * d1[1];
                double const p1drijz = p1 * d1[2];

                // Save force contributions in Atom storage.
                size_t const ki = memberIndex[k];
                // SIMPLE EXPRESSIONS:
                //atom.dGdr[ki]              += dij;
                //atom.neighbors[j].dGdr[ki] -= dij;

                double* dGdr = atom.dGdr[ki].r;
                dGdr[0] += p1drijx;
                dGdr[1] += p1drijy;
                dGdr[2] += p1drijz;

                dGdr = n.dGdr[ki].r;
                dGdr[0] -= p1drijx;
                dGdr[1] -= p1drijy;
                dGdr[2] -= p1drijz;
            }
        }
    }

    for (size_t k = 0; k < members.size(); ++k)
    {
        members[k]->updateStatisticsGeneric(result[k], atom, statistics);
        atom.G[memberIndex[k]] = members[k]->scale(result[k]);
    }

    delete[] result;

    return;
}

vector<string> SymmetryFunctionGroupRadial::parameterLines() const
{
    vector<string> v;

    v.push_back(strpr(getPrintFormatCommon().c_str(),
                      index + 1,
                      elementMap[ec].c_str(),
                      type,
                      elementMap[e1].c_str(),
                      rc,
                      (int)cutoffType,
                      cutoffAlpha));

    for (size_t i = 0; i < members.size(); ++i)
    {
        v.push_back(strpr(getPrintFormatMember().c_str(),
                          members[i]->getEta(),
                          members[i]->getRs(),
                          members[i]->getLineNumber(),
                          i + 1,
                          members[i]->getIndex() + 1));
    }

    return v;
}

