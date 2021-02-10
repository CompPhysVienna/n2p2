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

#include "SymGrpExpRad.h"
#include "Atom.h"
#include "SymFnc.h"
#include "SymFncExpRad.h"
#include "Vec3D.h"
#include "utility.h"
#include <algorithm> // std::sort
#include <cmath>     // exp
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

SymGrpExpRad::SymGrpExpRad(ElementMap const& elementMap) :
    SymGrpBaseCutoff(2, elementMap),
    e1(0)
{
    parametersCommon.insert("e1");

    parametersMember.insert("eta");
    parametersMember.insert("rs/rl");
    parametersMember.insert("mindex");
    parametersMember.insert("sfindex");
}

bool SymGrpExpRad::operator==(SymGrp const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymGrpExpRad const& c = dynamic_cast<SymGrpExpRad const&>(rhs);
    if (cutoffType  != c.cutoffType ) return false;
    if (cutoffAlpha != c.cutoffAlpha) return false;
    if (rc          != c.rc         ) return false;
    if (e1          != c.e1         ) return false;
    return true;
}

bool SymGrpExpRad::operator<(SymGrp const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymGrpExpRad const& c = dynamic_cast<SymGrpExpRad const&>(rhs);
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

bool SymGrpExpRad::addMember(SymFnc const* const symmetryFunction)
{
    if (symmetryFunction->getType() != type) return false;

    SymFncExpRad const* sf =
        dynamic_cast<SymFncExpRad const*>(symmetryFunction);

    if (members.empty())
    {
        cutoffType  = sf->getCutoffType();
        subtype     = sf->getSubtype();
        cutoffAlpha = sf->getCutoffAlpha();
        ec          = sf->getEc();
        rc          = sf->getRc();
        e1          = sf->getE1();
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
    if (sf->getConvLength()  != convLength )
    {
        throw runtime_error("ERROR: Unable to add symmetry function members "
                            "with different conversion factors.\n");
    }

    members.push_back(sf);

    return true;
}

void SymGrpExpRad::sortMembers()
{
    sort(members.begin(),
         members.end(),
         comparePointerTargets<SymFncExpRad const>);

    for (size_t i = 0; i < members.size(); i++)
    {
        memberIndex.push_back(members[i]->getIndex());
        eta.push_back(members[i]->getEta());
        rs.push_back(members[i]->getRs());
        memberIndexPerElement.push_back(members[i]->getIndexPerElement());
    }

    return;
}

void SymGrpExpRad::setScalingFactors()
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
void SymGrpExpRad::calculate(Atom& atom, bool const derivatives) const
{
#ifndef NNP_NO_SF_CACHE
    // Can use cache indices of any member because this group is defined via
    // identical symmetry function type, neighbors and cutoff functions.
    auto cacheIndices = members.at(0)->getCacheIndices();
    bool unique = true;
    size_t c0 = 0;
    size_t c1 = 0;
    if (cacheIndices.at(e1).size() > 0)
    {
        unique = false;
        c0 = cacheIndices.at(e1).at(0);
        c1 = cacheIndices.at(e1).at(1);
    }
#endif
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
            double pfc;
            double pdfc;
#ifndef NNP_NO_SF_CACHE
            if (unique) fc.fdf(rij, pfc, pdfc);
            else
            {
                double& cfc = n.cache[c0];
                double& cdfc = n.cache[c1];
                if (cfc < 0) fc.fdf(rij, cfc, cdfc);
                pfc = cfc;
                pdfc = cdfc;
            }
#else
            fc.fdf(rij, pfc, pdfc);
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
#ifndef NNP_FULL_SFD_MEMORY
                size_t ki = memberIndex[k];
#else
                size_t const ki = memberIndex[k];
#endif
                // SIMPLE EXPRESSIONS:
                //atom.dGdr[ki]              += dij;
                //atom.neighbors[j].dGdr[ki] -= dij;

                double* dGdr = atom.dGdr[ki].r;
                dGdr[0] += p1drijx;
                dGdr[1] += p1drijy;
                dGdr[2] += p1drijz;

#ifndef NNP_FULL_SFD_MEMORY
                ki = memberIndexPerElement[k][e1];
#endif
                dGdr = n.dGdr[ki].r;
                dGdr[0] -= p1drijx;
                dGdr[1] -= p1drijy;
                dGdr[2] -= p1drijz;
            }
        }
    }

    for (size_t k = 0; k < members.size(); ++k)
    {
        atom.G[memberIndex[k]] = members[k]->scale(result[k]);
    }

    delete[] result;

    return;
}

vector<string> SymGrpExpRad::parameterLines() const
{
    vector<string> v;

    v.push_back(strpr(getPrintFormatCommon().c_str(),
                      index + 1,
                      elementMap[ec].c_str(),
                      type,
                      subtype.c_str(),
                      elementMap[e1].c_str(),
                      rc / convLength,
                      cutoffAlpha));

    for (size_t i = 0; i < members.size(); ++i)
    {
        v.push_back(strpr(getPrintFormatMember().c_str(),
                          members[i]->getEta() * convLength * convLength,
                          members[i]->getRs() / convLength,
                          members[i]->getLineNumber() + 1,
                          i + 1,
                          members[i]->getIndex() + 1));
    }

    return v;
}

