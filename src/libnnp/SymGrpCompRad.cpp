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

#include "SymGrpCompRad.h"
#include "Atom.h"
#include "SymFnc.h"
#include "SymFncCompRad.h"
#include "Vec3D.h"
#include "utility.h"
#include <algorithm> // std::sort
#include <cmath>     // exp
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

SymGrpCompRad::
SymGrpCompRad(ElementMap const& elementMap) :
    SymGrpBaseComp(20, elementMap),
    e1(0)
{
    parametersCommon.insert("e1");
}

bool SymGrpCompRad::operator==(SymGrp const& rhs) const
{
    if (ec   != rhs.getEc()  ) return false;
    if (type != rhs.getType()) return false;
    SymGrpCompRad const& c = dynamic_cast<SymGrpCompRad const&>(rhs);
    if (e1 != c.e1) return false;
    return true;
}

bool SymGrpCompRad::operator<(SymGrp const& rhs) const
{
    if      (ec   < rhs.getEc()  ) return true;
    else if (ec   > rhs.getEc()  ) return false;
    if      (type < rhs.getType()) return true;
    else if (type > rhs.getType()) return false;
    SymGrpCompRad const& c = dynamic_cast<SymGrpCompRad const&>(rhs);
    if      (e1 < c.e1) return true;
    else if (e1 > c.e1) return false;
    return false;
}

bool SymGrpCompRad::addMember(SymFnc const* const symmetryFunction)
{
    if (symmetryFunction->getType() != type) return false;

    SymFncCompRad const* sf =
        dynamic_cast<SymFncCompRad const*>(symmetryFunction);

    if (members.empty())
    {
        ec          = sf->getEc();
        e1          = sf->getE1();
        convLength  = sf->getConvLength();
    }

    if (sf->getEc()          != ec         ) return false;
    if (sf->getE1()          != e1         ) return false;
    if (sf->getConvLength()  != convLength )
    {
        throw runtime_error("ERROR: Unable to add symmetry function members "
                            "with different conversion factors.\n");
    }

    if (sf->getRl() <= 0.0) rmin = 0.0;
    else rmin = min( rmin, sf->getRl() );
    rmax = max( rmax, sf->getRc() );

    members.push_back(sf);

    return true;
}

void SymGrpCompRad::sortMembers()
{
    sort(members.begin(),
         members.end(),
         comparePointerTargets<SymFncCompRad const>);

    for (size_t i = 0; i < members.size(); i++)
    {
        memberIndex.push_back(members[i]->getIndex());
        memberIndexPerElement.push_back(members[i]->getIndexPerElement());
    }

    mrl.resize(members.size(), 0.0);
    mrc.resize(members.size(), 0.0);
    for (size_t i = 0; i < members.size(); i++)
    {
        mrl.at(i) = members[i]->getRl();
        mrc.at(i) = members[i]->getRc();
    }

#ifndef NNP_NO_SF_CACHE
    mci.resize(members.size());
    for (size_t k = 0; k < members.size(); ++k)
    {
        mci.at(k) = members.at(k)->getCacheIndices()[e1];
    }
#endif

    return;
}

void SymGrpCompRad::setScalingFactors()
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
// temporary objects has been minimized. Some of the originally coded
// expressions are kept in comments marked with "SIMPLE EXPRESSIONS:".
void SymGrpCompRad::calculate(Atom& atom, bool const derivatives) const
{
    double* result = new double[members.size()];
    for (size_t k = 0; k < members.size(); ++k)
    {
        result[k] = 0.0;
    }

    for (size_t j = 0; j < atom.numNeighbors; ++j)
    {
        Atom::Neighbor& n = atom.neighbors[j];
        double const rij = n.d;
        if (e1 == n.element && rij < rmax && rij > rmin)
        {
            // Energy calculation.
            double const* const d1 = n.dr.r;
            for (size_t k = 0; k < members.size(); ++k)
            {
                SymFncCompRad const& sf = *(members[k]);
                if (rij <= mrl[k] || rij >= mrc[k]) continue;
                double rad;
                double drad;
#ifndef NNP_NO_SF_CACHE
                if (mci[k].size() == 0) sf.getCompactOnly(rij, rad, drad);
                else
                {
                    double& crad = n.cache[mci[k][0]];
                    double& cdrad = n.cache[mci[k][1]];
                    if (crad < 0) sf.getCompactOnly(rij, crad, cdrad);
                    rad = crad;
                    drad = cdrad;
                }
#else
                sf.getCompactOnly(rij, rad, drad);
#endif
                result[k] += rad;

                // Force calculation.
                if (!derivatives || drad == 0.0) continue;
                drad *= scalingFactors[k] / rij;
                // SIMPLE EXPRESSIONS:
                //Vec3D const dij = p1 * atom.neighbors[j].dr;
                double const p1drijx = drad * d1[0];
                double const p1drijy = drad * d1[1];
                double const p1drijz = drad * d1[2];

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

vector<string> SymGrpCompRad::parameterLines() const
{
    vector<string> v;

    v.push_back(strpr(getPrintFormatCommon().c_str(),
                      index + 1,
                      elementMap[ec].c_str(),
                      type,
                      elementMap[e1].c_str(),
                      rmin / convLength,
                      rmax / convLength));

    for (size_t i = 0; i < members.size(); ++i)
    {
        v.push_back(strpr(getPrintFormatMember().c_str(),
                          members[i]->getSubtype().c_str(),
                          members[i]->getRl() / convLength,
                          members[i]->getRc() / convLength,
                          members[i]->getLineNumber() + 1,
                          i + 1,
                          members[i]->getIndex() + 1));
    }

    return v;
}
