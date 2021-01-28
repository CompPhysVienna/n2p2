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

#include "Atom.h"
#include "Element.h"
#include "NeuralNetwork.h"
#include "SymFnc.h"
#include "SymFncBaseCutoff.h"
#include "SymFncExpRad.h"
#include "SymFncCompRad.h"
#include "SymFncExpAngn.h"
#include "SymFncExpAngw.h"
#include "SymFncCompAngw.h"
#include "SymFncCompAngn.h"
#include "SymFncExpRadWeighted.h"
#include "SymFncExpAngnWeighted.h"
#include "SymFncCompRadWeighted.h"
#include "SymFncCompAngnWeighted.h"
#include "SymFncCompAngwWeighted.h"
#include "SymGrp.h"
#include "SymGrpExpRad.h"
#include "SymGrpCompRad.h"
#include "SymGrpExpAngn.h"
#include "SymGrpExpAngw.h"
#include "SymGrpCompAngw.h"
#include "SymGrpCompAngn.h"
#include "SymGrpExpRadWeighted.h"
#include "SymGrpExpAngnWeighted.h"
#include "SymGrpCompRadWeighted.h"
#include "SymGrpCompAngnWeighted.h"
#include "SymGrpCompAngwWeighted.h"
#include "utility.h"
#include <iostream>  // std::cerr
#include <cstdlib>   // atoi
#include <algorithm> // std::sort, std::min, std::max
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

Element::Element(size_t const index, ElementMap const& elementMap) :
    elementMap         (elementMap                    ),
    index              (index                         ),
    atomicNumber       (elementMap.atomicNumber(index)),
    atomicEnergyOffset (0.0                           ),
    symbol             (elementMap.symbol(index)      )
{
}

Element::~Element()
{
    for (vector<SymFnc*>::const_iterator
         it = symmetryFunctions.begin(); it != symmetryFunctions.end(); ++it)
    {
        delete *it;
    }

    for (vector<SymGrp*>::const_iterator
         it = symmetryFunctionGroups.begin();
         it != symmetryFunctionGroups.end(); ++it)
    {
        delete *it;
    }
}

void Element::addSymmetryFunction(string const& parameters,
                                  size_t const& lineNumber)
{
    vector<string> args = split(reduce(parameters));
    size_t         type = (size_t)atoi(args.at(1).c_str());

    if (type == 2)
    {
        symmetryFunctions.push_back(new SymFncExpRad(elementMap));
    }
    else if (type == 3)
    {
        symmetryFunctions.push_back(new SymFncExpAngn(elementMap));
    }
    else if (type == 9)
    {
        symmetryFunctions.push_back(new SymFncExpAngw(elementMap));
    }
    else if (type == 12)
    {
        symmetryFunctions.push_back(new SymFncExpRadWeighted(elementMap));
    }
    else if (type == 13)
    {
        symmetryFunctions.push_back(new SymFncExpAngnWeighted(elementMap));
    }
    else if (type == 20)
    {
        symmetryFunctions.push_back(new SymFncCompRad(elementMap));
    }
    else if (type == 21)
    {
        symmetryFunctions.push_back(new SymFncCompAngn(elementMap));
    }
    else if (type == 22)
    {
        symmetryFunctions.push_back(new SymFncCompAngw(elementMap));
    }
    else if (type == 23)
    {
        symmetryFunctions.push_back(new SymFncCompRadWeighted(elementMap));
    }
    else if (type == 24)
    {
        symmetryFunctions.push_back(new SymFncCompAngnWeighted(elementMap));
    }
    else if (type == 25)
    {
        symmetryFunctions.push_back(new SymFncCompAngwWeighted(elementMap));
    }
    else
    {
        throw runtime_error("ERROR: Unknown symmetry function type.\n");
    }

    symmetryFunctions.back()->setParameters(parameters);
    symmetryFunctions.back()->setLineNumber(lineNumber);

    return;
}

void Element::changeLengthUnitSymmetryFunctions(double convLength)
{
    for (vector<SymFnc*>::iterator it = symmetryFunctions.begin();
         it != symmetryFunctions.end(); ++it)
    {
        (*it)->changeLengthUnit(convLength);
    }

    return;
}

void Element::sortSymmetryFunctions()
{
    sort(symmetryFunctions.begin(),
         symmetryFunctions.end(),
         comparePointerTargets<SymFnc>);

    for (size_t i = 0; i < symmetryFunctions.size(); ++i)
    {
        symmetryFunctions.at(i)->setIndex(i);
    }

    return;
}

vector<string> Element::infoSymmetryFunctionParameters() const
{
    vector<string> v;

    for (vector<SymFnc*>::const_iterator
         sf = symmetryFunctions.begin(); sf != symmetryFunctions.end(); ++sf)
    {
        v.push_back((*sf)->parameterLine());
    }

    return v;
}

vector<string> Element::infoSymmetryFunctionScaling() const
{
    vector<string> v;

    for (vector<SymFnc*>::const_iterator
         sf = symmetryFunctions.begin(); sf != symmetryFunctions.end(); ++sf)
    {
        v.push_back((*sf)->scalingLine());
    }

    return v;
}

void Element::setupSymmetryFunctionGroups()
{
    for (vector<SymFnc*>::const_iterator
         sf = symmetryFunctions.begin(); sf != symmetryFunctions.end(); ++sf)
    {
        bool createNewGroup = true;
        for (vector<SymGrp*>::const_iterator
             sfg = symmetryFunctionGroups.begin();
             sfg != symmetryFunctionGroups.end(); ++sfg)
        {
            if ((*sfg)->addMember((*sf)))
            {
                createNewGroup = false;
                break;
            }
        }
        if (createNewGroup)
        {
            if ((*sf)->getType() == 2)
            {
                symmetryFunctionGroups.push_back((SymGrp*)
                    new SymGrpExpRad(elementMap));
            }
            else if ((*sf)->getType() == 3)
            {
                symmetryFunctionGroups.push_back((SymGrp*)
                    new SymGrpExpAngn(elementMap));
            }
            else if ((*sf)->getType() == 9)
            {
                symmetryFunctionGroups.push_back((SymGrp*)
                    new SymGrpExpAngw(elementMap));
            }
            else if ((*sf)->getType() == 12)
            {
                symmetryFunctionGroups.push_back((SymGrp*)
                    new SymGrpExpRadWeighted(elementMap));
            }
            else if ((*sf)->getType() == 13)
            {
                symmetryFunctionGroups.push_back((SymGrp*)
                    new SymGrpExpAngnWeighted(elementMap));
            }
            else if ((*sf)->getType() == 20)
            {
                symmetryFunctionGroups.push_back((SymGrp*)
                    new SymGrpCompRad(elementMap));
            }
            else if ((*sf)->getType() == 21)
            {
                symmetryFunctionGroups.push_back((SymGrp*)
                    new SymGrpCompAngn(elementMap));
            }
            else if ((*sf)->getType() == 22)
            {
                symmetryFunctionGroups.push_back((SymGrp*)
                    new SymGrpCompAngw(elementMap));
            }
            else if ((*sf)->getType() == 23)
            {
                symmetryFunctionGroups.push_back((SymGrp*)
                    new SymGrpCompRadWeighted(elementMap));
            }
            else if ((*sf)->getType() == 24)
            {
                symmetryFunctionGroups.push_back((SymGrp*)
                    new SymGrpCompAngnWeighted(elementMap));
            }
            else if ((*sf)->getType() == 25)
            {
                symmetryFunctionGroups.push_back((SymGrp*)
                    new SymGrpCompAngwWeighted(elementMap));
            }
            else
            {
                throw runtime_error("ERROR: Unknown symmetry function group"
                                    " type.\n");
            }
            symmetryFunctionGroups.back()->addMember(*sf);
        }
    }

    sort(symmetryFunctionGroups.begin(),
         symmetryFunctionGroups.end(),
         comparePointerTargets<SymGrp>);

    for (size_t i = 0; i < symmetryFunctionGroups.size(); ++i)
    {
        symmetryFunctionGroups.at(i)->sortMembers();
        symmetryFunctionGroups.at(i)->setIndex(i);
    }

    return;
}

void Element::setupSymmetryFunctionMemory()
{
    symmetryFunctionTable.clear();
    symmetryFunctionTable.resize(elementMap.size());
    for (auto const& s : symmetryFunctions)
    {
        for (size_t i = 0; i < elementMap.size(); ++i)
        {
            if (s->checkRelevantElement(i))
            {
                s->setIndexPerElement(i, symmetryFunctionTable.at(i).size());
                symmetryFunctionTable.at(i).push_back(s->getIndex());
            }
        }
    }
    for (size_t i = 0; i < elementMap.size(); ++i)
    {
        symmetryFunctionNumTable.push_back(symmetryFunctionTable.at(i).size());
    }

    return;
}

vector<string> Element::infoSymmetryFunctionGroups() const
{
    vector<string> v;

    for (vector<SymGrp*>::const_iterator
         it = symmetryFunctionGroups.begin();
         it != symmetryFunctionGroups.end(); ++it)
    {
        vector<string> lines = (*it)->parameterLines();
        v.insert(v.end(), lines.begin(), lines.end());
    }

    return v;
}

void Element::setCutoffFunction(CutoffFunction::CutoffType const cutoffType,
                                double const                     cutoffAlpha)
{
    for (vector<SymFnc*>::const_iterator
         it = symmetryFunctions.begin(); it != symmetryFunctions.end(); ++it)
    {
        SymFncBaseCutoff* sfcb = dynamic_cast<SymFncBaseCutoff*>(*it);
        if (sfcb != nullptr)
        {
            sfcb->setCutoffFunction(cutoffType, cutoffAlpha);
        }
    }

    return;
}

void Element::setScalingNone() const
{
    for (size_t i = 0; i < symmetryFunctions.size(); ++i)
    {
        string scalingLine = strpr("%d %d 0.0 0.0 0.0 0.0",
                                   symmetryFunctions.at(i)->getEc(),
                                   i + 1);
        symmetryFunctions.at(i)->setScalingType(SymFnc::ST_NONE,
                                                scalingLine,
                                                0.0,
                                                0.0);
    }
    for (size_t i = 0; i < symmetryFunctionGroups.size(); ++i)
    {
        symmetryFunctionGroups.at(i)->setScalingFactors();
    }
    
    return;
}

void Element::setScaling(SymFnc::ScalingType scalingType,
                         vector<string> const&         statisticsLine,
                         double                        Smin,
                         double                        Smax) const
{
    for (size_t i = 0; i < symmetryFunctions.size(); ++i)
    {
        symmetryFunctions.at(i)->setScalingType(scalingType,
                                                statisticsLine.at(i),
                                                Smin,
                                                Smax);
    }
    for (size_t i = 0; i < symmetryFunctionGroups.size(); ++i)
    {
        symmetryFunctionGroups.at(i)->setScalingFactors();
    }

    return;
}

size_t Element::getMinNeighbors() const
{
    size_t minNeighbors = 0;

    for (vector<SymFnc*>::const_iterator
         it = symmetryFunctions.begin(); it != symmetryFunctions.end(); ++it)
    {
        minNeighbors = max((*it)->getMinNeighbors(), minNeighbors);
    }

    return minNeighbors;
}

double Element::getMinCutoffRadius() const
{
    double minCutoffRadius = numeric_limits<double>::max();

    // MPB: Hack to work with negative radii
    //      Exploit the fact that all allowed symmetry functions are either
    //      defined for a domain > 0 or have to be symmetric around 0.

    for (vector<SymFnc*>::const_iterator
         it = symmetryFunctions.begin(); it != symmetryFunctions.end(); ++it)
    {
        minCutoffRadius = min((*it)->getRc(), minCutoffRadius);
    }

    return minCutoffRadius;
}

double Element::getMaxCutoffRadius() const
{
    double maxCutoffRadius = 0.0;

    for (vector<SymFnc*>::const_iterator
         it = symmetryFunctions.begin(); it != symmetryFunctions.end(); ++it)
    {
        maxCutoffRadius = max((*it)->getRc(), maxCutoffRadius);
    }

    return maxCutoffRadius;
}

void Element::calculateSymmetryFunctions(Atom&      atom,
                                         bool const derivatives) const
{
    for (vector<SymFnc*>::const_iterator
         it = symmetryFunctions.begin();
         it != symmetryFunctions.end(); ++it)
    {
        //cerr << (*it)->getIndex() << " "
        //     << elementMap[(*it)->getEc()] << " "
        //     << (*it)->getUnique() << "\n";
        //auto cid = (*it)->getCacheIdentifiers();
        //for (auto icid : cid) cerr << icid << "\n";
        //auto ci = (*it)->getCacheIndices();
        //for (auto eci : ci)
        //{
        //    for (auto ici : eci) cerr << ici << " ";
        //    cerr << "\n";
        //}
        (*it)->calculate(atom, derivatives);
    }

    return;
}

void Element::calculateSymmetryFunctionGroups(Atom&      atom,
                                              bool const derivatives) const
{
    for (vector<SymGrp*>::const_iterator
         it = symmetryFunctionGroups.begin();
         it != symmetryFunctionGroups.end(); ++it)
    {
        (*it)->calculate(atom, derivatives);
    }

    return;
}

size_t Element::updateSymmetryFunctionStatistics(Atom const& atom)
{
    size_t countExtrapolationWarnings = 0;
    double epsilon = 1000.0 * numeric_limits<double>::epsilon();

    if (atom.element != index)
    {
        throw runtime_error("ERROR: Atom has a different element index.\n");
    }

    if (atom.numSymmetryFunctions != symmetryFunctions.size())
    {
        throw runtime_error("ERROR: Number of symmetry functions"
                            " does not match.\n");
    }

    for (size_t i = 0; i < atom.G.size(); ++i)
    {
        double const Gmin = symmetryFunctions.at(i)->getGmin();
        double const Gmax = symmetryFunctions.at(i)->getGmax();
        double const value = symmetryFunctions.at(i)->unscale(atom.G.at(i));
        size_t const sfindex = symmetryFunctions.at(i)->getIndex();
        size_t const type = symmetryFunctions.at(i)->getType();
        if (statistics.collectStatistics)
        {
            statistics.addValue(sfindex, atom.G.at(i));
        }

        // Avoid "fake" EWs at the boundaries.
        if (value + epsilon < Gmin || value - epsilon > Gmax)
        {
            countExtrapolationWarnings++;
            if (statistics.collectExtrapolationWarnings)
            {
                statistics.addExtrapolationWarning(sfindex,
                                                   type,
                                                   value,
                                                   Gmin,
                                                   Gmax,
                                                   symbol,
                                                   atom.indexStructure,
                                                   atom.tag);
            }
            if (statistics.writeExtrapolationWarnings)
            {
                cerr << strpr("### NNP EXTRAPOLATION WARNING ### "
                              "STRUCTURE: %6zu ATOM: %9zu ELEMENT: %2s "
                              "SYMFUNC: %4zu TYPE: %2zu VALUE: %10.3E "
                              "MIN: %10.3E MAX: %10.3E\n",
                              atom.indexStructure,
                              atom.tag,
                              symbol.c_str(),
                              sfindex + 1,
                              type,
                              value,
                              Gmin,
                              Gmax);
            }
            if (statistics.stopOnExtrapolationWarnings)
            {
                throw out_of_range(
                        strpr("### NNP EXTRAPOLATION WARNING ### "
                              "STRUCTURE: %6zu ATOM: %9zu ELEMENT: %2s "
                              "SYMFUNC: %4zu TYPE: %2zu VALUE: %10.3E "
                              "MIN: %10.3E MAX: %10.3E\n"
                              "ERROR: Symmetry function value out of range.\n",
                              atom.indexStructure,
                              atom.tag,
                              symbol.c_str(),
                              sfindex + 1,
                              type,
                              value,
                              Gmin,
                              Gmax));
            }
        }
    }

    return countExtrapolationWarnings;
}

#ifndef NNP_NO_SF_CACHE
void Element::setCacheIndices(vector<vector<SFCacheList>> cacheLists)
{
    this->cacheLists = cacheLists;
    for (size_t i = 0; i < cacheLists.size(); ++i)
    {
        for (size_t j = 0; j < cacheLists.at(i).size(); ++j)
        {
            SFCacheList const& c = cacheLists.at(i).at(j);
            for (size_t k = 0; k < c.indices.size(); ++k)
            {
                SymFnc*& sf = symmetryFunctions.at(c.indices.at(k));
                sf->addCacheIndex(c.element, j, c.identifier);
            }
        }
    }

    return;
}

vector<size_t> Element::getCacheSizes() const
{
    vector<size_t> cacheSizes;
    for (auto const& c : cacheLists)
    {
        cacheSizes.push_back(c.size());
    }

    return cacheSizes;
}
#endif
