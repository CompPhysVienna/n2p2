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
#include "SymmetryFunction.h"
#include "SymmetryFunctionRadial.h"
#include "SymmetryFunctionAngularNarrow.h"
#include "SymmetryFunctionAngularWide.h"
#include "SymmetryFunctionWeightedRadial.h"
#include "SymmetryFunctionWeightedAngular.h"
#include "SymmetryFunctionGroup.h"
#include "SymmetryFunctionGroupRadial.h"
#include "SymmetryFunctionGroupAngularNarrow.h"
#include "SymmetryFunctionGroupAngularWide.h"
#include "SymmetryFunctionGroupWeightedRadial.h"
#include "SymmetryFunctionGroupWeightedAngular.h"
#include "utility.h"
#include <iostream>  // std::cerr
#include <cstdlib>   // atoi
#include <algorithm> // std::sort, std::min, std::max
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

Element::Element(size_t const index, ElementMap const& elementMap) :
    neuralNetwork     (NULL                          ),
    elementMap        (elementMap                    ),
    index             (index                         ),
    atomicNumber      (elementMap.atomicNumber(index)),
    atomicEnergyOffset(0.0                           ),
    symbol            (elementMap.symbol(index)      )
{
}

Element::~Element()
{
    for (vector<SymmetryFunction*>::const_iterator
         it = symmetryFunctions.begin(); it != symmetryFunctions.end(); ++it)
    {
        delete *it;
    }

    for (vector<SymmetryFunctionGroup*>::const_iterator
         it = symmetryFunctionGroups.begin();
         it != symmetryFunctionGroups.end(); ++it)
    {
        delete *it;
    }

    if (neuralNetwork != NULL)
    {
        delete neuralNetwork;
    }
}

void Element::addSymmetryFunction(string const& parameters,
                                  size_t const& lineNumber)
{
    vector<string> args = split(reduce(parameters));
    size_t         type = (size_t)atoi(args.at(1).c_str());

    if (type == 2)
    {
        symmetryFunctions.push_back(
            new SymmetryFunctionRadial(elementMap));
    }
    else if (type == 3)
    {
        symmetryFunctions.push_back(
            new SymmetryFunctionAngularNarrow(elementMap));
    }
    else if (type == 9)
    {
        symmetryFunctions.push_back(
            new SymmetryFunctionAngularWide(elementMap));
    }
    else if (type == 12)
    {
        symmetryFunctions.push_back(
            new SymmetryFunctionWeightedRadial(elementMap));
    }
    else if (type == 13)
    {
        symmetryFunctions.push_back(
            new SymmetryFunctionWeightedAngular(elementMap));
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
    for (vector<SymmetryFunction*>::iterator it = symmetryFunctions.begin();
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
         comparePointerTargets<SymmetryFunction>);

    for (size_t i = 0; i < symmetryFunctions.size(); ++i)
    {
        symmetryFunctions.at(i)->setIndex(i);
    }

    return;
}

vector<string> Element::infoSymmetryFunctionParameters() const
{
    vector<string> v;

    for (vector<SymmetryFunction*>::const_iterator
         sf = symmetryFunctions.begin(); sf != symmetryFunctions.end(); ++sf)
    {
        v.push_back((*sf)->parameterLine());
    }

    return v;
}

vector<string> Element::infoSymmetryFunctionScaling() const
{
    vector<string> v;

    for (vector<SymmetryFunction*>::const_iterator
         sf = symmetryFunctions.begin(); sf != symmetryFunctions.end(); ++sf)
    {
        v.push_back((*sf)->scalingLine());
    }

    return v;
}

void Element::setupSymmetryFunctionGroups()
{
    for (vector<SymmetryFunction*>::const_iterator
         sf = symmetryFunctions.begin(); sf != symmetryFunctions.end(); ++sf)
    {
        bool createNewGroup = true;
        for (vector<SymmetryFunctionGroup*>::const_iterator
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
                symmetryFunctionGroups.push_back((SymmetryFunctionGroup*)
                    new SymmetryFunctionGroupRadial(elementMap));
            }
            else if ((*sf)->getType() == 3)
            {
                symmetryFunctionGroups.push_back((SymmetryFunctionGroup*)
                    new SymmetryFunctionGroupAngularNarrow(elementMap));
            }
            else if ((*sf)->getType() == 9)
            {
                symmetryFunctionGroups.push_back((SymmetryFunctionGroup*)
                    new SymmetryFunctionGroupAngularWide(elementMap));
            }
            else if ((*sf)->getType() == 12)
            {
                symmetryFunctionGroups.push_back((SymmetryFunctionGroup*)
                    new SymmetryFunctionGroupWeightedRadial(elementMap));
            }
            else if ((*sf)->getType() == 13)
            {
                symmetryFunctionGroups.push_back((SymmetryFunctionGroup*)
                    new SymmetryFunctionGroupWeightedAngular(elementMap));
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
         comparePointerTargets<SymmetryFunctionGroup>);

    for (size_t i = 0; i < symmetryFunctionGroups.size(); ++i)
    {
        symmetryFunctionGroups.at(i)->sortMembers();
        symmetryFunctionGroups.at(i)->setIndex(i);
    }

    return;
}

vector<string> Element::infoSymmetryFunctionGroups() const
{
    vector<string> v;

    for (vector<SymmetryFunctionGroup*>::const_iterator
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
    for (vector<SymmetryFunction*>::const_iterator
         it = symmetryFunctions.begin(); it != symmetryFunctions.end(); ++it)
    {
        (*it)->setCutoffFunction(cutoffType, cutoffAlpha);
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
        symmetryFunctions.at(i)->setScalingType(SymmetryFunction::ST_NONE,
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

void Element::setScaling(SymmetryFunction::ScalingType scalingType,
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

    for (vector<SymmetryFunction*>::const_iterator
         it = symmetryFunctions.begin(); it != symmetryFunctions.end(); ++it)
    {
        minNeighbors = max((*it)->getMinNeighbors(), minNeighbors);
    }

    return minNeighbors;
}

double Element::getMinCutoffRadius() const
{
    double minCutoffRadius = numeric_limits<double>::max();

    for (vector<SymmetryFunction*>::const_iterator
         it = symmetryFunctions.begin(); it != symmetryFunctions.end(); ++it)
    {
        minCutoffRadius = min((*it)->getRc(), minCutoffRadius);
    }

    return minCutoffRadius;
}

double Element::getMaxCutoffRadius() const
{
    double maxCutoffRadius = 0.0;

    for (vector<SymmetryFunction*>::const_iterator
         it = symmetryFunctions.begin(); it != symmetryFunctions.end(); ++it)
    {
        maxCutoffRadius = max((*it)->getRc(), maxCutoffRadius);
    }

    return maxCutoffRadius;
}

void Element::calculateSymmetryFunctions(Atom&      atom,
                                         bool const derivatives) const
{
    for (vector<SymmetryFunction*>::const_iterator
         it = symmetryFunctions.begin();
         it != symmetryFunctions.end(); ++it)
    {
        (*it)->calculate(atom, derivatives);
    }

    return;
}

void Element::calculateSymmetryFunctionGroups(Atom&      atom,
                                              bool const derivatives) const
{
    for (vector<SymmetryFunctionGroup*>::const_iterator
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
    double epsilon = 10.0 * numeric_limits<double>::epsilon();

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
        size_t const index = symmetryFunctions.at(i)->getIndex();
        if (statistics.collectStatistics)
        {
            statistics.addValue(index, atom.G.at(i));
        }

        // Avoid "fake" EWs at the boundaries.
        if (value + epsilon < Gmin || value - epsilon > Gmax)
        {
            countExtrapolationWarnings++;
            if (statistics.collectExtrapolationWarnings)
            {
                statistics.addExtrapolationWarning(index,
                                                   value,
                                                   Gmin,
                                                   Gmax,
                                                   atom.indexStructure,
                                                   atom.tag);
            }
            if (statistics.writeExtrapolationWarnings)
            {
                cerr << strpr("### NNP EXTRAPOLATION WARNING ### "
                              "STRUCTURE: %6zu ATOM: %6zu SYMFUNC: %4zu "
                              "VALUE: %10.3E MIN: %10.3E MAX: %10.3E\n",
                              atom.indexStructure,
                              atom.tag,
                              index,
                              value,
                              Gmin,
                              Gmax);
            }
            if (statistics.stopOnExtrapolationWarnings)
            {
                throw out_of_range(
                        strpr("### NNP EXTRAPOLATION WARNING ### "
                              "STRUCTURE: %6zu ATOM: %6zu SYMFUNC: %4zu "
                              "VALUE: %10.3E MIN: %10.3E MAX: %10.3E\n"
                              "ERROR: Symmetry function value out of range.\n",
                              atom.indexStructure,
                              atom.tag,
                              index,
                              value,
                              Gmin,
                              Gmax));
            }
        }
    }

    return countExtrapolationWarnings;
}
