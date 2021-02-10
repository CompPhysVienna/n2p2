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

#include "SymFncStatistics.h"
#include <algorithm> // std::min, std::max
#include <limits>    // std::numeric_limits
#include "utility.h"

using namespace std;
using namespace nnp;

SymFncStatistics::Container::Container() :
    count  (0                             ),
    countEW(0                             ),
    type   (0                             ),
    min    ( numeric_limits<double>::max()),
    max    (-numeric_limits<double>::max()),
    Gmin   (0.0                           ),
    Gmax   (0.0                           ),
    sum    (0.0                           ),
    sum2   (0.0                           ),
    element(""                            )
{
}

void SymFncStatistics::Container::reset()
{
    resetStatistics();
    resetExtrapolationWarnings();

    return;
}

void SymFncStatistics::Container::resetStatistics()
{
    count = 0;
    min   =  numeric_limits<double>::max();
    max   = -numeric_limits<double>::max();
    sum   = 0.0;
    sum2  = 0.0;

    return;
}

void SymFncStatistics::Container::resetExtrapolationWarnings()
{
    countEW = 0;
    Gmin = 0.0;
    Gmax = 0.0;
    type = 0;
    element = "";
    indexStructureEW.clear();
    indexAtomEW.clear();
    valueEW.clear();

    return;
}

SymFncStatistics::SymFncStatistics() :
    collectStatistics           (false),
    collectExtrapolationWarnings(false),
    writeExtrapolationWarnings  (false),
    stopOnExtrapolationWarnings (false)
{
}

void SymFncStatistics::addValue(size_t index, double value)
{
    data[index].count++;
    data[index].sum += value;
    data[index].sum2 += value * value;
    data[index].min = min(data[index].min, value);
    data[index].max = max(data[index].max, value);

    return;
}

void SymFncStatistics::addExtrapolationWarning(size_t index,
                                               size_t type,
                                               double value,
                                               double Gmin,
                                               double Gmax,
                                               string element,
                                               size_t indexStructure,
                                               size_t indexAtom)
{
    data[index].countEW++;
    data[index].Gmin = Gmin;
    data[index].Gmax = Gmax;
    data[index].type = type;
    data[index].element = element;
    data[index].valueEW.push_back(value);
    data[index].indexStructureEW.push_back(indexStructure);
    data[index].indexAtomEW.push_back(indexAtom);

    return;
}

vector<string> SymFncStatistics::getExtrapolationWarningLines() const
{
    vector<string> vs;
    for (map<size_t, Container>::const_iterator it = data.begin();
         it != data.end(); ++it)
    {
        SymFncStatistics::Container const& d = it->second;
        for (size_t i = 0; i < d.valueEW.size(); ++i)
        {
            vs.push_back(strpr("### NNP EXTRAPOLATION WARNING ### "
                               "STRUCTURE: %6zu ATOM: %9zu ELEMENT: %2s "
                               "SYMFUNC: %4zu TYPE: %2zu VALUE: %10.3E "
                               "MIN: %10.3E MAX: %10.3E\n",
                               d.indexStructureEW[i],
                               d.indexAtomEW[i],
                               d.element.c_str(),
                               it->first + 1,
                               d.type,
                               d.valueEW[i],
                               d.Gmin,
                               d.Gmax));
        }
    }

    return vs;
}

size_t SymFncStatistics::countExtrapolationWarnings() const
{
    size_t n = 0;

    for (map<size_t, Container>::const_iterator it = data.begin();
         it != data.end(); ++it)
    {
        n += it->second.countEW;
    }

    return n;
}

void SymFncStatistics::resetStatistics()
{
    for (map<size_t, Container>::iterator it = data.begin();
         it != data.end(); ++it)
    {
        it->second.resetStatistics();
    }

    return;
}

void SymFncStatistics::resetExtrapolationWarnings()
{
    for (map<size_t, Container>::iterator it = data.begin();
         it != data.end(); ++it)
    {
        it->second.resetExtrapolationWarnings();
    }

    return;
}

void SymFncStatistics::clear()
{
    data.clear();
    return;
}
