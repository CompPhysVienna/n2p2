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

#include "ElementMap.h"
#include "utility.h"
#include <algorithm> // std::sort
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

string const ElementMap::knownElements[] = {
"H" , "He", "Li", "Be", "B" , "C" , "N" , "O" , "F" , "Ne", "Na", "Mg", "Al",
"Si", "P" , "S" , "Cl", "Ar", "K" , "Ca", "Sc", "Ti", "V" , "Cr", "Mn", "Fe",
"Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y" ,
"Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te",
"I" , "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
"Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W" , "Re", "Os", "Ir", "Pt",
"Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
"U" , "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No"
};

size_t ElementMap::registerElements(string const& elementLine)
{
    vector<string> elements = split(reduce(elementLine));

    sort(elements.begin(), elements.end(), compareAtomicNumber);

    if (!forwardMap.empty())
    {
        throw runtime_error("ERROR: Element map not empty.\n");
    }

    for (size_t i = 0; i < elements.size(); i++)
    {
        forwardMap[elements[i]] = i;
        reverseMap[i] = elements[i];
    }

    return forwardMap.size();
}

string ElementMap::getElementsString() const
{
    string elements = "";

    if (size() == 0) return elements;
    elements += symbol(0);
    if (size() == 1) return elements;
    for (size_t i = 1; i < size(); ++i)
    {
        elements += strpr(" %s", symbol(i).c_str());
    }

    return elements;
}

size_t ElementMap::index(string const& symbol) const
{
    return safeFind(forwardMap, symbol);
}

string ElementMap::symbol(size_t const index) const
{
    return safeFind(reverseMap, index);
}

void ElementMap::deregisterElements()
{
    forwardMap.clear();
    reverseMap.clear();

    return;
}

size_t ElementMap::atomicNumber(string const& symbol)
{
    size_t numKnownElements = sizeof(knownElements) / sizeof(*knownElements);

    for (size_t i = 0; i < numKnownElements; i++)
    {
        if (knownElements[i] == symbol)
        {
            return i + 1;
        }
    }

    throw runtime_error("ERROR: Element \"" + symbol + "\" unknown.\n");

    return 0;
}

string ElementMap::symbolFromAtomicNumber(size_t atomicNumber)
{
    size_t numKnownElements = sizeof(knownElements) / sizeof(*knownElements);

    if (atomicNumber >= numKnownElements)
    {
        throw runtime_error(strpr("ERROR: Only the first %zu elements are "
                                  "known to this library.\n",
                                  numKnownElements));
    }
    else if (atomicNumber == 0)
    {
        throw runtime_error("ERROR: Invalid atomic number.\n");
    }

    atomicNumber--;
    return knownElements[atomicNumber];
}

vector<string> ElementMap::info() const
{
    vector<string> v;

    v.push_back(strpr("********************************\n"));
    v.push_back(strpr("ELEMENT MAP                     \n"));
    v.push_back(strpr("********************************\n"));
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("forwardMap                 [*] : %d\n", forwardMap.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (map<string, size_t>::const_iterator it = forwardMap.begin();
         it != forwardMap.end(); ++it)
    {
        v.push_back(strpr("%29s  : %d\n", it->first.c_str(), it->second));
    }
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("reverseMap                 [*] : %d\n", reverseMap.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (map<size_t, string>::const_iterator it = reverseMap.begin();
         it != reverseMap.end(); ++it)
    {
        v.push_back(strpr("%29d  : %s\n", it->first, it->second.c_str()));
    }
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("********************************\n"));

    return v;
}
