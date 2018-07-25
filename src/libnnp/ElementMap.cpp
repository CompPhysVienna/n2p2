// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
        throw runtime_error("ERROR: Atomic number too high.\n");
    }

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
