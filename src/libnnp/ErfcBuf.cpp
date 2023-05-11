//
// Created by philipp on 5/23/22.
//

#include "ErfcBuf.h"

using namespace std;
using namespace nnp;

void ErfcBuf::reset(std::vector<Atom> const& atoms, size_t const valuesPerPair)
{
    f.resize(atoms.size());
    for(size_t i = 0; i < atoms.size(); ++i)
    {
        f[i].resize(atoms[i].numNeighbors * valuesPerPair);
        fill(f[i].begin(), f[i].end(), -1.0);
    }
    numValuesPerPair = valuesPerPair;
}

double ErfcBuf::getf(const size_t atomIndex,
                     const size_t neighIndex,
                     const size_t valIndex,
                     const double x)
{
    size_t j = neighIndex * numValuesPerPair + valIndex;
    if (f.at(atomIndex).at(j) == -1.0)
    {
        f[atomIndex][j] = erfc(x);
    }
    return f[atomIndex][j];
}