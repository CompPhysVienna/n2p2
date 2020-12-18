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
#include "Prediction.h"
#include "Structure.h"
#include "utility.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    structureInfo = (bool)atoi(argv[1]);

    ofstream logFile;
    logFile.open("nnp-predict.log");
    Prediction prediction;
    prediction.log.registerStreamPointer(&logFile);
    prediction.setup();
    prediction.log << "\n";
    prediction.log << "*** PREDICTION **************************"
                      "**************************************\n";
    prediction.log << "\n";
    prediction.log << "Reading structure file...\n";
    prediction.readStructureFromFile("input.data");
    Structure& s = prediction.structure;
    prediction.log << strpr("Structure contains %d atoms (%d elements).\n",
                            s.numAtoms, s.numElements);
    prediction.log << "Calculating NNP prediction...\n";
    prediction.predict();
    prediction.log << "\n";
    prediction.log << "-----------------------------------------"
                      "--------------------------------------\n";
    prediction.log << strpr("NNP energy: %16.8E\n",
                            prediction.structure.energy);
    prediction.log << "\n";
    prediction.log << "NNP forces:\n";
    for (vector<Atom>::const_iterator it = s.atoms.begin();
         it != s.atoms.end(); ++it)
    {
        prediction.log << strpr("%10zu %2s %16.8E %16.8E %16.8E\n",
                                it->index + 1,
                                prediction.elementMap[it->element].c_str(),
                                it->element,
                                it->f[0],
                                it->f[1],
                                it->f[2]);
    }
    prediction.log << "-----------------------------------------"
                      "--------------------------------------\n";
    prediction.log << "Finished.\n";
    prediction.log << "*****************************************"
                      "**************************************\n";
    logFile.close();

    return 0;
}
