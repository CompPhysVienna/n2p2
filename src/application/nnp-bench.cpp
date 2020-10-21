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
#include "Stopwatch.h"
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
    bool structureInfo = false;

    if (argc != 2)
    {
        cout << "USAGE: " << argv[0] << " <info>\n"
             << "       <info> ... Write structure information for debugging "
                "to \"structure.out (0/1)\".\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - input.nn (NNP settings)\n"
             << "       - scaling.data (symmetry function scaling data)\n"
             << "       - \"weights.%%03d.data\" (weights files)\n";
        return 1;
    }

    structureInfo = (bool)atoi(argv[1]);

    ofstream logFile;
    logFile.open("nnp-predict.log");
    Prediction prediction;
    prediction.log.registerStreamPointer(&logFile);
    prediction.setup();
    prediction.setupSymmetryFunctionStatistics(false, false, false, false);
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
    //prediction.predict();
    s.calculateNeighborList(prediction.getMaxCutoffRadius());
    Stopwatch sw;
    for (size_t i = 0; i < 10; ++i)
    {
        sw.start();
#ifndef NOSFGROUPS
        prediction.calculateSymmetryFunctionGroups(s, true);
#else
        prediction.calculateSymmetryFunctions(s, true);
#endif
        sw.stop();
        s.freeAtoms(true);
    }
    prediction.log << strpr("time = %24.16E\n", sw.getTimeElapsed());
    //prediction.calculateAtomicNeuralNetworks(s, true);
    //prediction.calculateEnergy(s);
    //prediction.calculateForces(s);
    prediction.log << "\n";
    prediction.log << "-----------------------------------------"
                      "--------------------------------------\n";
    logFile.close();

    return 0;
}
