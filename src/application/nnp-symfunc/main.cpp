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

#include "SetupAnalysis.h"
#include "utility.h"
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    size_t numPoints = 0;
    ofstream logFile;

    if (argc != 2)
    {
        cout << "USAGE: " << argv[0] << " <npoints>\n"
                "        <npoints> ... Number of data points in symmetry"
                " function files.\n"
                "       Execute in directory with these NNP files present:\n"
             << "       - input.nn (NNP settings)\n";
        return 1;
    }

    numPoints = atof(argv[1]);
    
    logFile.open("nnp-symfunc.log");
    SetupAnalysis setupAnalysis;
    setupAnalysis.log.registerStreamPointer(&logFile);

    setupAnalysis.initialize();
    setupAnalysis.loadSettingsFile();
    setupAnalysis.setupElementMap();
    setupAnalysis.setupElements();
    setupAnalysis.setupCutoff();
    setupAnalysis.setupSymmetryFunctions();

    setupAnalysis.writeSymmetryFunctionShape(numPoints);

    logFile.close();

    return 0;
}
