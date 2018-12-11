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
#include "Log.h"
#include "Structure.h"
#include "utility.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        cout << "USAGE: " << argv[0] << " <format> <elem1 <elem2 ...>>\n"
             << "       <format> ... Structure file output format "
                "(xyz/poscar).\n"
             << "       <elemN> .... Symbol for Nth element.\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n";
        return 1;
    }

    ofstream logFile;
    logFile.open("nnp-convert.log");
    Log log;
    log.registerStreamPointer(&logFile);

    log << "\n";
    log << "*** NNP-CONVERT *************************"
           "**************************************\n";
    log << "\n";

    string format = argv[1];
    log << strpr("Requested file format  : %s\n", format.c_str());
    string outputFileName;
    string outputFileNamePrefix;
    if (format == "xyz")
    {
        outputFileName = "input.xyz";
        log << strpr("Output file name       : %s\n", outputFileName.c_str());
    }
    else if (format == "poscar")
    {
        outputFileNamePrefix = "POSCAR";
        log << strpr("Output file name prefix: %s\n",
                     outputFileNamePrefix.c_str());
    }
    else
    {
        log << "ERROR: Unknown output file format.\n";
        return 1;
    }
    size_t numElements = argc - 2;
    log << strpr("Number of elements     : %zu\n", numElements);
    string elements;
    elements += argv[2];
    for (size_t i = 3; i < numElements + 2; ++i)
    {
        elements += " ";
        elements += argv[i];
    }
    log << strpr("Element string         : %s\n", elements.c_str());
    log << "*****************************************"
           "**************************************\n";

    ElementMap elementMap;
    elementMap.registerElements(elements);

    ifstream inputFile;
    inputFile.open("input.data");
    Structure structure;
    structure.setElementMap(elementMap);

    size_t countStructures = 0;
    ofstream outputFile;
    if (format == "xyz")
    {
        outputFile.open(outputFileName.c_str());
    }

    while (inputFile.peek() != EOF)
    {
        structure.readFromFile(inputFile);
        if (format == "xyz")
        {
            structure.writeToFileXyz(&outputFile);
        }
        else if (format == "poscar")
        {
            outputFileName = strpr("%s_%d",
                                   outputFileNamePrefix.c_str(),
                                   countStructures + 1);
            outputFile.open(outputFileName.c_str());
            structure.writeToFilePoscar(&outputFile, elements);
            outputFile.close();
        }
        countStructures++;
        log << strpr("Configuration %7zu: %7zu atoms\n",
                     countStructures,
                     structure.numAtoms);
        structure.reset();
    }

    if (format == "xyz")
    {
        outputFile.close();
    }

    log << "*****************************************"
           "**************************************\n";
    logFile.close();

    return 0;
}
