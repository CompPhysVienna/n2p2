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

#include "Mode.h"
#include "utility.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdexcept>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    double threshold;
    string pruneMode;

    if (!(argc == 3 || argc == 4))
    {
        cout << "USAGE: " << argv[0] << " <mode> <threshold> <<sensmode>>\n"
             << "       <mode> ........ Use scaling data \"range\" or "
                "sensitivity analysis data \"sensitivity\".\n"
             << "       <threshold> ... Prune symmetry function lines in "
                "settings file below this threshold.\n"
             << "       <sensmode> .... If <mode> is sensitivity, select mean "
                "square average or maximum sensitivity (msa/max).\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.nn (NNP settings)\n"
             << "       - sensitivity.???.out (Data for sensitivity mode).\n";
        return 1;
    }

    pruneMode = argv[1];
    if (! (pruneMode == "range" || pruneMode == "sensitivity"))
    {
        throw invalid_argument("ERROR: Unknown pruning mode.\n");
    }
    threshold = atof(argv[2]);

    ofstream logFile;
    logFile.open("nnp-prune.log");
    Mode mode;
    mode.log.registerStreamPointer(&logFile);
    mode.initialize();
    mode.loadSettingsFile();
    mode.setupGeneric();
    if (pruneMode == "range")
    {
        mode.setupSymmetryFunctionScaling();
    }

    mode.log << "\n";
    mode.log << "*** PRUNE SYMMETRY FUNCTIONS ************"
                "**************************************\n";
    mode.log << "\n";

    vector<size_t> prune;
    if (pruneMode == "range")
    {
        mode.log << "Pruning symmetry functions according to their range.\n";
        mode.log << strpr("Symmetry function threshold for pruning |Gmax - "
                          "Gmin| < %10.2E\n", threshold);
        prune = mode.pruneSymmetryFunctionsRange(threshold);
        mode.log << strpr("Number of symmetry functions to be pruned: %zu\n",
                          prune.size());
        mode.writePrunedSettingsFile(prune, "output-prune-range.nn");
        mode.log << "Pruned settings file written to: output-prune-range.nn\n";
    }
    else if (pruneMode == "sensitivity")
    {
        if (argc != 4)
        {
            throw invalid_argument("ERROR: Please select sensitivity mode.\n");
        }
        string sensMode = argv[3];
        size_t column = 0;
        if (sensMode == "msa")
        {
            column = 1;
            mode.log << strpr("Pruning symmetry functions below mean square "
                              "average sensitivity threshold: %10.2E.\n",
                              threshold);
        }
        else if (sensMode == "max")
        {
            column = 2;
            mode.log << strpr("Pruning symmetry functions below maximum "
                              "sensitivity threshold: %10.2E.\n",
                              threshold);
        }
        else
        {
            throw invalid_argument("ERROR: Unknown sensitivity mode.\n");
        }
        vector<vector<double> > sensitivity;
        size_t numElements = mode.getNumElements();
        vector<size_t> numSymmetryFunctions = mode.getNumSymmetryFunctions();
        sensitivity.resize(numElements);
        for (size_t i = 0; i < numElements; ++i)
        {
            ifstream sensFile;
            string sensFileName = strpr("sensitivity.%03d.out",
                                        mode.elementMap.atomicNumber(i));
            sensFile.open(sensFileName.c_str());
            string line;
            vector<string> lines;
            while (getline(sensFile, line))
            {
                if (line.at(0) != '#')
                {
                    lines.push_back(line);
                }
            }
            if (lines.size() != numSymmetryFunctions.at(i))
            {
                throw range_error(strpr("ERROR: Inconsistent sensistivity"
                                        " data file \"%s\".\n",
                                        sensFileName.c_str()));
            }
            for (size_t j = 0; j < numSymmetryFunctions.at(i); ++j)
            {
                vector<string> split = nnp::split(reduce(lines.at(j)));
                double s = atof(split.at(column).c_str()); 
                sensitivity.at(i).push_back(s);
            }
            sensFile.close();
        }
        prune = mode.pruneSymmetryFunctionsSensitivity(threshold, sensitivity);
        mode.log << strpr("Number of symmetry functions to be pruned: %zu\n",
                          prune.size());
        mode.writePrunedSettingsFile(prune, "output-prune-sensitivity.nn");
        mode.log << "Pruned settings file written to: "
                    "output-prune-sensitivity.nn\n";
    }

    mode.log << "*****************************************"
                "**************************************\n";
    logFile.close();

    return 0;
}
