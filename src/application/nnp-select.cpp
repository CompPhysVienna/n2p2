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

#include "Log.h"
#include "utility.h"
#include <cstddef>  // std::size_t
#include <cstdlib>  // srand, rand
#include <cstring>  // strcmp
#include <iostream> // std::cout
#include <fstream>  // std::ofstream

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        cout << "USAGE: " << argv[0] << " <mode> <arg1 <arg2>>\n"
             << "       <mode> ... Choose selection mode (random/interval).\n"
             << "       Arguments for mode \"random\":\n"
             << "         <arg1> ... Ratio of selected structures "
             << "(1.0 equals 100 %).\n"
             << "         <arg2> ... Seed for random number generator "
             << "(integer).\n"
             << "       Arguments for mode \"interval\":\n"
             << "         <arg1> ... Select structures in this interval "
             << "(integer).\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n";
        return 1;
    }

    bool mode = false; // 0 = random, 1 = interval.
    bool writeStructure = false;
    size_t countStructures = 0;
    size_t countSelected = 0;
    size_t interval = 10;
    int seed = 10;
    double ratio = 1.0;

    Log log;
    ifstream inputFile;
    ofstream outputFile;
    ofstream rejectFile;
    ofstream logFile;

    logFile.open("nnp-select.log");
    log.registerStreamPointer(&logFile);

    log << "\n";
    log << "*** NNP-SELECT **************************"
           "**************************************\n";
    log << "\n";

    if (strcmp(argv[1], "random") == 0)
    {
        mode = false;
        if (argc != 4)
        {
            throw invalid_argument("ERROR: Wrong number of arguments.\n");
        }
        ratio = atof(argv[2]);
        seed = atoi(argv[3]);
        srand(seed);
        log << strpr("Selecting randomly %.2f %% of all structures.\n",
                     ratio * 100.0);
        log << strpr("Random number generator seed: %d.\n", seed);
    }
    else if (strcmp(argv[1], "interval") == 0)
    {
        mode = true; 
        if (argc != 3)
        {
            throw invalid_argument("ERROR: Wrong number of arguments.\n");
        }
        interval = (size_t)atoi(argv[2]);
        log << strpr("Selecting every %d structure.\n", interval);
    }
    else
    {
        throw invalid_argument("ERROR: Unknown selection mode.\n");
    }

    log << "*****************************************"
           "**************************************\n";

    inputFile.open("input.data");
    outputFile.open("output.data");
    rejectFile.open("reject.data");
    string line;
    while (getline(inputFile, line))
    {
        if (split(reduce(line)).at(0) == "begin")
        {
            if (mode)
            {
                if (countStructures % interval == 0) writeStructure = true; 
                else writeStructure = false;
            }
            else
            {
                if ((double)rand() / RAND_MAX <= ratio) writeStructure = true;
                else writeStructure = false;
            }
            countStructures++; 
            if (writeStructure)
            {
                log << strpr("Structure %7d selected.\n", countStructures);
                countSelected++;
            }
        }
        if (writeStructure)
        {
            outputFile << line << '\n';
        }
        else
        {
            rejectFile << line << '\n';
        }
    }
    inputFile.close();
    outputFile.close();
    rejectFile.close();

    log << "*****************************************"
           "**************************************\n";
    log << strpr("Total    structures           : %7d\n", countStructures);
    log << strpr("Selected structures           : %7d\n", countSelected);
    log << strpr("Selected structures percentage: %.3f %%\n",
                 countSelected / (double)countStructures * 100.0);
    log << "*****************************************"
           "**************************************\n";
    logFile.close();

    return 0;
}
