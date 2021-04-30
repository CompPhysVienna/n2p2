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
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace nnp;

double const massH = 1.00784;
double const massO = 15.99903;
double const massCG = massO + 2.0 * massH;
string const elements = "H O";
string const elementCG = "Cm";

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        cout << "USAGE: " << argv[0] << " <rcut>\n"
             << "       <rcut> .... Cutoff radius.\n";
        return 1;
    }

    double const cutoffRadius = atof(argv[1]);

    ofstream logFile;
    logFile.open("nnp-cgdata.log");
    Log log;
    log.registerStreamPointer(&logFile);

    log << "\n";
    log << "*** NNP-CGDATA **************************"
           "**************************************\n";
    log << strpr("Cutoff radius: %f\n", cutoffRadius);
    log << "\n";


    ifstream inputFile;
    inputFile.open("input.data");
    ElementMap elementMap;
    elementMap.registerElements(elements);
    Structure structure;
    structure.setElementMap(elementMap);

    ofstream outputFile;
    outputFile.open("output.data");
    ElementMap elementMapCG;
    elementMapCG.registerElements(elementCG);
    Structure structureCG;
    structureCG.setElementMap(elementMapCG);

    size_t countStructures = 0;
    while (inputFile.peek() != EOF)
    {
        structure.readFromFile(inputFile);
        structure.calculateNeighborList(cutoffRadius);
        // Copy structure properties.
        structureCG.comment = structure.comment;
        structureCG.isPeriodic = structure.isPeriodic;
        structureCG.isTriclinic = structure.isTriclinic;
        structureCG.volume = structure.volume;
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                structureCG.box[i][j] = structure.box[i][j];
                structureCG.invbox[i][j] = structure.invbox[i][j];
            }
        }
        // Set atom tag to zero (used here for duplicate hydrogen usage check.
        for (vector<Atom>::iterator it = structure.atoms.begin();
             it != structure.atoms.end(); ++it)
        {
            it->tag = 0;
        }
        // Main loop over atoms.
        for (vector<Atom>::iterator it = structure.atoms.begin();
             it != structure.atoms.end(); ++it)
        {
            if (elementMap[it->element] == "O")
            {
                vector<Atom::Neighbor const*> hn;
                sort(it->neighbors.begin(), it->neighbors.end());
                for (size_t i = 0; i < it->numNeighbors; ++i)
                {
                    Atom::Neighbor& n = it->neighbors.at(i);
                    if (elementMap[n.element] == "H")
                    {
                        hn.push_back(&n);
                    }
                }
                if (hn.size() != 2)
                {
                    throw runtime_error(strpr("ERROR: Could not find two "
                                              "hydrogen neighbors (structure "
                                              "%zu, atom %zu, #Hs = %zu!",
                                              countStructures,
                                              it->index,
                                              hn.size()));
                }
                vector<Atom*> ha;
                // Check if a hydrogen atom is already part of a molecule.
                for (size_t i = 0; i < hn.size(); ++i)
                {
                    ha.push_back(&structure.atoms.at(hn[i]->index));
                    if (ha.back()->tag > 0)
                    {
                        throw runtime_error(strpr("ERROR: Hydrogen atom %zu ("
                                                  "structure %zu) is already "
                                                  "part of another "
                                                  "molecule!\n",
                                                  ha.back()->index,
                                                  countStructures));
                    }
                }
                Atom atomCG;
                atomCG.r = it->r
                         - 2.0 * massH * (hn[0]->dr + hn[1]->dr) / massCG;
                if (structureCG.isPeriodic) structureCG.remap(atomCG);
                atomCG.fRef = it->fRef + ha[0]->fRef + ha[1]->fRef;
                structureCG.addAtom(atomCG, elementCG);
                // Increment hydrogen counter.
                ha[0]->tag++;
                ha[1]->tag++;
            }
        }
        structureCG.numElementsPresent = 1;
        structureCG.writeToFile(&outputFile, true);
        countStructures++;
        log << strpr("Configuration %7zu: %7zu atoms\n",
                     countStructures,
                     structure.numAtoms);
        structure.reset();
        structureCG.reset();
    }
    outputFile.close();


    log << "*****************************************"
           "**************************************\n";
    logFile.close();

    return 0;
}
