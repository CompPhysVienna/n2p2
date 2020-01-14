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
    bool structureInfo = false;

    if (argc != 2)
    {
        cout << "USAGE: " << argv[0] << " <info>\n"
             << "       <info> ... Write structure information to "
                " \"structure.out (0/1)\".\n"
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
    prediction.log << "Writing output files...\n";
    prediction.log << " - energy.out\n";
    ofstream file;
    file.open("energy.out");

    // File header.
    vector<string> title;
    vector<string> colName;
    vector<string> colInfo;
    vector<size_t> colSize;
    title.push_back("Energy comparison.");
    colSize.push_back(16);
    colName.push_back("Ennp");
    colInfo.push_back("Potential energy predicted by NNP.");
    colSize.push_back(16);
    colName.push_back("Eref");
    colInfo.push_back("Reference potential energy.");
    colSize.push_back(16);
    colName.push_back("Ediff");
    colInfo.push_back("Difference between reference and NNP prediction.");
    colSize.push_back(16);
    colName.push_back("E_offset");
    colInfo.push_back("Sum of atomic offset energies "
                      "(included in column Ennp).");
    appendLinesToFile(file,
                      createFileHeader(title, colSize, colName, colInfo));

    file << strpr("%16.8E %16.8E %16.8E %16.8E\n",
                  prediction.structure.energy,
                  prediction.structure.energyRef,
                  prediction.structure.energyRef - prediction.structure.energy,
                  prediction.getEnergyOffset(prediction.structure));
    file.close();

    prediction.log << " - nnatoms.out\n";
    file.open("nnatoms.out");

    // File header.
    title.clear();
    colName.clear();
    colInfo.clear();
    colSize.clear();
    title.push_back("Energy contributions calculated from NNP.");
    colSize.push_back(10);
    colName.push_back("index");
    colInfo.push_back("Atom index.");
    colSize.push_back(2);
    colName.push_back("e");
    colInfo.push_back("Element of atom.");
    colSize.push_back(16);
    colName.push_back("charge");
    colInfo.push_back("Atomic charge (not used).");
    colSize.push_back(16);
    colName.push_back("Ennp_atom");
    colInfo.push_back("Atomic energy contribution (physical units, no mean "
                      "or offset energy added).");
    appendLinesToFile(file,
                      createFileHeader(title, colSize, colName, colInfo));

    for (vector<Atom>::const_iterator it = s.atoms.begin();
         it != s.atoms.end(); ++it)
    {
        file << strpr("%10d %2s %16.8E %16.8E\n",
                      it->index,
                      prediction.elementMap[it->element].c_str(),
                      it->charge,
                      it->energy);
    }
    file.close();

    prediction.log << " - nnforces.out\n";
    file.open("nnforces.out");

    // File header.
    title.clear();
    colName.clear();
    colInfo.clear();
    colSize.clear();
    title.push_back("Atomic force comparison (ordered by atom index).");
    colSize.push_back(16);
    colName.push_back("fx");
    colInfo.push_back("Force in x direction.");
    colSize.push_back(16);
    colName.push_back("fy");
    colInfo.push_back("Force in y direction.");
    colSize.push_back(16);
    colName.push_back("fz");
    colInfo.push_back("Force in z direction.");
    colSize.push_back(16);
    colName.push_back("fxRef");
    colInfo.push_back("Reference force in x direction.");
    colSize.push_back(16);
    colName.push_back("fyRef");
    colInfo.push_back("Reference force in y direction.");
    colSize.push_back(16);
    colName.push_back("fzRef");
    colInfo.push_back("Reference force in z direction.");
    colSize.push_back(16);
    colName.push_back("fxDiff");
    colInfo.push_back("Difference between reference and NNP force in x dir.");
    colSize.push_back(16);
    colName.push_back("fyDiff");
    colInfo.push_back("Difference between reference and NNP force in y dir.");
    colSize.push_back(16);
    colName.push_back("fzDiff");
    colInfo.push_back("Difference between reference and NNP force in z dir.");
    appendLinesToFile(file,
                      createFileHeader(title, colSize, colName, colInfo));

    for (vector<Atom>::const_iterator it = s.atoms.begin();
         it != s.atoms.end(); ++it)
    {
        file << strpr("%16.8E %16.8E %16.8E %16.8E %16.8E "
                      "%16.8E %16.8E %16.8E %16.8E\n",
                      it->f[0],
                      it->f[1],
                      it->f[2],
                      it->fRef[0],
                      it->fRef[1],
                      it->fRef[2],
                      it->fRef[0] - it->f[0],
                      it->fRef[1] - it->f[1],
                      it->fRef[2] - it->f[2]);
    }
    file.close();

    prediction.log << "Writing structure with NNP prediction "
                      "to \"output.data\".\n";
    file.open("output.data");
    prediction.structure.writeToFile(&file, false);
    file.close();

    if (structureInfo)
    {
        prediction.log << "Writing detailed structure information to "
                          "\"structure.out\".\n";
        file.open("structure.out");
        vector<string> info = prediction.structure.info();
        appendLinesToFile(file, info);
        file.close();
    }

    prediction.log << "Finished.\n";
    prediction.log << "*****************************************"
                      "**************************************\n";
    logFile.close();

    return 0;
}
