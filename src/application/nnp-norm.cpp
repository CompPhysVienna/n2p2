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

#include "Dataset.h"
#include "mpi-extra.h"
#include "utility.h"
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    int      numProcs    = 0;
    int      myRank      = 0;
    ofstream myLog;

    if (argc != 1)
    {
        cout << "USAGE: " << argv[0] << "\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - input.nn (NNP settings)\n";
        return 1;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    Dataset dataset;
    if (myRank != 0) dataset.log.writeToStdout = false;
    myLog.open(strpr("nnp-norm.log.%04d", myRank).c_str());
    dataset.log.registerStreamPointer(&myLog);
    dataset.setupMPI();
    dataset.initialize();
    dataset.loadSettingsFile();
    dataset.setupElementMap();
    dataset.setupElements();
    dataset.distributeStructures(false);

    dataset.log << "\n";
    dataset.log << "*** DATA SET NORMALIZATION **************"
                   "**************************************\n";
    dataset.log << "\n";

    if (dataset.settingsKeywordExists("mean_energy") ||
        dataset.settingsKeywordExists("conv_energy") ||
        dataset.settingsKeywordExists("conv_length"))
    {
        throw runtime_error("ERROR: Normalization keywords found in settings, "
                            "please remove them first.\n");
    }

    ofstream fileEvsV;
    fileEvsV.open(strpr("evsv.dat.%04d", myRank).c_str());

    // File header.
    vector<string> title;
    vector<string> colName;
    vector<string> colInfo;
    vector<size_t> colSize;
    title.push_back("Energy vs. volume comparison.");
    colSize.push_back(16);
    colName.push_back("V_atom");
    colInfo.push_back("Volume per atom.");
    colSize.push_back(16);
    colName.push_back("Eref_atom");
    colInfo.push_back("Reference energy per atom.");
    colSize.push_back(10);
    colName.push_back("N");
    colInfo.push_back("Number of atoms.");
    colSize.push_back(16);
    colName.push_back("V");
    colInfo.push_back("Volume of structure.");
    colSize.push_back(16);
    colName.push_back("Eref");
    colInfo.push_back("Reference energy of structure.");
    colSize.push_back(16);
    colName.push_back("Eref_offset");
    colInfo.push_back("Reference energy of structure (including offset).");
    appendLinesToFile(fileEvsV,
                      createFileHeader(title, colSize, colName, colInfo));

    size_t numAtomsTotal = 0;
    size_t numStructures = 0;
    double meanEnergyPerAtom = 0.0;
    double sigmaEnergyPerAtom = 0.0;
    double meanForce = 0.0;
    double sigmaForce = 0.0;
    for (vector<Structure>::const_iterator it = dataset.structures.begin();
         it != dataset.structures.end(); ++it)
    {
        numStructures++;
        numAtomsTotal += it->numAtoms;
        meanEnergyPerAtom += it->energyRef / it->numAtoms;
        fileEvsV << strpr("%16.8E %16.8E %10zu %16.8E %16.8E %16.8E\n",
                          it->volume / it->numAtoms,
                          it->energyRef / it->numAtoms,
                          it->numAtoms,
                          it->volume,
                          it->energyRef,
                          dataset.getEnergyWithOffset(*it, true));
        for (vector<Atom>::const_iterator it2 = it->atoms.begin();
             it2 != it->atoms.end(); ++it2)
        {
            meanForce += it2->fRef[0] + it2->fRef[1] + it2->fRef[2];
        }
    }
    fileEvsV.flush();
    fileEvsV.close();
    MPI_Barrier(MPI_COMM_WORLD);
    dataset.log << "Writing energy/atom vs. volume/atom data "
                << "to \"evsv.dat\".\n";
    if (myRank == 0) dataset.combineFiles("evsv.dat");
    MPI_Allreduce(MPI_IN_PLACE, &numStructures    , 1, MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &numAtomsTotal    , 1, MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanEnergyPerAtom, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanForce        , 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    meanEnergyPerAtom /= numStructures;
    meanForce /= 3 * numAtomsTotal;
    for (vector<Structure>::const_iterator it = dataset.structures.begin();
         it != dataset.structures.end(); ++it)
    {
        double ediff = it->energyRef / it->numAtoms - meanEnergyPerAtom;
        sigmaEnergyPerAtom += ediff * ediff;
        for (vector<Atom>::const_iterator it2 = it->atoms.begin();
             it2 != it->atoms.end(); ++it2)
        {
            double fdiff = it2->fRef[0] - meanForce;
            sigmaForce += fdiff * fdiff;
            fdiff = it2->fRef[1] - meanForce;
            sigmaForce += fdiff * fdiff;
            fdiff = it2->fRef[2] - meanForce;
            sigmaForce += fdiff * fdiff;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &sigmaEnergyPerAtom, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sigmaForce        , 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    sigmaEnergyPerAtom = sqrt(sigmaEnergyPerAtom / (numStructures - 1));
    sigmaForce = sqrt(sigmaForce / (3 * numAtomsTotal - 1));
    dataset.log << "\n";
    dataset.log << strpr("Total number of structures: %zu\n", numStructures);
    dataset.log << strpr("Total number of atoms     : %zu\n", numAtomsTotal);
    dataset.log << strpr("Mean/sigma energy per atom: %16.8E +/- %16.8E\n",
                         meanEnergyPerAtom,
                         sigmaEnergyPerAtom);
    dataset.log << strpr("Mean/sigma force          : %16.8E +/- %16.8E\n",
                         meanForce,
                         sigmaForce);
    double convEnergy = 1.0 / sigmaEnergyPerAtom;
    double convLength = sigmaForce / sigmaEnergyPerAtom;
    dataset.log << strpr("Conversion factor energy  : %24.16E\n", convEnergy);
    dataset.log << strpr("Conversion factor length  : %24.16E\n", convLength);

    ofstream fileCfg;
    fileCfg.open(strpr("output.data.%04d", myRank).c_str());
    for (vector<Structure>::iterator it = dataset.structures.begin();
         it != dataset.structures.end(); ++it)
    {
        it->energyRef = (it->energyRef - meanEnergyPerAtom * it->numAtoms)
                      * convEnergy;
        it->box[0] *= convLength;
        it->box[1] *= convLength;
        it->box[2] *= convLength;
        for (vector<Atom>::iterator it2 = it->atoms.begin();
             it2 != it->atoms.end(); ++it2)
        {
            it2->r *= convLength;
            it2->fRef *= convEnergy / convLength;
        }
        it->writeToFile(&fileCfg);
    }
    fileCfg.flush();
    fileCfg.close();
    MPI_Barrier(MPI_COMM_WORLD);
    dataset.log << "\n";
    dataset.log << "Writing converted data file to \"output.data\".\n";
    dataset.log << "WARNING: This data set is provided for debugging "
                   "purposes only and is NOT intended for training.\n";
    if (myRank == 0) dataset.combineFiles("output.data");

    if (myRank == 0)
    {
        dataset.log << "\n";
        dataset.log << "Writing backup of original settings file to "
                       "\"input.nn.bak\".\n";
        ofstream fileSettings;
        fileSettings.open("input.nn.bak");
        dataset.writeSettingsFile(&fileSettings);
        fileSettings.close();

        dataset.log << "\n";
        dataset.log << "Writing extended settings file to \"input.nn\".\n";
        dataset.log << "Use this settings file for normalized training.\n";
        fileSettings.open("input.nn");
        fileSettings << "#########################################"
                        "######################################\n";
        fileSettings << "# DATA SET NORMALIZATION\n";
        fileSettings << "#########################################"
                        "######################################\n";
        fileSettings << "# This section was automatically added by nnp-norm.\n";
        fileSettings << strpr("mean_energy %24.16E\n", meanEnergyPerAtom);
        fileSettings << strpr("conv_energy %24.16E\n", convEnergy);
        fileSettings << strpr("conv_length %24.16E\n", convLength);
        fileSettings << "#########################################"
                        "######################################\n";
        fileSettings << "\n";
        dataset.writeSettingsFile(&fileSettings);
        fileSettings.close();
    }

    dataset.log << "*****************************************"
                   "**************************************\n";

    myLog.close();

    MPI_Finalize();

    return 0;
}
