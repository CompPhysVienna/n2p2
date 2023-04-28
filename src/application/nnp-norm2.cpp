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

#include "Training.h"
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
    int      numProcs = 0;
    int      myRank   = 0;
    size_t   stage    = 0;
    ofstream myLog;

    if (argc > 2)
    {
        cout << "USAGE: " << argv[0] << " <stage>\n"
             << "       <stage> ... Training stage (only required if training"
                " is a multi-stage process.\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - input.nn (NNP settings)\n"
             << "       - scaling.data (symmetry function scaling data)\n"
             << "       - \"weights(e).%%03d.data\" (weights files)\n";
        return 1;
    }

    string suffix = "";
    if (argc > 1)
    {
        stage = (size_t)atoi(argv[1]);
        suffix = strpr(".stage-%zu", stage);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    Training training;
    if (myRank != 0) training.log.writeToStdout = false;
    myLog.open((strpr("nnp-norm2.log.%04d", myRank) + suffix).c_str());
    training.log.registerStreamPointer(&myLog);
    training.setupMPI();
    training.initialize();
    training.loadSettingsFile();
    training.setStage(stage);
    training.setupGeneric();
    training.setupSymmetryFunctionScaling();
    training.setupSymmetryFunctionStatistics(false, false, false, false);

    auto nnpType = training.getNnpType();
    if ( (nnpType == Training::NNPType::HDNNP_4G ||
          nnpType == Training::NNPType::HDNNP_Q) && stage == 1)
    {
        throw runtime_error("ERROR: Normalization of charges not yet "
                            "implemented\n.");
    }

    if (training.settingsKeywordExists("mean_energy") ||
        training.settingsKeywordExists("conv_energy") ||
        training.settingsKeywordExists("conv_length"))
    {
        throw runtime_error("ERROR: Normalization keywords found in settings, "
                            "please remove them first.\n");
    }

    if (!training.settingsKeywordExists("use_short_forces"))
    {
        throw runtime_error("ERROR: Normalization is only possible if forces "
                            "are used (keyword \"use_short_forces\").\n");
    }

    // Need RNG for initial random weights.
    training.setupRandomNumberGenerator();

    // Distribute structures to MPI processes.
    training.distributeStructures(false);

    // Initialize weights and biases for neural networks.
    training.initializeWeights();

    training.writeWeights("short", "weights.%03zu.norm");
    if ( (nnpType == Training::NNPType::HDNNP_4G ||
          nnpType == Training::NNPType::HDNNP_Q) && stage == 2)
    {
        training.writeWeights("charge", "weightse.%03zu.norm");
    }

    training.log << "\n";
    training.log << "*** DATA SET NORMALIZATION **************"
                    "**************************************\n";
    training.log << "\n";

    ofstream fileEvsV;
    fileEvsV.open(strpr("evsv.dat.%04d", myRank).c_str());
    if (myRank == 0)
    {
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
    }

    size_t numAtomsTotal         = 0;
    size_t numStructures         = 0;
    double meanEnergyPerAtomRef  = 0.0;
    double meanEnergyPerAtomNnp  = 0.0;
    double sigmaEnergyPerAtomRef = 0.0;
    double sigmaEnergyPerAtomNnp = 0.0;
    double meanForceRef          = 0.0;
    double meanForceNnp          = 0.0;
    double sigmaForceRef         = 0.0;
    double sigmaForceNnp         = 0.0;
    training.log << "Computing initial prediction for all structures...\n";
    for (auto& s : training.structures)
    {
        // File output for evsv.dat.
        fileEvsV << strpr("%16.8E %16.8E %10zu %16.8E %16.8E %16.8E\n",
                          s.volume / s.numAtoms,
                          s.energyRef / s.numAtoms,
                          s.numAtoms,
                          s.volume,
                          s.energyRef,
                          training.getEnergyWithOffset(s, true));
        s.calculateNeighborList(training.getMaxCutoffRadius());
#ifdef N2P2_NO_SF_GROUPS
        training.calculateSymmetryFunctions(s, true);
#else
        training.calculateSymmetryFunctionGroups(s, true);
#endif
        training.calculateAtomicNeuralNetworks(s, true);
        training.calculateEnergy(s);
        training.calculateForces(s);
        s.clearNeighborList();

        numStructures++;
        numAtomsTotal += s.numAtoms;
        meanEnergyPerAtomRef += s.energyRef / s.numAtoms;
        meanEnergyPerAtomNnp += s.energy    / s.numAtoms;
        for (auto& a : s.atoms)
        {
            meanForceRef += a.fRef[0] + a.fRef[1] + a.fRef[2];
            meanForceNnp += a.f   [0] + a.f   [1] + a.f   [2];
        }
    }

    fileEvsV.flush();
    fileEvsV.close();
    MPI_Barrier(MPI_COMM_WORLD);
    training.log << "Writing energy/atom vs. volume/atom data "
                 << "to \"evsv.dat\".\n";
    if (myRank == 0) training.combineFiles("evsv.dat");
    MPI_Allreduce(MPI_IN_PLACE, &numStructures       , 1, MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &numAtomsTotal       , 1, MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanEnergyPerAtomRef, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanEnergyPerAtomNnp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanForceRef        , 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &meanForceNnp        , 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    meanEnergyPerAtomRef /= numStructures;
    meanEnergyPerAtomNnp /= numStructures;
    meanForceRef /= 3 * numAtomsTotal;
    meanForceNnp /= 3 * numAtomsTotal;
    for (auto const& s : training.structures)
    {
        double ediffRef = s.energyRef / s.numAtoms - meanEnergyPerAtomRef;
        double ediffNnp = s.energy    / s.numAtoms - meanEnergyPerAtomNnp;
        sigmaEnergyPerAtomRef += ediffRef * ediffRef;
        sigmaEnergyPerAtomNnp += ediffNnp * ediffNnp;
        for (auto const& a : s.atoms)
        {
            double fdiffRef = a.fRef[0] - meanForceRef;
            double fdiffNnp = a.f   [0] - meanForceNnp;
            sigmaForceRef += fdiffRef * fdiffRef;
            sigmaForceNnp += fdiffNnp * fdiffNnp;
            fdiffRef = a.fRef[1] - meanForceRef;
            fdiffNnp = a.f   [1] - meanForceNnp;
            sigmaForceRef += fdiffRef * fdiffRef;
            sigmaForceNnp += fdiffNnp * fdiffNnp;
            fdiffRef = a.fRef[2] - meanForceRef;
            fdiffNnp = a.f   [2] - meanForceNnp;
            sigmaForceRef += fdiffRef * fdiffRef;
            sigmaForceNnp += fdiffNnp * fdiffNnp;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &sigmaEnergyPerAtomRef, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sigmaEnergyPerAtomNnp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sigmaForceRef        , 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sigmaForceNnp        , 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    sigmaEnergyPerAtomRef = sqrt(sigmaEnergyPerAtomRef / (numStructures - 1));
    sigmaEnergyPerAtomNnp = sqrt(sigmaEnergyPerAtomNnp / (numStructures - 1));
    sigmaForceRef = sqrt(sigmaForceRef / (3 * numAtomsTotal - 1));
    sigmaForceNnp = sqrt(sigmaForceNnp / (3 * numAtomsTotal - 1));
    training.log << "\n";
    training.log << strpr("Total number of structures : %zu\n", numStructures);
    training.log << strpr("Total number of atoms      : %zu\n", numAtomsTotal);
    training.log << "----------------------------------\n";
    training.log << "Reference data statistics:\n";
    training.log << "----------------------------------\n";
    training.log << strpr("Mean/sigma energy per atom : %16.8E +/- %16.8E\n",
                         meanEnergyPerAtomRef,
                         sigmaEnergyPerAtomRef);
    training.log << strpr("Mean/sigma force           : %16.8E +/- %16.8E\n",
                         meanForceRef,
                         sigmaForceRef);
    training.log << "----------------------------------\n";
    training.log << "Initial NNP prediction statistics:\n";
    training.log << "----------------------------------\n";
    training.log << strpr("Mean/sigma energy per atom : %16.8E +/- %16.8E\n",
                         meanEnergyPerAtomNnp,
                         sigmaEnergyPerAtomNnp);
    training.log << strpr("Mean/sigma force           : %16.8E +/- %16.8E\n",
                         meanForceNnp,
                         sigmaForceNnp);
    training.log << "----------------------------------\n";
    double convEnergy = sigmaForceNnp / sigmaForceRef;
    double convLength = sigmaForceNnp;
    training.log << strpr("Conversion factor energy   : %24.16E\n", convEnergy);
    training.log << strpr("Conversion factor length   : %24.16E\n", convLength);

    //ofstream fileCfg;
    //fileCfg.open(strpr("output.data.%04d", myRank).c_str());
    //for (vector<Structure>::iterator it = training.structures.begin();
    //     it != training.structures.end(); ++it)
    //{
    //    it->energyRef = (it->energyRef - meanEnergyPerAtom * it->numAtoms)
    //                  * convEnergy;
    //    it->box[0] *= convLength;
    //    it->box[1] *= convLength;
    //    it->box[2] *= convLength;
    //    for (vector<Atom>::iterator it2 = it->atoms.begin();
    //         it2 != it->atoms.end(); ++it2)
    //    {
    //        it2->r *= convLength;
    //        it2->fRef *= convEnergy / convLength;
    //    }
    //    it->writeToFile(&fileCfg);
    //}
    //fileCfg.flush();
    //fileCfg.close();
    //MPI_Barrier(MPI_COMM_WORLD);
    //training.log << "\n";
    //training.log << "Writing converted data file to \"output.data\".\n";
    //training.log << "WARNING: This data set is provided for debugging "
    //               "purposes only and is NOT intended for training.\n";
    //if (myRank == 0) training.combineFiles("output.data");

    if (myRank == 0)
    {
        training.log << "\n";
        training.log << "Writing backup of original settings file to "
                       "\"input.nn.bak\".\n";
        ofstream fileSettings;
        fileSettings.open("input.nn.bak");
        training.writeSettingsFile(&fileSettings);
        fileSettings.close();

        training.log << "\n";
        training.log << "Writing extended settings file to \"input.nn\".\n";
        training.log << "Use this settings file for normalized training.\n";
        fileSettings.open("input.nn");
        fileSettings << "#########################################"
                        "######################################\n";
        fileSettings << "# DATA SET NORMALIZATION\n";
        fileSettings << "#########################################"
                        "######################################\n";
        fileSettings << strpr("mean_energy %24.16E # nnp-norm2\n",
                              meanEnergyPerAtomRef);
        fileSettings << strpr("conv_energy %24.16E # nnp-norm2\n", convEnergy);
        fileSettings << strpr("conv_length %24.16E # nnp-norm2\n", convLength);
        fileSettings << "#########################################"
                        "######################################\n";
        fileSettings << "\n";
        training.writeSettingsFile(&fileSettings);
        fileSettings.close();
    }

    training.log << "*****************************************"
                    "**************************************\n";

    myLog.close();

    MPI_Finalize();

    return 0;
}
