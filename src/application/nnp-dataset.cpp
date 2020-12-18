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
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    bool                shuffle         = false;
    bool                useForces       = false;
    bool                normalize       = false;
    int                 numProcs        = 0;
    int                 myRank          = 0;
    size_t              countEnergy     = 0;
    size_t              countForces     = 0;
    map<string, double> errorEnergy;
    map<string, double> errorForces;
    string              fileName;
    ofstream            fileEnergy;
    ofstream            fileForces;
    ofstream            fileOutputData;
    ofstream            myLog;

    errorEnergy["RMSEpa"] = 0.0;
    errorEnergy["RMSE"] = 0.0;
    errorEnergy["MAEpa"] = 0.0;
    errorEnergy["MAE"] = 0.0;
    errorForces["RMSE"] = 0.0;
    errorForces["MAE"] = 0.0;

    if (argc != 2)
    {
        cout << "USAGE: " << argv[0] << " <shuffle>\n"
             << "       <shuffle> ... Randomly distribute structures to MPI"
                " processes (0/1 = no/yes).\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - input.nn (NNP settings)\n"
             << "       - scaling.data (symmetry function scaling data)\n"
             << "       - \"weights.%%03d.data\" (weights files)\n";
        return 1;
    }

    shuffle = (bool)atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    Dataset dataset;
    if (myRank != 0) dataset.log.writeToStdout = false;
    myLog.open(strpr("nnp-dataset.log.%04d", myRank).c_str());
    dataset.log.registerStreamPointer(&myLog);
    dataset.setupMPI();
    dataset.initialize();
    dataset.loadSettingsFile();
    dataset.setupGeneric();
    normalize = dataset.useNormalization();
    dataset.setupSymmetryFunctionScaling();
    dataset.setupSymmetryFunctionStatistics(false, false, true, false);
    dataset.setupNeuralNetworkWeights();
    if (shuffle) dataset.setupRandomNumberGenerator();
    dataset.distributeStructures(shuffle);
    if (normalize) dataset.toNormalizedUnits();

    dataset.log << "\n";
    dataset.log << "*** DATA SET PREDICTION *****************"
                   "**************************************\n";
    dataset.log << "\n";

    useForces = dataset.settingsKeywordExists("use_short_forces");
    if (useForces)
    {
        dataset.log << "Energies and forces are predicted.\n";
    }
    else
    {
        dataset.log << "Only energies are predicted.\n";
    }

    // Set up sensitivity vectors.
    size_t numElements = dataset.getNumElements();
    vector<size_t> numSymmetryFunctions = dataset.getNumSymmetryFunctions();
    vector<size_t> count(numElements, 0);
    vector<vector<double> > sensMean;
    vector<vector<double> > sensMax;
    sensMean.resize(numElements);
    sensMax.resize(numElements);
    for (size_t i = 0; i < numElements; ++i)
    {
        sensMean.at(i).resize(numSymmetryFunctions.at(i), 0.0);
        sensMax.at(i).resize(numSymmetryFunctions.at(i), 0.0);
    }

    // Set up error files for energy and forces RMSEs.
    fileName = strpr("energy.comp.%04d", myRank);
    fileEnergy.open(fileName.c_str());

    // File header.
    if (myRank == 0)
    {
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back("Energy comparison.");
        colSize.push_back(10);
        colName.push_back("index");
        colInfo.push_back("Structure index.");
        colSize.push_back(10);
        colName.push_back("N");
        colInfo.push_back("Number of atoms in structure.");
        colSize.push_back(16);
        colName.push_back("Eref_phys");
        colInfo.push_back("Reference potential energy (physical units, "
                          "atomic energy offsets added).");
        colSize.push_back(16);
        colName.push_back("Ennp_phys");
        colInfo.push_back("NNP potential energy (physical units, "
                          "atomic energy offsets added).");
        if (normalize)
        {
            colSize.push_back(16);
            colName.push_back("Eref_int");
            colInfo.push_back("Reference potential energy (internal units).");
            colSize.push_back(16);
            colName.push_back("Ennp_int");
            colInfo.push_back("NNP potential energy (internal units).");
        }
        appendLinesToFile(fileEnergy,
                          createFileHeader(title, colSize, colName, colInfo));
    }
    if (useForces)
    {
        fileName = strpr("forces.comp.%04d", myRank);
        fileForces.open(fileName.c_str());

        // File header.
        if (myRank == 0)
        {
            vector<string> title;
            vector<string> colName;
            vector<string> colInfo;
            vector<size_t> colSize;
            title.push_back("Force comparison.");
            colSize.push_back(10);
            colName.push_back("index_s");
            colInfo.push_back("Structure index.");
            colSize.push_back(10);
            colName.push_back("index_a");
            colInfo.push_back("Atom index (x, y, z components in consecutive "
                              "lines).");
            colSize.push_back(16);
            colName.push_back("Fref_phys");
            colInfo.push_back("Reference force (physical units).");
            colSize.push_back(16);
            colName.push_back("Fnnp_phys");
            colInfo.push_back("NNP force (physical units).");
            if (normalize)
            {
                colSize.push_back(16);
                colName.push_back("Fref_int");
                colInfo.push_back("Reference force (internal units).");
                colSize.push_back(16);
                colName.push_back("Fnnp_int");
                colInfo.push_back("NNP force (internal units).");
            }
            appendLinesToFile(fileForces,
                              createFileHeader(title,
                                               colSize,
                                               colName,
                                               colInfo));
        }
    }

    // Open output.data file.
    fileName = strpr("output.data.%04d", myRank);
    fileOutputData.open(fileName.c_str());

    for (vector<Structure>::iterator it = dataset.structures.begin();
         it != dataset.structures.end(); ++it)
    {
        it->calculateNeighborList(dataset.getMaxCutoffRadius());
#ifdef NNP_NO_SF_GROUPS
        dataset.calculateSymmetryFunctions((*it), useForces);
#else
        dataset.calculateSymmetryFunctionGroups((*it), useForces);
#endif
        // Manually allocate dEdG vectors.
        for (vector<Atom>::iterator it2 = it->atoms.begin();
             it2 != it->atoms.end(); ++it2)
        {
            size_t const& e = it2->element;
            it2->dEdG.resize(numSymmetryFunctions.at(e), 0.0);
        }
        // Set derivatives argument to true in any case to fill dEdG vectors
        // in atom storage.
        dataset.calculateAtomicNeuralNetworks((*it), true);
        dataset.calculateEnergy((*it));
        if (useForces) dataset.calculateForces((*it));
        // Loop over atoms, collect sensitivity data and clear memory.
        for (vector<Atom>::iterator it2 = it->atoms.begin();
             it2 != it->atoms.end(); ++it2)
        {
            // Collect sensitivity data.
            size_t const& e = it2->element;
            count.at(e)++;
            for (size_t i = 0; i < numSymmetryFunctions.at(e); ++i)
            {
                double const& s = it2->dEdG.at(i);
                sensMean.at(e).at(i) += s * s;
                sensMax.at(e).at(i) = max(sensMax.at(e).at(i), abs(s));
            }
            // Clear unneccessary memory (neighbor list and others), energies
            // and forces are still stored. Don't use these structures after
            // these operations unless you know what you do!
            it2->numNeighborsUnique = 0;
            it2->neighborsUnique.clear();
            vector<size_t>(it2->neighborsUnique).swap(it2->neighborsUnique);

            it2->numNeighborsPerElement.clear();
            vector<size_t>(it2->numNeighborsPerElement).swap(
                                                  it2->numNeighborsPerElement);

            it2->G.clear();
            vector<double>(it2->G).swap(it2->G);

            it2->dEdG.clear();
            vector<double>(it2->dEdG).swap(it2->dEdG);

#ifdef NNP_FULL_SFD_MEMORY
            it2->dGdxia.clear();
            vector<double>(it2->dGdxia).swap(it2->dGdxia);
#endif

            it2->dGdr.clear();
            vector<Vec3D>(it2->dGdr).swap(it2->dGdr);

            it2->numNeighbors = 0;
            it2->neighbors.clear();
            vector<Atom::Neighbor>(it2->neighbors).swap(it2->neighbors);

            it2->hasNeighborList = false;
            it2->hasSymmetryFunctions = false;
            it2->hasSymmetryFunctionDerivatives = false;
        }
        it->hasNeighborList = false;
        it->hasSymmetryFunctions = false;
        it->hasSymmetryFunctionDerivatives = false;
        it->updateError("energy", errorEnergy, countEnergy);
        fileEnergy << strpr("%10zu %10zu", it->index, it->numAtoms);
        if (normalize)
        {
            fileEnergy << strpr(" %16.8E %16.8E %16.8E %16.8E\n",
                                dataset.physicalEnergy(*it, true)
                                + dataset.getEnergyOffset(*it),
                                dataset.physicalEnergy(*it, false)
                                + dataset.getEnergyOffset(*it),
                                it->energyRef,
                                it->energy);
        }
        else
        {
            fileEnergy << strpr(" %16.8E %16.8E\n",
                                dataset.getEnergyWithOffset(*it, true),
                                dataset.getEnergyWithOffset(*it, false));
        }
        if (useForces)
        {
            it->updateError("force", errorForces, countForces);
            for (vector<Atom>::const_iterator it2 = it->atoms.begin();
                 it2 != it->atoms.end(); ++it2)
            {
                for (size_t i = 0; i < 3; ++i)
                {
                    fileForces << strpr("%10zu %10zu",
                                        it2->indexStructure,
                                        it2->index);
                    if (normalize)
                    {
                        fileForces << strpr(
                                       " %16.8E %16.8E",
                                       dataset.physical("force", it2->fRef[i]),
                                       dataset.physical("force", it2->f[i]));
                    }
                    fileForces << strpr(" %16.8E %16.8E\n",
                                        it2->fRef[i],
                                        it2->f[i]);
                }
            }
        }
        if (normalize)
        {
            it->toPhysicalUnits(dataset.getMeanEnergy(),
                                dataset.getConvEnergy(),
                                dataset.getConvLength());
        }
        dataset.addEnergyOffset(*it, false);
        it->writeToFile(&fileOutputData, false);
    }

    fileEnergy.close();
    if (useForces) fileForces.close();
    fileOutputData.close();
    MPI_Barrier(MPI_COMM_WORLD);

    if (myRank == 0)
    {
        fileName = "energy.comp";
        dataset.combineFiles(fileName);
        if (useForces)
        {
            fileName = "forces.comp";
            dataset.combineFiles(fileName);
        }
        fileName = "output.data";
        dataset.combineFiles(fileName);
    }

    dataset.collectError("energy", errorEnergy, countEnergy);
    if (useForces) dataset.collectError("force", errorForces, countForces);

    if (myRank == 0)
    {
        if (useForces)
        {
            dataset.log << "Energy and force comparison in files:\n";
            dataset.log << " - energy.comp\n";
            dataset.log << " - forces.comp\n";
        }
        else
        {
            dataset.log << "Energy comparison in file:\n";
            dataset.log << " - energy.comp\n";
        }
        dataset.log << "Predicted data set in \"output.data\"\n";
    }
    dataset.log << "Error metrics for energies and forces:\n";
    dataset.log << "-----------------------------------------"
                   "-----------------------------------------"
                   "--------------------------------------\n";
    dataset.log << "                      physical units                          ";
    if (normalize)
    {
        dataset.log << " |                    internal units                            ";
    }
    dataset.log << "\n";
    dataset.log << strpr("       %13s %13s %13s %13s",
                         "RMSEpa", "RMSE", "MAEpa", "MAE");
    if (normalize)
    {
        dataset.log << strpr(" | %13s %13s %13s %13s",
                             "RMSEpa", "RMSE", "MAEpa", "MAE");
    }
    dataset.log << "\n";
    dataset.log << "ENERGY";
    if (normalize)
    {
        dataset.log << strpr(
                          " %13.5E %13.5E %13.5E %13.5E |",
                          dataset.physical("energy", errorEnergy.at("RMSEpa")),
                          dataset.physical("energy", errorEnergy.at("RMSE")),
                          dataset.physical("energy", errorEnergy.at("MAEpa")),
                          dataset.physical("energy", errorEnergy.at("MAE")));
    }
    dataset.log << strpr(" %13.5E %13.5E %13.5E %13.5E\n",
                         errorEnergy.at("RMSEpa"),
                         errorEnergy.at("RMSE"),
                         errorEnergy.at("MAEpa"),
                         errorEnergy.at("MAE"));
    if (useForces)
    {
        dataset.log << "FORCES";
        if (normalize)
        {
            dataset.log << strpr(
                         " %13s %13.5E %13s %13.5E |", "",
                         dataset.physical("force", errorForces.at("RMSE")), "",
                         dataset.physical("force", errorForces.at("MAE")));
        }
        dataset.log << strpr(" %13s %13.5E %13s %13.5E\n", "",
                             errorForces.at("RMSE"), "",
                             errorForces.at("MAE"));
    }
    dataset.log << "-----------------------------------------"
                   "-----------------------------------------"
                   "--------------------------------------\n";
    dataset.log << "*****************************************"
                   "**************************************\n";

    dataset.log << "\n";
    dataset.log << "*** SENSITIVITY ANALYSIS ****************"
                   "**************************************\n";
    dataset.log << "\n";

    // Combine sensititvity data from all procs.
    if (myRank == 0)
    {
        dataset.log << "Writing sensitivity analysis data to files:\n";
        MPI_Reduce(MPI_IN_PLACE, &(count.front()), numElements, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
        for (size_t i = 0; i < numElements; ++i)
        {
            size_t const& n = numSymmetryFunctions.at(i);
            double sensMeanSum = 0.0;
            double sensMaxSum = 0.0;
            MPI_Reduce(MPI_IN_PLACE, &(sensMean.at(i).front()), n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE, &(sensMax.at(i).front() ), n, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            for (size_t j = 0; j < numSymmetryFunctions.at(i); ++j)
            {
                sensMean.at(i).at(j) = sqrt(sensMean.at(i).at(j)
                                            / count.at(i));
                sensMeanSum += sensMean.at(i).at(j);
                sensMaxSum += sensMax.at(i).at(j);
            }
            ofstream sensFile;
            string sensFileName = strpr("sensitivity.%03d.out",
                                        dataset.elementMap.atomicNumber(i));
            dataset.log << strpr(" - %s\n", sensFileName.c_str());
            sensFile.open(sensFileName.c_str());

            // File header.
            vector<string> title;
            vector<string> colName;
            vector<string> colInfo;
            vector<size_t> colSize;
            title.push_back(strpr("Sensitivity analysis for element %2s.",
                                   dataset.elementMap[i].c_str()));
            colSize.push_back(10);
            colName.push_back("index");
            colInfo.push_back("Symmetry function index.");
            colSize.push_back(16);
            colName.push_back("sens_msa_norm");
            colInfo.push_back("Mean square average sensitivity (normalized, "
                              "sum = 100%).");
            colSize.push_back(16);
            colName.push_back("sens_max_norm");
            colInfo.push_back("Maximum sensitivity (normalized, sum = 100%).");
            colSize.push_back(16);
            colName.push_back("sens_msa_phys");
            colInfo.push_back("Mean square average sensitivity (physical "
                              "energy units).");
            colSize.push_back(16);
            colName.push_back("sens_max_phys");
            colInfo.push_back("Maximum sensitivity (physical energy units).");
            if (normalize)
            {
                colSize.push_back(16);
                colName.push_back("sens_msa_int");
                colInfo.push_back("Mean square average sensitivity (internal "
                                  "units).");
                colSize.push_back(16);
                colName.push_back("sens_max_int");
                colInfo.push_back("Maximum sensitivity (internal units).");
            }
            appendLinesToFile(sensFile,
                              createFileHeader(title,
                                               colSize,
                                               colName,
                                               colInfo));

            for (size_t j = 0; j < numSymmetryFunctions.at(i); ++j)
            {
                sensFile << strpr("%10d", j + 1);
                sensFile << strpr(" %16.8E %16.8E",
                                  sensMean.at(i).at(j) / sensMeanSum * 100.0,
                                  sensMax.at(i).at(j) / sensMaxSum * 100.0);
                if (normalize)
                {
                    sensFile << strpr(" %16.8E %16.8E",
                                      dataset.physical("energy",
                                                       sensMean.at(i).at(j)),
                                      dataset.physical("energy",
                                                       sensMax.at(i).at(j)));
                }
                sensFile << strpr(" %16.8E %16.8E\n",
                                  sensMean.at(i).at(j),
                                  sensMax.at(i).at(j));
            }
            sensFile.close();
        }
    }
    else
    {
        MPI_Reduce(&(count.front()), &(count.front()), numElements, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
        for (size_t i = 0; i < numElements; ++i)
        {
            size_t const& n = numSymmetryFunctions.at(i);
            MPI_Reduce(&(sensMean.at(i).front()), &(sensMean.at(i).front()), n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&(sensMax.at(i).front() ), &(sensMax.at(i).front() ), n, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        }
    }

    dataset.log << "*****************************************"
                   "**************************************\n";

    myLog.close();

    MPI_Finalize();

    return 0;
}
