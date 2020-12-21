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
#include "Log.h"
#include "utility.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    string   mode;
    int      numProcs        = 0;
    int      numProcsNNP     = 0;
    int      numProcsData    = 0;
    int      myRank          = 0;
    int      myRankNNP       = 0;
    int      myRankData      = 0;
    int      myNNP           = 0;
    int      myData          = 0;
    int      numWorkers      = 0;
    size_t   thresholdEW     = 0;
    double   thresholdEnergy = 0.0;
    double   thresholdForce  = 0.0;
    string   elements;
    ofstream myLog;

    if (argc < 2)
    {
        cout << "USAGE: " << argv[0] << " <mode> <t_ew> <t_en <t_f> "
                "<elem1 <elem2 ...>>\n"
             << "       <mode> ... Either compare 2 NNPs (compare) or apply"
                " threshold to existing comparison data set (select).\n"
             << "       If <mode> is \"select\":\n"
             << "       <t_ew> ... Extrapolation warning threshold.\n"
             << "       <t_en> ... Energy per atom threshold.\n"
             << "       <t_f> .... Force threshold.\n"
             << "       <elem1 <elem2 ...>> ... Element strings in data set "
                "(e.g. H O).\n"
             << "       Execute in directory with the data set file\n"
             << "       - input.data\n"
             << "       and 2 subdirectories\n"
             << "       - nnp-data-1\n"
             << "       - nnp-data-2\n"
             << "       each containing these NNP files:\n"
             << "       - input.nn (NNP settings)\n"
             << "       - scaling.data (symmetry function scaling data)\n"
             << "       - \"weights.%%03d.data\" (weights files)\n";
        return 1;
    }

    mode = argv[1];
    if (mode == "compare")
    {
    }
    else if (mode == "select")
    {
        if (argc < 6)
        {
            throw runtime_error("ERROR: Wrong number of arguments.\n");
        }
        thresholdEW        = (size_t)atoi(argv[2]);
        thresholdEnergy    = atof(argv[3]);
        thresholdForce     = atof(argv[4]);
        size_t numElements = argc - 5;
        elements += argv[5];
        for (size_t i = 6; i < numElements + 5; ++i)
        {
            elements += " ";
            elements += argv[i];
        }
    }
    else
    {
        throw runtime_error("ERROR: Unknown mode selected.\n");
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    if ((mode == "compare") && (numProcs % 2 != 0))
    {
        throw runtime_error("ERROR: Please start with an even number of MPI"
                            " processes.\n");
    }
    else if ((mode == "select") && (numProcs != 1))
    {
        throw runtime_error("ERROR: Please start with a single MPI "
                            "processes.\n");
    }

    if (mode == "compare")
    {
        numWorkers = numProcs / 2;
        MPI_Comm commNNP;
        myNNP = myRank / numWorkers;
        MPI_Comm_split(MPI_COMM_WORLD, myNNP, myRank, &commNNP);
        MPI_Comm_size(commNNP, &numProcsNNP);
        MPI_Comm_rank(commNNP, &myRankNNP);

        MPI_Comm commData;
        myData = myRankNNP % numWorkers;
        MPI_Comm_split(MPI_COMM_WORLD, myData, myRank, &commData);
        MPI_Comm_size(commData, &numProcsData);
        MPI_Comm_rank(commData, &myRankData);

        Dataset dataset;
        if (myRank != 0) dataset.log.writeToStdout = false;
        myLog.open(strpr("nnp-comp2.log.%1d.%04d",
                         myNNP + 1,
                         myRankNNP).c_str());
        dataset.log.registerStreamPointer(&myLog);

        string dirNNP = strpr("nnp-data-%1d", myNNP + 1); 

        dataset.log << "\n";
        dataset.log << "*** 2-NNP COMPARISON INITIALIZATION *****"
                       "**************************************\n";
        dataset.log << "\n";
        dataset.log << strpr("Number of workers per NNP: %4d\n", numWorkers);
        dataset.log << strpr("NNP  Id                  : %4d\n", myNNP);
        dataset.log << strpr("Data Id                  : %4d\n", myData);
        dataset.log << "My NNP directory         : " + dirNNP + "\n";
        dataset.log << "\n";
        dataset.log << "Global Communicator:\n";
        dataset.log << strpr("- numProcs: %4d\n", numProcs);
        dataset.log << strpr("- myRank  : %4d\n", myRank);
        dataset.log << "\n";
        dataset.log << "NNP Communicator:\n";
        dataset.log << strpr("- numProcs: %4d\n", numProcsNNP);
        dataset.log << strpr("- myRank  : %4d\n", myRankNNP);
        dataset.log << "\n";
        dataset.log << "Data Communicator:\n";
        dataset.log << strpr("- numProcs: %4d\n", numProcsData);
        dataset.log << strpr("- myRank  : %4d\n", myRankData);
        dataset.log << "\n";
        dataset.log << "Starting NNP initialization...\n";
        dataset.log << "*****************************************"
                       "**************************************\n";

        dataset.setupMPI(&commNNP);
        dataset.initialize();
        dataset.loadSettingsFile(dirNNP + "/input.nn");
        dataset.setupGeneric();
        bool normalize = dataset.useNormalization();
        bool useForces = dataset.settingsKeywordExists("use_short_forces");
        dataset.setupSymmetryFunctionScaling(dirNNP + "/scaling.data");
        dataset.setupSymmetryFunctionStatistics(false, true, false, false);
        dataset.setupNeuralNetworkWeights(dirNNP + "/weights.%03d.data");
        dataset.distributeStructures(false);
        if (normalize) dataset.toNormalizedUnits();

        dataset.log << "\n";
        dataset.log << "*** 2-NNP COMPARISON ********************"
                       "**************************************\n";
        dataset.log << "\n";
        if (useForces)
        { 
            dataset.log << "Calculating energies and forces for dataset.\n";
        }
        else
        {
            dataset.log << "Calculating energies for dataset.\n";
        }

        for (vector<Structure>::iterator it = dataset.structures.begin();
             it != dataset.structures.end(); ++it)
        {
            it->calculateNeighborList(dataset.getMaxCutoffRadius());
#ifdef NNP_NO_SF_GROUPS
            dataset.calculateSymmetryFunctions((*it), useForces);
#else
            dataset.calculateSymmetryFunctionGroups((*it), useForces);
#endif
            dataset.calculateAtomicNeuralNetworks((*it), useForces);
            dataset.calculateEnergy((*it));
            if (useForces) dataset.calculateForces((*it));
            if (normalize) dataset.convertToPhysicalUnits((*it));
            dataset.addEnergyOffset((*it), true);
            dataset.addEnergyOffset((*it), false);
            // Store number of extrapolation warnings in unused
            // "numElementsPresent" field.
            it->clearNeighborList(); 
            it->numElementsPresent = dataset.getNumExtrapolationWarnings();
            dataset.resetExtrapolationWarnings();
        }

        // Receive data from other NNP worker (rank 1 in data comm) and
        // subtract results.
        if (myRankData == 0)
        {
            ofstream myComparisonData;
            myComparisonData.open(strpr("comp.data.%04d", myRankNNP).c_str());
            ofstream myDiff;
            myDiff.open(strpr("diff.out.%04d", myRankNNP).c_str());
            if (myRankNNP == 0)
            {
                // File header.
                vector<string> title;
                vector<string> colName;
                vector<string> colInfo;
                vector<size_t> colSize;
                title.push_back("2-NNP EW, energy and force comparison.");
                colSize.push_back(7);
                colName.push_back("index");
                colInfo.push_back("Structure index.");
                colSize.push_back(10);
                colName.push_back("nAtoms");
                colInfo.push_back("Number of atoms in structure.");
                colSize.push_back(10);
                colName.push_back("EW_NNP1");
                colInfo.push_back("Extrapolation warnings issued by NNP 1.");
                colSize.push_back(10);
                colName.push_back("EW_NNP2");
                colInfo.push_back("Extrapolation warnings issued by NNP 2.");
                colSize.push_back(16);
                colName.push_back("E_NNP1");
                colInfo.push_back("Energy prediction of NNP 1.");
                colSize.push_back(16);
                colName.push_back("E_NNP2");
                colInfo.push_back("Energy prediction of NNP 2.");
                colSize.push_back(16);
                colName.push_back("FDiffMean");
                colInfo.push_back("Mean absolute force difference.");
                colSize.push_back(16);
                colName.push_back("FDiffMax");
                colInfo.push_back("Maximum absolute force difference.");
                appendLinesToFile(myDiff,
                                  createFileHeader(title,
                                                   colSize,
                                                   colName,
                                                   colInfo));
            }

            for (vector<Structure>::iterator it = dataset.structures.begin();
                 it != dataset.structures.end(); ++it)
            {
                vector<double> buffer;
                size_t sizeBuffer = 2;
                if (useForces) sizeBuffer += 3 * it->numAtoms;
                buffer.resize(sizeBuffer, 0.0);
                MPI_Recv(&(buffer.front()), buffer.size(), MPI_DOUBLE, 1, 0, commData, MPI_STATUS_IGNORE);
                size_t count = 0;
                double eNNP2 = buffer.at(count++);
                size_t ewNNP2 = (size_t)buffer.at(count++);
                double meanForceDiff = 0.0;
                double maxForceDiff = 0.0;
                it->energyRef = it->energy - eNNP2;
                if (useForces)
                {
                    for (vector<Atom>::iterator it2 = it->atoms.begin();
                         it2 != it->atoms.end(); ++it2)
                    {
                        for (size_t i = 0; i < 3; ++i)
                        {
                            it2->fRef[i] = it2->f[i] - buffer.at(count++);
                            double const fAbsDiff = fabs(it2->fRef[i]);
                            meanForceDiff += fAbsDiff;
                            maxForceDiff = max(maxForceDiff, fAbsDiff);
                        }
                    }
                    meanForceDiff /= 3 * it->numAtoms;
                }
                // Add EW sum to comment line.
                myDiff << strpr("%7zu %10zu %10zu %10zu %16.8E %16.8E "
                                "%16.8E %16.8E\n",
                                it->index,
                                it->numAtoms,
                                it->numElementsPresent,
                                ewNNP2,
                                it->energy,
                                eNNP2,
                                meanForceDiff,
                                maxForceDiff);
                it->numElementsPresent += ewNNP2;
                it->comment = strpr("EWSUM %zu ", it->numElementsPresent)
                            + it->comment;
                it->writeToFile(&myComparisonData);
            }
            myComparisonData.close();
            myDiff.close();
        }
        // Send data to other NNP worker (rank 0 in data comm).
        else if (myRankData == 1)
        {
            for (vector<Structure>::const_iterator it = dataset.structures.begin();
                 it != dataset.structures.end(); ++it)
            {
                double ew = (double)it->numElementsPresent;
                vector<double> buffer;
                buffer.push_back(it->energy);
                buffer.push_back(ew);
                if (useForces)
                {
                    for (vector<Atom>::const_iterator it2 = it->atoms.begin();
                         it2 != it->atoms.end(); ++it2)
                    {
                        buffer.push_back(it2->f[0]);
                        buffer.push_back(it2->f[1]);
                        buffer.push_back(it2->f[2]);
                    }
                }
                MPI_Send(&(buffer.front()), buffer.size(), MPI_DOUBLE, 0, 0, commData);
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (myRank == 0)
        {
            dataset.log << "Difference structures written to \"comp.data\".\n";
            dataset.combineFiles("comp.data");
            dataset.log << "Difference data collected in \"diff.out\".\n";
            dataset.combineFiles("diff.out");
        }

        MPI_Comm_free(&commNNP);
        MPI_Comm_free(&commData);

        dataset.log << "*****************************************"
                       "**************************************\n";
    }
    else if (mode == "select")
    {
        ofstream logFile;
        logFile.open("nnp-comp2.log");
        Log log;
        log.registerStreamPointer(&logFile);
        log << "\n";
        log << "*** 2-NNP COMPARISON SELECTION **********"
               "**************************************\n";
        log << "\n";
        log << strpr("Element string: %s\n", elements.c_str());
        log << strpr("Threshold extrapolation warnings    : %d\n",
                     thresholdEW);
        log << strpr("Threshold energy difference per atom: %g\n",
                     thresholdEnergy);
        log << strpr("Threshold force difference          : %g\n",
                     thresholdForce);
        log << "*****************************************"
               "**************************************\n";

        ofstream selectFile;
        selectFile.open("select.out");
        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back("Selection results based on 2-NNP comparison.");
        colSize.push_back(7);
        colName.push_back("index");
        colInfo.push_back("Structure index.");
        colSize.push_back(5);
        colName.push_back("fAny");
        colInfo.push_back("Whether structure was selected.");
        colSize.push_back(5);
        colName.push_back("fEW");
        colInfo.push_back("Structure selected due to extrapolation warning.");
        colSize.push_back(5);
        colName.push_back("fE");
        colInfo.push_back("Structure selected due to energy difference.");
        colSize.push_back(5);
        colName.push_back("fF");
        colInfo.push_back("Structure selected due to force difference.");
        colSize.push_back(10);
        colName.push_back("nEW");
        colInfo.push_back("Number of extrapolation warnings (both NNPs "
                          "combined).");
        colSize.push_back(10);
        colName.push_back("nAtoms");
        colInfo.push_back("Number of atoms in structure.");
        colSize.push_back(16);
        colName.push_back("Ediff");
        colInfo.push_back("Energy difference per atom.");
        colSize.push_back(10);
        colName.push_back("nForce");
        colInfo.push_back("Number of forces exceeding threshold.");
        colSize.push_back(16);
        colName.push_back("FDiffMean");
        colInfo.push_back("Mean absolute force difference.");
        colSize.push_back(16);
        colName.push_back("FDiffMax");
        colInfo.push_back("Maximum absolute force difference.");
        appendLinesToFile(selectFile,
                          createFileHeader(title, colSize, colName, colInfo));

        ElementMap elementMap;
        elementMap.registerElements(elements);

        Structure structure;
        structure.setElementMap(elementMap);

        ifstream comparisonData;
        comparisonData.open("comp.data");

        ofstream selectionData;
        selectionData.open("comp-selection.data");

        size_t countEWHit      = 0;
        size_t countEnergyHit  = 0;
        size_t countForceHit   = 0;
        size_t countStructures = 0;
        size_t countSelected   = 0;
        while (comparisonData.peek() != EOF)
        {
            bool flagEW          = false;
            bool flagEnergy      = false;
            bool flagForce       = false;
            size_t countForce    = 0;
            double meanForceDiff = 0.0;
            double maxForceDiff  = 0.0;
            structure.readFromFile(comparisonData);
            structure.index = countStructures++;
            Structure const& s = structure;
            size_t countEW = (size_t)atoi(split(s.comment).at(1).c_str());
            if (thresholdEW > 0 && countEW >= thresholdEW)
            {
                flagEW = true;
                countEWHit++;
            }
            if (thresholdEnergy > 0 &&
                fabs(s.energyRef) / s.numAtoms > thresholdEnergy)
            {
                flagEnergy = true;
                countEnergyHit++;
            }
            if (thresholdForce > 0)
            {
                for (vector<Atom>::const_iterator it = s.atoms.begin();
                     it != s.atoms.end(); ++it)
                {
                    for (size_t i = 0; i < 3; ++i)
                    {
                        double const fAbsDiff = fabs(it->fRef[i]);
                        maxForceDiff = max(maxForceDiff, fAbsDiff);
                        meanForceDiff += fAbsDiff;
                        if (fAbsDiff > thresholdForce)
                        {
                            flagForce = true;
                            countForce++;
                        }
                    }
                } 
                meanForceDiff /= 3 * s.numAtoms;
                if (flagForce) countForceHit++;
            }
            if (flagEW || flagEnergy || flagForce)
            {
                s.writeToFile(&selectionData);
                countSelected++;
                log << strpr("Structure %7zu selected.\n", structure.index);
            }
            selectFile << strpr("%7zu %5d %5d %5d %5d %10zu %10zu %16.8E "
                                "%10zu %16.8E %16.8E\n",
                                structure.index,
                                (int)(flagEW || flagEnergy || flagForce),
                                (int)flagEW,
                                (int)flagEnergy,
                                (int)flagForce,
                                countEW,
                                structure.numAtoms,
                                structure.energyRef / structure.numAtoms,
                                countForce,
                                meanForceDiff,
                                maxForceDiff);
            structure.reset();
        }
        selectFile.close();

        log << "*****************************************"
               "**************************************\n";
        log << strpr("Total number of structures   : %7zu\n", countStructures);
        log << strpr("Number of selected structures: %7zu (%6.2f %)\n",
                     countSelected, countSelected * 100.0 / countStructures);
        log << strpr("EW     threshold exceeded    : %7zu (%6.2f %)\n",
                     countEWHit, countEWHit * 100.0 / countStructures);
        log << strpr("Energy threshold exceeded    : %7zu (%6.2f %)\n",
                     countEnergyHit, countEnergyHit * 100.0 / countStructures);
        log << strpr("Force  threshold exceeded    : %7zu (%6.2f %)\n",
                     countForceHit, countForceHit * 100.0 / countStructures);
        log << "Selected structures written to \"comp-selection.data\".\n";
        log << "*****************************************"
               "**************************************\n";
    }

    MPI_Finalize();

    return 0;
}
