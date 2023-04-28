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
    int                    warning = 0;
    int                    numProcs = 0;
    int                    myRank = 0;
    size_t                 stage = 0;
    double                 delta = 0.0;
    ofstream               myLog;
    ofstream               outFileSummary;
    map<string, double>    meanAbsError;
    map<string, double>    maxAbsError;
    map<string, string>    outFilesNames;
    map<string, ofstream*> outFiles;

    if (argc < 2 || argc > 3)
    {
        cout << "USAGE: " << argv[0] << " <stage> <<delta>>\n"
             << "       <stage> ..... Training stage (irrelevant for NNPs "
                "with only one stage).\n"
             << "       <<delta>> ... (optional) Displacement for central "
                "difference (default: 1.0e-4).\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - input.nn (NNP settings)\n"
             << "       - scaling.data (symmetry function scaling data)\n"
             << "       - \"weights.%%03d.data\" (weights files)\n";
        return 1;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    stage = (size_t)atoi(argv[1]);
    if (argc == 3) delta = atof(argv[2]);
    else delta = 1.0E-4;

    Training training;
    if (myRank != 0) training.log.writeToStdout = false;
    myLog.open(strpr("nnp-checkdw.log.%04d", myRank).c_str());
    training.log.registerStreamPointer(&myLog);
    training.setupMPI();
    training.initialize();
    training.loadSettingsFile();
    training.setStage(stage);
    training.setupGeneric();
    bool normalize = training.useNormalization();
    training.setupSymmetryFunctionScaling();
    training.setupSymmetryFunctionStatistics(false, false, true, false);
    training.setupNeuralNetworkWeights();
    training.distributeStructures(false);
    if (normalize) training.toNormalizedUnits();
    auto pk = training.setupNumericDerivCheck();

    training.log << "\n";
    training.log << "*** ANALYTIC/NUMERIC WEIGHT DERIVATIVES C"
                    "HECK *********************************\n";
    training.log << "\n";
    training.log << strpr("Delta for symmetric difference quotient: %11.3E\n",
                         delta);

    string fileName = "checkdw-summary.out";
    training.log << strpr("Per-structure summary of analytic/numeric "
                         "weight derivative comparison\n"
                         "will be written to \"%s\"\n",
                         fileName.c_str());
    fileName += strpr(".%04d", myRank);
    outFileSummary.open(fileName.c_str());
    if (myRank == 0)
    {
        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back(strpr("Per-structure summary of analytic vs. numeric "
                              "weight derivative comparison (delta = %11.3E).",
                              delta));
        colSize.push_back(10);
        colName.push_back("struct");
        colInfo.push_back("Structure index.");
        colSize.push_back(10);
        colName.push_back("numAtoms");
        colInfo.push_back("Number of atoms in structure.");
        for (auto k : pk)
        {
            colSize.push_back(16);
            colName.push_back("MAE_" + k);
            colInfo.push_back("Mean over all absolute differences between "
                              "analytic and numeric weight derivatives for "
                              "property \"" + k + "\".");
            colSize.push_back(16);
            colName.push_back("maxAE_" + k);
            colInfo.push_back("Maximum over all absolute differences between "
                              "analytic and numeric weight derivatives for "
                              "property \"" + k + "\".");
        }
        appendLinesToFile(outFileSummary,
                          createFileHeader(title, colSize, colName, colInfo));
    }

    for (auto k : pk)
    {
        string fileName = "checkdw-weights." + k + ".out";
        training.log << "Individual analytic/numeric weight derivatives for "
                        "property \"" + k + "\"\nwill be written to \""
                        + fileName + "\"\n";
        outFilesNames[k] = fileName;
        fileName += strpr(".%04d", myRank);
        outFiles[k] = new ofstream();
        outFiles.at(k)->open(fileName.c_str());
        if (myRank == 0)
        {
            // File header.
            vector<string> title;
            vector<string> colName;
            vector<string> colInfo;
            vector<size_t> colSize;
            title.push_back(strpr("Comparison of analytic and numeric weight "
                                  "derivatives for property \"%s\" (delta = "
                                  "%11.3E).", k.c_str(), delta));
            colSize.push_back(10);
            colName.push_back("struct");
            colInfo.push_back("Structure index.");
            colSize.push_back(10);
            colName.push_back("index");
            colInfo.push_back("Property index.");
            colSize.push_back(10);
            colName.push_back("weight");
            colInfo.push_back("Weight index.");
            //colSize.push_back(10);
            //colName.push_back("atom");
            //colInfo.push_back("Atom index (starting with 1).");
            //colSize.push_back(3);
            //colName.push_back("xyz");
            //colInfo.push_back("Force component (0 = x, 1 = y, 2 = z).");
            colSize.push_back(24);
            colName.push_back("analytic");
            colInfo.push_back("Analytic weight derivatives.");
            colSize.push_back(24);
            colName.push_back("numeric");
            colInfo.push_back("Numeric weight derivatives.");
            appendLinesToFile(*(outFiles.at(k)),
                              createFileHeader(title,
                                               colSize,
                                               colName,
                                               colInfo));
        }
    }

    training.log << "\n";
    training.log << "                      |";
    for (auto k : pk) training.log << strpr(" %33s |", k.c_str());
    training.log << "\n";
    training.log << "                      |";
    for (auto k : pk) training.log << " meanAbsError  maxAbsError verdict |";
    training.log << "\n";
    training.log << "-----------------------";
    for (auto k : pk) training.log << "------------------------------------";
    training.log << "\n";


    // Loop over all structures and compute symmetry functions and energies.
    bool useForces = training.settingsKeywordExists("use_short_forces");
    for (auto& s : training.structures)
    {
        s.calculateNeighborList(training.getMaxCutoffRadius());
#ifdef N2P2_NO_SF_GROUPS
        training.calculateSymmetryFunctions(s, useForces);
#else
        training.calculateSymmetryFunctionGroups(s, useForces);
#endif
        training.log << strpr("Configuration %6zu: ", s.index + 1);
        outFileSummary << strpr("%10zu %10zu", s.index + 1, s.numAtoms);
        for (auto k : pk)
        {
            vector<vector<double>> dPdc;
            training.dPdc(k, s, dPdc);
            vector<vector<double>> dPdcN;
            training.dPdcN(k, s, dPdcN, delta);
            meanAbsError[k] = 0.0;
            maxAbsError[k] = 0.0;
            size_t count = 0;
            for (size_t i = 0; i < dPdc.size(); ++i)
            {
                for (size_t j = 0; j < dPdc.at(i).size(); ++j)
                {
                    *(outFiles.at(k)) << strpr("%10zu %10zu %10zu "
                                               "%24.16E %24.16E\n",
                                               s.index + 1,
                                               i + 1,
                                               j + 1,
                                               dPdc.at(i).at(j),
                                               dPdcN.at(i).at(j));
                    outFiles.at(k)->flush();
                    double const error = fabs(dPdc.at(i).at(j) -
                                              dPdcN.at(i).at(j));
                    meanAbsError.at(k) += error;
                    maxAbsError.at(k) = max(error, maxAbsError.at(k));
                    count++;
                }
            }
            meanAbsError.at(k) /= count;
            string swarn = "OK";
            if (maxAbsError.at(k) > 100 * delta * delta)
            {
                swarn = "WARN";
                warning++;
            }
            training.log << strpr("  %12.3E %12.3E %7s ",
                                  meanAbsError.at(k),
                                  maxAbsError.at(k),
                                  swarn.c_str());
            outFileSummary << strpr(" %16.8E %16.8E",
                                    meanAbsError.at(k),
                                    maxAbsError.at(k));
        }
        training.log << "\n";
        outFileSummary << "\n";
        s.reset();
    }
    if (normalize) training.toPhysicalUnits();

    outFileSummary.close();
    MPI_Barrier(MPI_COMM_WORLD);
    if (myRank == 0) training.combineFiles("checkdw-summary.out");

    for (auto k : pk)
    {
        outFiles.at(k)->close();
        delete outFiles.at(k);
        MPI_Barrier(MPI_COMM_WORLD);
        if (myRank == 0) training.combineFiles(outFilesNames.at(k));
    }

    MPI_Allreduce(MPI_IN_PLACE, &warning, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (warning > 0)
    {
        training.log << "\n";
        training.log << "IMPORTANT: Some warnings were issued. By default, this happens if the maximum\n"
                        "           absolute error (\"maxAbsError\") is higher than 100 * delta².\n"
                        "           However, this does NOT mean that analytic derivatives are incorrect!\n"
                        "           Repeat this analysis with a different delta and check whether\n"
                        "           the error scales with O(delta²). The prefactor for your system\n"
                        "           could be higher than 10, hence, as long as there is a O(delta²)\n"
                        "           scaling the analytic derivatives are probably correct.\n";
    }

    training.log << "*****************************************"
                   "**************************************\n";

    myLog.close();

    MPI_Finalize();

    return 0;
}
