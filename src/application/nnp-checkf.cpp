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
    int      numProcs = 0;
    int      myRank = 0;
    size_t   numOriginals = 0;
    double   delta = 0.0;
    ifstream dataFile;
    ofstream myLog;
    ofstream outFile;

    if (argc > 2)
    {
        cout << "USAGE: " << argv[0] << " <<delta>>\n"
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

    if (argc == 2) delta = atof(argv[1]);
    else delta = 1.0E-4;

    Dataset dataset;
    if (myRank != 0) dataset.log.writeToStdout = false;
    myLog.open(strpr("nnp-checkf.log.%04d", myRank).c_str());
    dataset.log.registerStreamPointer(&myLog);
    dataset.setupMPI();
    dataset.initialize();
    dataset.loadSettingsFile();
    dataset.setupGeneric();
    bool normalize = dataset.useNormalization();
    dataset.setupSymmetryFunctionScaling();
    dataset.setupSymmetryFunctionStatistics(false, false, true, false);
    dataset.setupNeuralNetworkWeights();

    dataset.log << "\n";
    dataset.log << "*** ANALYTIC/NUMERIC FORCES CHECK *******"
                   "**************************************\n";
    dataset.log << "\n";
    dataset.log << strpr("Delta for symmetric difference quotient: %11.3E\n",
                         delta);
    if (myRank == 0)
    {
        string fileName = "forces.out";
        dataset.log << strpr("Individual analytic/numeric forces will be "
                             "written to \"%s\"\n", fileName.c_str());
        outFile.open(fileName.c_str());
        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back(strpr("Comparison of analytic and numeric forces "
                              "(delta = %11.3E).", delta));
        colSize.push_back(10);
        colName.push_back("struct");
        colInfo.push_back("Structure index (starting with 1).");
        colSize.push_back(10);
        colName.push_back("atom");
        colInfo.push_back("Atom index (starting with 1).");
        colSize.push_back(3);
        colName.push_back("xyz");
        colInfo.push_back("Force component (0 = x, 1 = y, 2 = z).");
        colSize.push_back(24);
        colName.push_back("F_analytic");
        colInfo.push_back("Force computed from analytic derivative of NNP "
                          "energy.");
        colSize.push_back(24);
        colName.push_back("F_numeric");
        colInfo.push_back("Force computed numerically from symmetric "
                          "difference quotient.");
        appendLinesToFile(outFile,
                          createFileHeader(title, colSize, colName, colInfo));
    }

    // First, check how many originals are in "input.data".
    if (myRank == 0)
    {
        string fileName = "input.data";
        dataFile.open(fileName.c_str());
        numOriginals = dataset.getNumStructures(dataFile);
        dataset.log << strpr("Found %zu configurations in data file: %s.\n",
                             numOriginals,
                             fileName.c_str());
        dataFile.clear();
        dataFile.seekg(0);
    }
    MPI_Bcast(&numOriginals, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
    dataset.log << strpr("Starting loop over %zu configurations...\n",
                          numOriginals);

    if (myRank == 0)
    {
        dataset.log << "\n";
        dataset.log << strpr("                      %10s %12s %12s  verdict\n",
                             "numForces", "meanAbsError", "maxAbsError");
        dataset.log << "-----------------------------------"
                       "--------------------------------\n";
    }

    bool warning = false;
    for (size_t is = 0; is < numOriginals; ++is)
    {
        // Read original, prepare and distribute modified copies.
        Structure original;
        original.setElementMap(dataset.elementMap);
        if (myRank == 0) original.readFromFile(dataFile);
        size_t numStructures = dataset.prepareNumericForces(original, delta);
        if (normalize) dataset.toNormalizedUnits();

        // Prepare arrays to collect central difference values and identifiers.
        vector<double> allEnergies(numStructures - 1, 0.0);
        vector<size_t> allAtoms(numStructures - 1, 0);
        vector<size_t> allXYZ(numStructures - 1, 0);
        vector<int> allSigns(numStructures - 1, 0);

        // Loop over all structures and compute symmetry functions and energies.
        for (auto& s : dataset.structures)
        {
            bool needForces = false;
            if (s.comment == "original") needForces = true;
            s.calculateNeighborList(dataset.getMaxCutoffRadius());
#ifdef N2P2_NO_SF_GROUPS
            dataset.calculateSymmetryFunctions(s, needForces);
#else
            dataset.calculateSymmetryFunctionGroups(s, needForces);
#endif
            dataset.calculateAtomicNeuralNetworks(s, needForces);
            dataset.calculateEnergy(s);
            if (needForces) dataset.calculateForces(s);
            s.freeAtoms(true);
        }
        if (normalize) dataset.toPhysicalUnits();

        // Loop over all structures collect energies and identifiers.
        for (auto& s : dataset.structures)
        {
            // Skip for original structure.
            if (s.comment == "original") continue;
            // Store central difference values in arrays (locally on each
            // processor, communicate later).
            vector<string> lsplit = split(s.comment);
            size_t count = atoi(lsplit.at(0).c_str());
            size_t iAtom = atoi(lsplit.at(1).c_str());
            size_t ixyz = atoi(lsplit.at(2).c_str());
            int sign = atoi(lsplit.at(3).c_str());
            allEnergies.at(count) = s.energy;
            allAtoms.at(count) = iAtom;
            allXYZ.at(count) = ixyz;
            allSigns.at(count) = sign;
        }

        // Collect data on rank 0.
        if (myRank == 0)
        {
            MPI_Reduce(MPI_IN_PLACE      , allEnergies.data(), numStructures - 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE      , allAtoms.data()   , numStructures - 1, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE      , allXYZ.data()     , numStructures - 1, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(MPI_IN_PLACE      , allSigns.data()   , numStructures - 1, MPI_INT   , MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Reduce(allEnergies.data(), allEnergies.data(), numStructures - 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(allAtoms.data()   , allAtoms.data()   , numStructures - 1, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(allXYZ.data()     , allXYZ.data()     , numStructures - 1, MPI_SIZE_T, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(allSigns.data()   , allSigns.data()   , numStructures - 1, MPI_INT   , MPI_SUM, 0, MPI_COMM_WORLD);
        }

        if (myRank == 0)
        {
            // Now temporarily store central difference energies in
            // fRef(+delta) and f(-delta) of original structure.
            for (size_t i = 0; i < numStructures - 1; ++i)
            {
                Atom& a = original.atoms.at(allAtoms.at(i));
                if (allSigns.at(i) == 1)
                {
                    a.fRef[allXYZ.at(i)] = allEnergies.at(i);
                }
                else if (allSigns.at(i) == -1)
                {
                    a.f[allXYZ.at(i)] = allEnergies.at(i);
                }
            }

            double maxAbsError = 0.0;
            double meanAbsError = 0.0;
            for (size_t i = 0; i < original.atoms.size(); ++i)
            {
                Atom const& ref = dataset.structures.at(0).atoms.at(i);
                Atom& a = original.atoms.at(i);
                for (size_t j = 0; j < 3; ++j)
                {
                    a.f[j] = - (a.fRef[j] - a.f[j]) / (2.0 * delta);
                    a.fRef[j] = ref.f[j];
                    double const error = fabs(a.f[j] - a.fRef[j]);
                    meanAbsError += error;
                    maxAbsError = max(error, maxAbsError);
                    outFile << strpr("%10zu %10zu %3zu %24.16E %24.16E\n",
                                     is + 1, i + 1, j, a.fRef[j], a.f[j]);
                }
            }
            size_t const numForces = 3 * original.atoms.size();
            meanAbsError /= numForces;
            dataset.log << strpr("Configuration %6zu: %10zu %12.3E %12.3E",
                                 is + 1, numForces, meanAbsError, maxAbsError);
            if (maxAbsError > 10 * delta * delta)
            {
                dataset.log << "  WARNING!\n";
                warning = true;
            }
            else dataset.log << "  OK.\n";
        }
    }

    if (myRank == 0)
    {
        if (warning)
        {
            dataset.log << "\n"; 
            dataset.log << "IMPORTANT: Some warnings were issued. By default, this happens if the maximum\n"
                           "           absolute error (\"maxAbsError\") is higher than 10 * delta².\n"
                           "           However, this does NOT mean that analytic forces are incorrect!\n"
                           "           Repeat this analysis with a different delta and check whether\n"
                           "           the error scales with O(delta²). The prefactor for your system\n"
                           "           could be higher than 10, hence, as long as there is a O(delta²)\n"
                           "           scaling the analytic forces are probably correct.\n";
        }
        outFile.close();
        dataFile.close();
    }

    dataset.log << "*****************************************"
                   "**************************************\n";

    myLog.close();

    MPI_Finalize();

    return 0;
}
