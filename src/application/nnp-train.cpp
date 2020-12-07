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
#include "utility.h"
#include <mpi.h>
#include <cstddef> // std::size_t
#include <cstdlib> // atoi
#include <fstream>
#include <string>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    int numProcs = 0;
    int myRank   = 0;
    size_t stage = 0;
    ofstream myLog;

    string suffix = "";
    if (argc > 1)
    {
        stage = (size_t)atoi(argv[1]);
        suffix = strpr(".stage-%zu", stage);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    // Basic setup.
    Training training;
    if (myRank != 0) training.log.writeToStdout = false;

    myLog.open((strpr("nnp-train.log.%04d", myRank) + suffix).c_str());
    training.log.registerStreamPointer(&myLog);
    training.setupMPI();
    training.initialize();
    training.loadSettingsFile();
    training.setStage(stage);
    training.setupGeneric();
    training.setupSymmetryFunctionScaling();
    training.setupSymmetryFunctionStatistics(false, false, false, false);
    training.setupRandomNumberGenerator();

    // Distribute structures to MPI processes.
    training.distributeStructures(true);

    // Randomly select training/test set and write them to separate files
    // (train.data/test.data, same format as input.data).
    training.selectSets();
    training.writeSetsToFiles();

    // Switch to normalized units, convert all structures.
    if (training.useNormalization()) training.toNormalizedUnits();

    // Initialize weights and biases for neural networks.
    training.initializeWeights();

    // General training settings and weight update routine.
    training.setupTraining();

    // Calculate neighbor lists for all structures.
    training.calculateNeighborLists();

    // The main training loop.
    training.loop();

    myLog.close();

    MPI_Finalize();

    return 0;
}
