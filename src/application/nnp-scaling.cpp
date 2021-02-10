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
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    bool   useForces = false;
    int    numProcs  = 0;
    int    myRank    = 0;
    long   memory    = 0;
    size_t count     = 0;
    ofstream myLog;

    if (argc != 2)
    {
        cout << "USAGE: " << argv[0] << " <nbins>\n"
             << "       <nbins> ... Number of symmetry function"
                " histogram bins.\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - input.nn (NNP settings)\n";
        return 1;
    }

    size_t numBins = (size_t)atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    Dataset dataset;
    myLog.open(strpr("nnp-scaling.log.%04d", myRank).c_str());
    if (myRank != 0) dataset.log.writeToStdout = false;
    dataset.log.registerStreamPointer(&myLog);
    dataset.setupMPI();
    dataset.initialize();
    dataset.loadSettingsFile();
    dataset.setupGeneric();
    dataset.setupSymmetryFunctionScalingNone();
    dataset.setupSymmetryFunctionStatistics(true, false, false, false);
    dataset.setupRandomNumberGenerator();
    dataset.distributeStructures(true);
    if (dataset.useNormalization()) dataset.toNormalizedUnits();
    useForces = dataset.settingsKeywordExists("use_short_forces");

    dataset.log << "\n";
    dataset.log << "*** CALCULATING SYMMETRY FUNCTIONS ******"
                   "**************************************\n";
    dataset.log << "\n";

    dataset.log << "Check the log files of all (!) MPI processes"
                   " for warnings in this section!\n";

    for (vector<Structure>::iterator it = dataset.structures.begin();
         it != dataset.structures.end(); ++it)
    {
        it->calculateNeighborList(dataset.getMaxCutoffRadius());
#ifdef NNP_NO_SF_GROUPS
        dataset.calculateSymmetryFunctions((*it), useForces);
#else
        dataset.calculateSymmetryFunctionGroups((*it), useForces);
#endif
        memory += dataset.calculateBufferSize((*it));
        count++;
        // Clear unneccessary memory (neighbor list and others), leave only
        // numNeighbors and symmetry functions (G) for histogram calculation.
        // Don't use these structures after these operations unless you know
        // what you do!
        for (vector<Atom>::iterator it2 = it->atoms.begin();
             it2 != it->atoms.end(); ++it2)
        {
            it2->numNeighborsUnique = 0;
            it2->neighborsUnique.clear();
            vector<size_t>(it2->neighborsUnique).swap(it2->neighborsUnique);

            it2->numNeighborsPerElement.clear();
            vector<size_t>(it2->numNeighborsPerElement).swap(
                                                  it2->numNeighborsPerElement);

            it2->dEdG.clear();
            vector<double>(it2->dEdG).swap(it2->dEdG);

#ifdef NNP_FULL_SFD_MEMORY
            it2->dGdxia.clear();
            vector<double>(it2->dGdxia).swap(it2->dGdxia);
#endif

            it2->dGdr.clear();
            vector<Vec3D>(it2->dGdr).swap(it2->dGdr);

            // Leave this for the histogram.
            //it2->numNeighbors = 0;
            it2->neighbors.clear();
            vector<Atom::Neighbor>(it2->neighbors).swap(it2->neighbors);
        }
    }
    dataset.log << "*****************************************"
                   "**************************************\n";

    dataset.collectSymmetryFunctionStatistics();
    dataset.writeSymmetryFunctionScaling();
    dataset.writeNeighborHistogram();
    dataset.writeSymmetryFunctionHistograms(numBins);
    dataset.writeSymmetryFunctionFile();

    dataset.log << "\n";
    dataset.log << "*** MEMORY USAGE ESTIMATION *************"
                   "**************************************\n";
    dataset.log << "\n";
    dataset.log << "Estimated memory usage for training (keyword"
                   " \"memorize_symfunc_results\":\n";
    if (useForces)
    {
        dataset.log << "Valid for training of energies and forces.\n";
    }
    else
    {
        dataset.log << "Valid for training of energies only.\n";
    }
    dataset.log << strpr("Memory for local structures  : "
                         "%15ld bytes (%.2f MiB = %.2f GiB).\n",
                         memory,
                         memory / 1024. / 1024.,
                         memory / 1024. / 1024. / 1024.);
    MPI_Allreduce(MPI_IN_PLACE, &memory, 1, MPI_LONG  , MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &count , 1, MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
    dataset.log << strpr("Memory for all structures    : "
                         "%15ld bytes (%.2f MiB = %.2f GiB).\n",
                         memory,
                         memory / 1024. / 1024.,
                         memory / 1024. / 1024. / 1024.);
    dataset.log << strpr("Average memory per structure : "
                         "%15.0f bytes (%.2f MiB).\n",
                         memory / (double)count,
                         memory / (double)count / 1024. / 1024.);
    dataset.log << "*****************************************"
                   "**************************************\n";

    myLog.close();

    MPI_Finalize();

    return 0;
}
