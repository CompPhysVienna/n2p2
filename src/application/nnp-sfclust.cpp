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
    ofstream myLog;

    if (argc < 3)
    {
        cout << "USAGE: " << argv[0] << " <nbins> <ncutij> \n"
             << "       <nbins> ... Number of symmetry function"
                " histogram bins.\n"
             << "       <ncutij> ... Maximum number of neighbor symmetry "
                "functions written (for each element combination).\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - input.nn (NNP settings)\n"
             << "       - scaling.data (symmetry function scaling data)\n";
        return 1;
    }

    size_t numBins = (size_t)atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    Dataset dataset;
    myLog.open(strpr("nnp-sfclust.log.%04d", myRank).c_str());
    if (myRank != 0) dataset.log.writeToStdout = false;
    dataset.log.registerStreamPointer(&myLog);
    dataset.setupMPI();
    dataset.initialize();
    dataset.loadSettingsFile();
    dataset.setupGeneric();
    dataset.setupSymmetryFunctionScaling();
    dataset.setupSymmetryFunctionStatistics(true, false, true, false);
    dataset.setupRandomNumberGenerator();
    dataset.distributeStructures(false);
    if (dataset.useNormalization()) dataset.toNormalizedUnits();

    size_t n = dataset.getNumElements();
    if ((size_t)argc - 2 != n * n)
    {
        throw runtime_error("ERROR: Wrong number of neighbor cutoffs.\n");
    }
    vector<vector<size_t> > neighCutoff(n);
    size_t count = 0;
    for (size_t i = 0; i < n; ++i)
    {
        neighCutoff.at(i).resize(n, 0);
        for (size_t j = 0; j < n; ++j)
        {
            neighCutoff.at(i).at(j) = atof(argv[count+2]);
            count++;
        }
    }
    useForces = dataset.settingsKeywordExists("use_short_forces");

    dataset.log << "\n";
    dataset.log << "*** CALCULATING SYMMETRY FUNCTIONS ******"
                   "**************************************\n";
    dataset.log << "\n";

    for (vector<Structure>::iterator it = dataset.structures.begin();
         it != dataset.structures.end(); ++it)
    {
        it->calculateNeighborList(dataset.getMaxCutoffRadius());
#ifdef NNP_NO_SF_GROUPS
        dataset.calculateSymmetryFunctions((*it), useForces);
#else
        dataset.calculateSymmetryFunctionGroups((*it), useForces);
#endif
        // Clear unneccessary memory, don't use these structures after these
        // operations unless you know what you do!
        for (vector<Atom>::iterator it2 = it->atoms.begin();
             it2 != it->atoms.end(); ++it2)
        {
            it2->numNeighborsUnique = 0;
            it2->neighborsUnique.clear();
            vector<size_t>(it2->neighborsUnique).swap(it2->neighborsUnique);

            it2->dEdG.clear();
            vector<double>(it2->dEdG).swap(it2->dEdG);

#ifdef NNP_FULL_SFD_MEMORY
            it2->dGdxia.clear();
            vector<double>(it2->dGdxia).swap(it2->dGdxia);
#endif

            if (!useForces)
            {
                it2->dGdr.clear();
                vector<Vec3D>(it2->dGdr).swap(it2->dGdr);
            }
        }
    }
    dataset.log << "*****************************************"
                   "**************************************\n";

    dataset.collectSymmetryFunctionStatistics();
    dataset.writeSymmetryFunctionHistograms(numBins,
                                            "sf-scaled.%03zu.%04zu.histo");
    dataset.writeNeighborHistogram();
    dataset.sortNeighborLists();
    dataset.writeNeighborLists();
    dataset.writeAtomicEnvironmentFile(neighCutoff, useForces);

    myLog.close();

    MPI_Finalize();

    return 0;
}
