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

//define DEBUG

int main(int argc, char* argv[])
{
    bool                   useForces = false;
    int                    numProcs  = 0;
    int                    myRank    = 0;
    long                   memory    = 0;
    size_t                 nconf     = 0;
    size_t                 count     = 0;
    vector<int>            stypes;
    vector<bool>           statflag;
    vector<vector<double>> Gij;
    ofstream               myLog;

    if (argc != 3)
    {
        cout << "USAGE: " << argv[0] << " <nconfig> <mode>\n"
             << "       <nconfig> ... Number of configuration to keep\n"
             << "       <mode>    ... 1 for memory saving (cpu expensive),\n"
             << "                 ... 0 for memory intensive "
                "(faster runtime).\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - input.nn (NNP settings)\n"
             << "SERIAL mode only, no mpirun\n";
        return 1;
    }

    size_t numConfig = (size_t)atoi(argv[1]);
    bool memflag = (bool)atoi(argv[2]);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    Dataset dataset;
    myLog.open(strpr("nnp-fps.log.%04d", myRank).c_str());
    if (numProcs > 1) 
    {
        throw runtime_error("ERROR: Use this application only in serial "
                            "mode!");
    }
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

    dataset.log << "\n";
    dataset.log << "*** FARTHEST POINT SAMPLING *************"
                   "**************************************\n";
    dataset.log << "\n";
    if (memflag == true)
    {
        dataset.log << "memory saving mode true (cpu expensive)\n";
    }
    
    dataset.log << "Check the log files of all (!) MPI processes"
                   " for warnings in this section!\n";

    stypes.resize(dataset.structures.size(), 0);
    Gij.resize(dataset.structures.size(),
               vector<double>(dataset.structures.size()));

    int is = 0;
    int tp = 1;
    dataset.log << strpr("size of dataset: %zu\n\n",
                         dataset.structures.size());
    for (vector<Structure>::iterator it = dataset.structures.begin();
         it != dataset.structures.end(); ++it)
    {
        if (stypes[is] == 0)
        {
            stypes[is] = tp;
            int js = is + 1;
            for (vector<Structure>::iterator jt = it + 1;
                 jt != dataset.structures.end(); ++jt)
            {
                if (it->numAtomsPerElement == jt->numAtomsPerElement)
                {
                    stypes[js]=tp;
                }   
                js++;
            }
            tp++;
        }
        dataset.log << strpr("Structure types (stypes): %d  %d\n",
                             is, stypes[is]);
        is++;
    }

    // is = 0;
    // calculate Gij matrix (distance of symmetry functions)
    for (vector<Structure>::iterator it = dataset.structures.begin();
         it != dataset.structures.end(); ++it)
    {  
        // dataset.log << strpr("neighborlist step: %d\n", is);
        it->hasNeighborList = false;
        //calculateNeighborList(dataset.getMaxCutoffRadius());
        it->hasallG = false;
        // is++;
    }
    dataset.log << "starting with symmetry function calculations.\n";

    for (size_t i = 0; i < dataset.structures.size(); i++)
    {
#ifdef DEBUG
        dataset.log << strpr(
                         "hasSymmetryFunctionDerivative %d\n",
                         dataset.structures[i].hasSymmetryFunctionDerivatives);
        dataset.log << strpr("hasSymmetryFunctions %d\n",
                             dataset.structures[i].hasSymmetryFunctions);
#endif
        if (dataset.structures[i].hasallG == false)
        {
            if (dataset.structures[i].hasNeighborList==false)
            {
                dataset.structures[i].calculateNeighborList(
                                                 dataset.getMaxCutoffRadius());
            }
            dataset.calculateSymmetryFunctions(dataset.structures[i],
                                               useForces);
            dataset.structures[i].updateallG();
        }
        if (i % 10 ==0)
        {
            dataset.log << strpr(" step outer loop: %zu\n", i);
        }
      
#ifdef DEBUG
        for (size_t k = 0; k < dataset.structures[i].atoms.size(); k++)
        {
            dataset.log << strpr("size of atoms %d: %ld\n" ,
                                 k, dataset.structures[i].atoms[k].G.size()); 
        }
        dataset.log << strpr("size of allG: %zu\n",
                             dataset.structures[i].allG.size()); 
        dataset.log << strpr("size allG[0]: %zu\n",
                             dataset.structures[i].allG[0].size());
        if (i < 3)
        {
            dataset.log << strpr("DEBUG _i_ = %zu\n", i);
        }
        for (int k = 0; k < dataset.structures[i].allG.size(); k++)
        {
            if ((k == 0) && (i < 3))
            {
                dataset.log << strpr("          element = %d\n", k);
            }
            for (int m = 0; m < dataset.structures[i].allG[k].size(); m++)
            {
                if ((k == 0) && (m < 4) && (i < 3))
                {
                    dataset.log << strpr("              (%d)  Gvalue = %lf\n",
                                         m, dataset.structures[i].allG[k][m]);
                }
            }
        }
#endif
    
        for(size_t j = i + 1; j < dataset.structures.size(); j++)
        {
            if ((j % 1000) == 0)
            {
                dataset.log << strpr("step inner loop: %zu\n", j);
            }
            if (stypes[i] == stypes[j])
            {
                double dist = 0;
                if (dataset.structures[j].hasallG == false)
                {
                    if (dataset.structures[j].hasNeighborList == false)
                    {
                        dataset.structures[j].calculateNeighborList(
                                                 dataset.getMaxCutoffRadius());
                    }
                    dataset.calculateSymmetryFunctions(dataset.structures[j],
                                                       useForces);
                    dataset.structures[j].updateallG();
                }
                int nG = 0;
    
                for (size_t k = 0;k < dataset.structures[i].allG.size(); k++)
                {
                    for (size_t m = 0; m < dataset.structures[i].allG[k].size();
                         m++)
                    {
                        dist += abs(dataset.structures[i].allG[k][m]
                                    -dataset.structures[j].allG[k][m]);
    
#ifdef DEBUG
                        if ((k == 0) && (m < 3) && (i < 2) && (j < 3) )
                        {
                            dataset.log << strpr(
                                          "              (%d)  Gvalue = %lf\n",
                                          m,
                                          dataset.structures[j].allG[k][m]);
                        }
#endif
                    }
                    nG += dataset.structures[i].allG[k].size();
                }
                dist /= nG;
                Gij[i][j] = dist;
                Gij[j][i] = dist;
                for (vector<Atom>::iterator it2 =
                     dataset.structures[j].atoms.begin();
                     it2 != dataset.structures[j].atoms.end(); ++it2)
                {
                    it2->G.clear();
                    it2->hasSymmetryFunctions = false;
                    vector<double>(it2->G).swap(it2->G);
                }
                dataset.structures[j].hasSymmetryFunctions = false;
                // clear memory of structure j
                if(memflag == true)
                {
                    dataset.structures[j].clearallG();
                    dataset.structures[j].clearNeighborList();
                }
            }
        }
        //clear memory of structure i   
        for (vector<Atom>::iterator it2 = dataset.structures[i].atoms.begin();
             it2 != dataset.structures[i].atoms.end(); ++it2)
        {
            it2->G.clear();
            it2->hasSymmetryFunctions = false;
            vector<double>(it2->G).swap(it2->G);
        }
        dataset.structures[i].hasSymmetryFunctions = false;
        if(memflag==true)
        {
            dataset.structures[i].clearallG();
            dataset.structures[i].clearNeighborList();
        }
    }
   


    statflag.resize(dataset.structures.size(), false);
    nconf = 0;
    // initialize statflag to (1,0,0,0,0...,1,0,0,....1,0,0,...) for first
    // appearance of a new stype
    for (int t = 1; t < tp; t++)
    {
        for(size_t i = 0;i < stypes.size(); i++)
        {
            if (stypes[i] == t)
            {
                statflag[i] = 1;
                nconf++;
                break;
            }
        }
    }

#ifdef DEBUG
    for (int i = 0; i < statflag.size(); i++)
    {
        if (statflag[i] == 1)
        {
            dataset.log << strpr("initial statflag 1 for i= %zu\n", i); 
        }
    }
#endif

    // farthest point samplign algorithm for symmetry function "distances" Gij
    while (nconf < numConfig)
    {
        int    imax    = 0;
        // int    jselect = 0;
        double dmax    = 0;
        double dijmin  = 0;
        double dist    = 0;
        for(size_t i = 0; i < statflag.size(); i++)
        {
            if (statflag[i] == 0)
            {
                dijmin = 1.7e300; // a very large number
                for(size_t j = 0; j < statflag.size(); j++)
                {
                    if (j != i && stypes[i] == stypes[j])
                    {
                        if (statflag[j] == 1)
                        {
                            // int nG = 0;
                            dist = Gij[i][j];
                            // dataset.log << strpr("Gdistance: %f\n", dist);
                            if (dist < dijmin)
                            {
                                dijmin = dist;
                                // jselect = j;
                            }
                        }
                    
                    }
                }
                // dataset.log << "\n";
                if (dijmin > dmax)
                {
                    dmax = dijmin;
                    imax = i;
                }
            }
        }
        // dataset.log << strpr("Gdistance max chosen: %f\n", dmax);
        // dataset.log << strpr("Structure chosen: %d\n", imax);
        statflag[imax] = 1;
        nconf++;
        if (nconf % 10 == 0)
        {
            dataset.log << strpr("number of configurations already "
                                 "chosen: %zu\n", nconf);
        }
    }
    dataset.log << strpr("number of configurations chosen in total: %zu\n", 
                         nconf);

    dataset.log << "\n";   
    for (size_t i = 0; i < statflag.size(); i++)
    {
        if (statflag[i] == 1)
        {
            dataset.log << strpr("chosen structures: %zu\n", i);
        }
    }

    //dirty writing of output.data file    
    ifstream inputFile;
    ofstream outputFile;
    bool writeStructure = false;

    inputFile.open("input.data");
    outputFile.open("output.data");
    string line;
    is = -1;
    while (getline(inputFile, line))
    {
        if (split(reduce(line)).at(0) == "begin")
        {   
            is++;
            if (statflag[is] == 1)
            {
                writeStructure = true;
            }
            else writeStructure = false;
        }
        if (writeStructure)
        {
            outputFile << line << '\n';
        }
    }
    inputFile.close();
    outputFile.close();

    MPI_Allreduce(MPI_IN_PLACE, &memory, 1, MPI_LONG  , MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &count , 1, MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);

    myLog.close();

    MPI_Finalize();

    return 0;
}
