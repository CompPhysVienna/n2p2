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
    bool   useForces = false;
    int    numProcs  = 0;
    int    myRank    = 0;
    long   memory    = 0;
    size_t count     = 0;
    vector< int > stypes;
    vector< bool > statflag;
    vector< vector < double> > Gij;
    int nconf;
    ofstream myLog;

    if (argc != 3)
    {
        cout << "USAGE: " << argv[0] << " <nconfig> <mode>\n"
             << "       <nconfig> ... Number of configuration to keep\n"
	     << "       <mode>    ... 1 for memory saving (cpu expensive),\n"
	     << "                 ... 0 for memory intensive (faster runtime).\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n"
             << "       - input.nn (NNP settings)\n"
             << "SERIAL mode only, no mpirun\n";
        return 1;
    }

    size_t numConfig = (size_t)atoi(argv[1]);
    bool memflag= (bool)atoi(argv[2]);
    if (memflag == true){
	cout << "memory saving mode true (cpu expensive)\n";
    }
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

    Dataset dataset;
    myLog.open(strpr("nnp-fpssampling.log.%04d", myRank).c_str());
    if (myRank != 0) 
        {
	cout << "Use this application only in serial mode!";
	return 1;
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
    dataset.log << "*** CALCULATING SYMMETRY FUNCTIONS ******"
                   "**************************************\n";
    dataset.log << "\n";

    dataset.log << "Check the log files of all (!) MPI processes"
                   " for warnings in this section!\n";

    stypes.resize(dataset.structures.size(),0);
    Gij.resize(dataset.structures.size(),vector<double>(dataset.structures.size()));

    int is=0;
    int tp=1;
    cout << "size of dataset:" << dataset.structures.size() << "\n\n";
    for (vector<Structure>::iterator it = dataset.structures.begin();
         it != dataset.structures.end(); ++it)
    {


	if (stypes[is]==0){
		stypes[is]=tp;
		int js=is+1;
		for (vector<Structure>::iterator jt = it+1; jt != dataset.structures.end(); ++jt){
			if (it->numAtomsPerElement==jt->numAtomsPerElement){
				stypes[js]=tp;
		}	
		js=js+1;
		}
		tp=tp+1;
	}
	cout << "Structure types (stypes): " << is << "  " << stypes[is] << "\n";
	is = is+1;
    }

//is=0;
// calculate Gij matrix (distance of symmetry functions)
    for (vector<Structure>::iterator it = dataset.structures.begin();
         it != dataset.structures.end(); ++it)
    {  //cout << "neighborlist step:" << is << "\n";
        it->hasNeighborList=false;
//calculateNeighborList(dataset.getMaxCutoffRadius());
	it->hasallG=false;
  //      is=is+1;
    }
cout << "starting with symmetry function calculations.\n";

    for(int i=0;i<dataset.structures.size();i++){
#ifdef DEBUG
cout << "hasSymmetryFunctionDerivative " << dataset.structures[i].hasSymmetryFunctionDerivatives << endl; 
cout << "hasSymmetryFunctions " << dataset.structures[i].hasSymmetryFunctions << endl; 
#endif
    if (dataset.structures[i].hasallG==false){
	if (dataset.structures[i].hasNeighborList==false){
		dataset.structures[i].calculateNeighborList(dataset.getMaxCutoffRadius());
	}
        dataset.calculateSymmetryFunctions(dataset.structures[i],useForces);
        dataset.structures[i].updateallG(&myLog);
    }
	if (i % 10 ==0){
          cout << " step outer loop: " << i <<"\n";
	}
  
#ifdef DEBUG
for (int k=0;k<dataset.structures[i].atoms.size();k++){
    printf("size of atoms %d: %ld\n" ,k, dataset.structures[i].atoms[k].G.size()); 
}
cout << "size of allG: " << dataset.structures[i].allG.size() << "\n"; 
cout << "size allG[0]: " << dataset.structures[i].allG[0].size() << "\n"; 
if ((i < 3)) {
  printf("DEBUG _i_ = %d\n", i);
}
for (int k=0;k<dataset.structures[i].allG.size();k++){
  if ((k == 0) && (i < 3)) {
    printf("          element = %d\n", k);
  }
  for (int m=0;m<dataset.structures[i].allG[k].size();m++){
	if ((k == 0) && (m < 4) && (i < 3)) {
   printf("              (%d)  Gvalue = %lf\n", m, dataset.structures[i].allG[k][m]);
	}
  }
}
#endif

	for(int j=i+1;j<dataset.structures.size();j++){
 	        if ((j % 1000) == 0){
                    cout << "step inner loop: " << j << "\n"; 
	        }
		if (stypes[i]==stypes[j]){
			double dist=0;
			if (dataset.structures[j].hasallG==false){
	 		  if (dataset.structures[j].hasNeighborList==false){
				dataset.structures[j].calculateNeighborList(dataset.getMaxCutoffRadius());
			  }
       			  dataset.calculateSymmetryFunctions(dataset.structures[j],useForces);
        		  dataset.structures[j].updateallG(&myLog);
			}
			int nG=0;

			for (int k=0;k<dataset.structures[i].allG.size();k++){
			  for (int m=0;m<dataset.structures[i].allG[k].size();m++){
				  dist=dist+abs(dataset.structures[i].allG[k][m]-dataset.structures[j].allG[k][m]);

#ifdef DEBUG
if ((k == 0) && (m < 3) && (i < 2) && (j < 3) ) {
  printf("              (%d)  Gvalue = %lf\n", m, dataset.structures[j].allG[k][m]);
}
#endif
			  }
			  nG=nG+dataset.structures[i].allG[k].size();
			}
		dist=dist/nG;
		Gij[i][j]=dist;
		Gij[j][i]=dist;
		for (vector<Atom>::iterator it2 = dataset.structures[j].atoms.begin();
			it2 != dataset.structures[j].atoms.end(); ++it2)
        	{
		it2->G.clear();
		it2->hasSymmetryFunctions = false;
                vector<double>(it2->G).swap(it2->G);
		}
		dataset.structures[j].hasSymmetryFunctions = false;
		// clear memory of structure j
		if(memflag==true){
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
 	if(memflag==true){
	  dataset.structures[i].clearallG();
	  dataset.structures[i].clearNeighborList();
	}
    }
   


    statflag.resize(dataset.structures.size(),false);
    nconf=0;
//initialize statflag to (1,0,0,0,0...,1,0,0,....1,0,0,...) for first appearance of a new stype
    for(int t = 1; t < tp; t++){
    	for(int i = 0;i<stypes.size();i++){
		if (stypes[i]==t){
		statflag[i]=1;
		nconf=nconf+1;
		break;
		}
	}
    }

#ifdef DEBUG
    for (int i=0;i<statflag.size();i++){
	if (statflag[i]==1){
	      cout << "initial statflag 1 for i= " << i << "\n"; 
	}
    }
#endif

//farthest point samplign algorithm for symmetry function "distances" Gij
    while(nconf<numConfig){
	double dmax=0,dijmin,dist;
	int imax,jselect;
	for(int i=0;i<statflag.size();i++){
	if (statflag[i]==0){
		dijmin=1.7e300; // a very large number
		for(int j=0;j<statflag.size();j++){
		if (j!=i and stypes[i]==stypes[j]){
			if (statflag[j]==1){
				int nG=0;
				dist=Gij[i][j];
//				cout << "Gdistance: " << dist << "\n";
				if (dist<dijmin){
				dijmin=dist;
				jselect=j;
				}
			}
		
		}
		}
//		cout << "\n";
		if (dijmin>dmax){
			dmax=dijmin;
			imax=i;
		}
	}
	}
//	cout << "Gdistance max chosen: " << dmax << "\n";
//	cout << "Structure chosen: " << imax << "\n";
	statflag[imax]=1;
	nconf=nconf+1;
	if (nconf % 10 == 0){
	cout << "number of configurations already chosen: " << nconf << "\n";
	}
    }
    cout << "number of configurations chosen in total: " << nconf << "\n";

    cout << "\n";	
    for (int i=0;i<statflag.size();i++){
            if (statflag[i]==1){
	    cout << "chosen structures: " << i << "\n";
	    }
    }

//dirty writing of output.data file    
    ifstream inputFile;
    ofstream outputFile;
    bool writeStructure;

    inputFile.open("input.data");
    outputFile.open("output.data");
    string line;
    is=-1;
    while (getline(inputFile, line))
    {
        if (split(reduce(line)).at(0) == "begin")
        {   is=is+1;
            if (statflag[is]==1)
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


