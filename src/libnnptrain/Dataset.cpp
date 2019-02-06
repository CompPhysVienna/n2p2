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
#include "SymmetryFunction.h"
#include "mpi-extra.h"
#include "utility.h"
#include <algorithm> // std::max, std::find
#include <cmath>     // sqrt, fabs
#include <cstdlib>   // atoi
#include <cstdio>    // fprintf, fopen, fclose, remove
#include <iostream>  // std::ios::binary
#include <fstream>   // std::ifstream, std::ofstream
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_rng.h>

using namespace std;
using namespace nnp;

Dataset::Dataset() : Mode(),
                     myRank       (0   ),
                     numProcs     (0   ),
                     numStructures(0   ),
                     myName       (""  ),
                     rng          (NULL),
                     rngGlobal    (NULL)
{
}

Dataset::~Dataset()
{
    if (rng != NULL) gsl_rng_free(rng);
    if (rngGlobal != NULL) gsl_rng_free(rngGlobal);
}

void Dataset::setupMPI()
{
    MPI_Comm tmpComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tmpComm);
    setupMPI(&tmpComm);
    MPI_Comm_free(&tmpComm);

    return;
}

void Dataset::setupMPI(MPI_Comm* communicator)
{
    log << "\n";
    log << "*** SETUP: MPI **************************"
           "**************************************\n";
    log << "\n";

    int        bufferSize = 0;
    char       line[MPI_MAX_PROCESSOR_NAME];
    MPI_Status ms;

    MPI_Comm_dup(*communicator, &comm);
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &numProcs);
    MPI_Get_processor_name(line, &bufferSize);
    myName = line;

    log << strpr("Number of processors: %d\n", numProcs);
    log << strpr("Process %d of %d (rank %d): %s\n",
                 myRank + 1,
                 numProcs,
                 myRank,
                 myName.c_str());
    for (int i = 1; i < numProcs; ++i)
    {
        if (myRank == 0)
        {
            MPI_Recv(&bufferSize, 1, MPI_INT, i, 0, comm, &ms);
            MPI_Recv(line, bufferSize, MPI_CHAR, i, 0, comm, &ms);
            log << strpr("Process %d of %d (rank %d): %s\n",
                         i + 1,
                         numProcs,
                         i,
                         line);
        }
        else if (myRank == i)
        {
            MPI_Send(&bufferSize, 1, MPI_INT, 0, 0, comm);
            MPI_Send(line, bufferSize, MPI_CHAR, 0, 0, comm);
        }
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Dataset::setupRandomNumberGenerator()
{
    log << "\n";
    log << "*** SETUP: RANDOM NUMBER GENERATOR ******"
           "**************************************\n";
    log << "\n";

    // Get random seed from settings file.
    unsigned long seed = atoi(settings["random_seed"].c_str());
    unsigned long seedGlobal = 0;

    if (myRank == 0)
    {
        log << strpr("Random number generator seed: %d\n", seed);
        if (seed == 0)
        {
            log << "WARNING: Seed set to 0. This is a special value for the "
                   "gsl_rng_set() routine (see GSL docs).\n";
        }
        // Initialize personal RNG of process 0 with the given seed.
        rng = gsl_rng_alloc(gsl_rng_mt19937);
        gsl_rng_set(rng, seed);
        log << strpr("Seed for rank %d: %lu\n", 0, seed);
        for (int i = 1; i < numProcs; ++i)
        {
            // Get new seeds for all remaining processes with the RNG.
            seed = gsl_rng_get(rng);
            log << strpr("Seed for rank %d: %lu\n", i, seed);
            MPI_Send(&seed, 1, MPI_UNSIGNED_LONG, i, 0, comm);
        }
        // Set seed for global RNG.
        seedGlobal = gsl_rng_get(rng);
    }
    else
    {
        // Receive seed for personal RNG.
        MPI_Status ms;
        MPI_Recv(&seed, 1, MPI_UNSIGNED_LONG, 0, 0, comm, &ms);
        log << strpr("Seed for rank %d: %lu\n", myRank, seed);
        rng = gsl_rng_alloc(gsl_rng_taus);
        gsl_rng_set(rng, seed);
    }
    // Rank 0 broadcasts global seed.
    MPI_Bcast(&seedGlobal, 1, MPI_UNSIGNED_LONG, 0, comm);
    log << strpr("Seed for global RNG: %lu\n", seedGlobal);
    // All processes initialize global RNG.
    rngGlobal = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(rngGlobal, seedGlobal);

    log << "*****************************************"
           "**************************************\n";

    return;
}

int Dataset::calculateBufferSize(Structure const& structure) const
{
    int              bs  = 0;         // Send buffer size.
    int              is  = 0;         // int size.
    int              ss  = 0;         // size_t size.
    int              ds  = 0;         // double size.
    int              cs  = 0;         // char size.
    Structure const& s   = structure; // Shortcut for structure.

    MPI_Pack_size(1, MPI_INT   , comm, &is);
    MPI_Pack_size(1, MPI_SIZE_T, comm, &ss);
    MPI_Pack_size(1, MPI_DOUBLE, comm, &ds);
    MPI_Pack_size(1, MPI_CHAR  , comm, &cs);

    // Structure
    bs += 5 * cs + 4 * ss + 4 * is + 4 * ds;
    // Structure.comment
    bs += ss;
    bs += (s.comment.length() + 1) * cs;
    // Structure.box
    bs += 9 * ds;
    // Structure.invbox
    bs += 9 * ds;
    // Structure.numAtomsPerElement
    bs += ss;
    bs += s.numAtomsPerElement.size() * ss;
    // Structure.atoms
    bs += ss;
    bs += s.atoms.size() * (3 * cs + 7 * ss + 2 * ds + 3 * 3 * ds);
    for (vector<Atom>::const_iterator it = s.atoms.begin();
         it != s.atoms.end(); ++it)
    {
        // Atom.neighborsUnique
        bs += ss;
        bs += it->neighborsUnique.size() * ss;
        // Atom.numNeighborsPerElement
        bs += ss;
        bs += it->numNeighborsPerElement.size() * ss;
        // Atom.G
        bs += ss;
        bs += it->G.size() * ds;
        // Atom.dEdG
        bs += ss;
        bs += it->dEdG.size() * ds;
        // Atom.dGdxia
        bs += ss;
        bs += it->dGdxia.size() * ds;
        // Atom.dGdr
        bs += ss;
        bs += it->dGdr.size() * 3 * ds;
        // Atom.neighbors
        bs += ss;
        for (vector<Atom::Neighbor>::const_iterator it2 =
             it->neighbors.begin(); it2 != it->neighbors.end(); ++it2)
        {
            // Neighbor
            bs += 3 * ss + 5 * ds + is + 3 * ds;

            // Neighbor.dGdr
            bs += ss;
            bs += it2->dGdr.size() * 3 * ds;
        }
    }

    return bs;
}

int Dataset::sendStructure(Structure const& structure, int dest) const
{
    unsigned char*   buf = 0;         // Send buffer.
    int              bs  = 0;         // Send buffer size.
    int              p   = 0;         // Send buffer position.
    int              ts  = 0;         // Size for temporary stuff.
    Structure const& s   = structure; // Shortcut for structure.

    bs = calculateBufferSize(s);
    buf = new unsigned char[bs];

    // Structure
    MPI_Pack(&(s.isPeriodic                    ), 1, MPI_CHAR  , buf, bs, &p, comm);
    MPI_Pack(&(s.isTriclinic                   ), 1, MPI_CHAR  , buf, bs, &p, comm);
    MPI_Pack(&(s.hasNeighborList               ), 1, MPI_CHAR  , buf, bs, &p, comm);
    MPI_Pack(&(s.hasSymmetryFunctions          ), 1, MPI_CHAR  , buf, bs, &p, comm);
    MPI_Pack(&(s.hasSymmetryFunctionDerivatives), 1, MPI_CHAR  , buf, bs, &p, comm);
    MPI_Pack(&(s.index                         ), 1, MPI_SIZE_T, buf, bs, &p, comm);
    MPI_Pack(&(s.numAtoms                      ), 1, MPI_SIZE_T, buf, bs, &p, comm);
    MPI_Pack(&(s.numElements                   ), 1, MPI_SIZE_T, buf, bs, &p, comm);
    MPI_Pack(&(s.numElementsPresent            ), 1, MPI_SIZE_T, buf, bs, &p, comm);
    MPI_Pack(&(s.pbc                           ), 3, MPI_INT   , buf, bs, &p, comm);
    MPI_Pack(&(s.energy                        ), 1, MPI_DOUBLE, buf, bs, &p, comm);
    MPI_Pack(&(s.energyRef                     ), 1, MPI_DOUBLE, buf, bs, &p, comm);
    MPI_Pack(&(s.chargeRef                     ), 1, MPI_DOUBLE, buf, bs, &p, comm);
    MPI_Pack(&(s.volume                        ), 1, MPI_DOUBLE, buf, bs, &p, comm);
    MPI_Pack(&(s.sampleType                    ), 1, MPI_INT   , buf, bs, &p, comm);

    // Strucuture.comment
    ts = s.comment.length() + 1;
    MPI_Pack(&ts, 1, MPI_SIZE_T, buf, bs, &p, comm);
    MPI_Pack(s.comment.c_str(), ts, MPI_CHAR, buf, bs, &p, comm);

    // Structure.box
    for (size_t i = 0; i < 3; ++i)
    {
        MPI_Pack(s.box[i].r, 3, MPI_DOUBLE, buf, bs, &p, comm);
    }

    // Structure.invbox
    for (size_t i = 0; i < 3; ++i)
    {
        MPI_Pack(s.invbox[i].r, 3, MPI_DOUBLE, buf, bs, &p, comm);
    }

    // Structure.numAtomsPerElement
    ts = s.numAtomsPerElement.size();
    MPI_Pack(&ts, 1, MPI_SIZE_T, buf, bs, &p, comm);
    if (ts > 0)
    {
        MPI_Pack(&(s.numAtomsPerElement.front()), ts, MPI_SIZE_T, buf, bs, &p, comm);
    }

    // Structure.atoms
    ts = s.atoms.size();
    MPI_Pack(&ts, 1, MPI_SIZE_T, buf, bs, &p, comm);
    if (ts > 0)
    {
        for (vector<Atom>::const_iterator it = s.atoms.begin();
             it != s.atoms.end(); ++it)
        {
            // Atom
            MPI_Pack(&(it->hasNeighborList               ), 1, MPI_CHAR  , buf, bs, &p, comm);
            MPI_Pack(&(it->hasSymmetryFunctions          ), 1, MPI_CHAR  , buf, bs, &p, comm);
            MPI_Pack(&(it->hasSymmetryFunctionDerivatives), 1, MPI_CHAR  , buf, bs, &p, comm);
            MPI_Pack(&(it->index                         ), 1, MPI_SIZE_T, buf, bs, &p, comm);
            MPI_Pack(&(it->indexStructure                ), 1, MPI_SIZE_T, buf, bs, &p, comm);
            MPI_Pack(&(it->tag                           ), 1, MPI_SIZE_T, buf, bs, &p, comm);
            MPI_Pack(&(it->element                       ), 1, MPI_SIZE_T, buf, bs, &p, comm);
            MPI_Pack(&(it->numNeighbors                  ), 1, MPI_SIZE_T, buf, bs, &p, comm);
            MPI_Pack(&(it->numNeighborsUnique            ), 1, MPI_SIZE_T, buf, bs, &p, comm);
            MPI_Pack(&(it->numSymmetryFunctions          ), 1, MPI_SIZE_T, buf, bs, &p, comm);
            MPI_Pack(&(it->energy                        ), 1, MPI_DOUBLE, buf, bs, &p, comm);
            MPI_Pack(&(it->charge                        ), 1, MPI_DOUBLE, buf, bs, &p, comm);
            MPI_Pack(&(it->r.r                           ), 3, MPI_DOUBLE, buf, bs, &p, comm);
            MPI_Pack(&(it->f.r                           ), 3, MPI_DOUBLE, buf, bs, &p, comm);
            MPI_Pack(&(it->fRef.r                        ), 3, MPI_DOUBLE, buf, bs, &p, comm);

            // Atom.neighborsUnique
            size_t ts2 = it->neighborsUnique.size();
            MPI_Pack(&ts2, 1, MPI_SIZE_T, buf, bs, &p, comm);
            if (ts2 > 0)
            {
                MPI_Pack(&(it->neighborsUnique.front()), ts2, MPI_SIZE_T, buf, bs, &p, comm);
            }

            // Atom.numNeighborsPerElement
            ts2 = it->numNeighborsPerElement.size();
            MPI_Pack(&ts2, 1, MPI_SIZE_T, buf, bs, &p, comm);
            if (ts2 > 0)
            {
                MPI_Pack(&(it->numNeighborsPerElement.front()), ts2, MPI_SIZE_T, buf, bs, &p, comm);
            }

            // Atom.G
            ts2 = it->G.size();
            MPI_Pack(&ts2, 1, MPI_SIZE_T, buf, bs, &p, comm);
            if (ts2 > 0)
            {
                MPI_Pack(&(it->G.front()), ts2, MPI_DOUBLE, buf, bs, &p, comm);
            }

            // Atom.dEdG
            ts2 = it->dEdG.size();
            MPI_Pack(&ts2, 1, MPI_SIZE_T, buf, bs, &p, comm);
            if (ts2 > 0)
            {
                MPI_Pack(&(it->dEdG.front()), ts2, MPI_DOUBLE, buf, bs, &p, comm);
            }

            // Atom.dGdxia
            ts2 = it->dGdxia.size();
            MPI_Pack(&ts2, 1, MPI_SIZE_T, buf, bs, &p, comm);
            if (ts2 > 0)
            {
                MPI_Pack(&(it->dGdxia.front()), ts2, MPI_DOUBLE, buf, bs, &p, comm);
            }

            // Atom.dGdr
            ts2 = it->dGdr.size();
            MPI_Pack(&ts2, 1, MPI_SIZE_T, buf, bs, &p, comm);
            if (ts2 > 0)
            {
                for (vector<Vec3D>::const_iterator it2 = it->dGdr.begin();
                     it2 != it->dGdr.end(); ++it2)
                {
                    MPI_Pack(it2->r, 3, MPI_DOUBLE, buf, bs, &p, comm);
                }
            }

            // Atom.neighbors
            ts2 = it->neighbors.size();
            MPI_Pack(&ts2, 1, MPI_SIZE_T, buf, bs, &p, comm);
            if (ts2 > 0)
            {
                for (vector<Atom::Neighbor>::const_iterator it2 =
                     it->neighbors.begin(); it2 != it->neighbors.end(); ++it2)
                {
                    // Neighbor
                    MPI_Pack(&(it2->index      ), 1, MPI_SIZE_T, buf, bs, &p, comm);
                    MPI_Pack(&(it2->tag        ), 1, MPI_SIZE_T, buf, bs, &p, comm);
                    MPI_Pack(&(it2->element    ), 1, MPI_SIZE_T, buf, bs, &p, comm);
                    MPI_Pack(&(it2->d          ), 1, MPI_DOUBLE, buf, bs, &p, comm);
                    MPI_Pack(&(it2->fc         ), 1, MPI_DOUBLE, buf, bs, &p, comm);
                    MPI_Pack(&(it2->dfc        ), 1, MPI_DOUBLE, buf, bs, &p, comm);
                    MPI_Pack(&(it2->rc         ), 1, MPI_DOUBLE, buf, bs, &p, comm);
                    MPI_Pack(&(it2->cutoffAlpha), 1, MPI_DOUBLE, buf, bs, &p, comm);
                    MPI_Pack(&(it2->cutoffType ), 1, MPI_INT   , buf, bs, &p, comm);
                    MPI_Pack(  it2->dr.r        , 3, MPI_DOUBLE, buf, bs, &p, comm);

                    // Neighbor.dGdr
                    size_t ts3 = it2->dGdr.size();
                    MPI_Pack(&ts3, 1, MPI_SIZE_T, buf, bs, &p, comm);
                    if (ts3 > 0)
                    {
                        for (vector<Vec3D>::const_iterator it3 =
                             it2->dGdr.begin(); it3 != it2->dGdr.end(); ++it3)
                        {
                            MPI_Pack(it3->r, 3, MPI_DOUBLE, buf, bs, &p, comm);
                        }
                    }
                }
            }
        }
    }

    MPI_Send(&bs, 1, MPI_INT, dest, 0, comm);
    MPI_Send(buf, bs, MPI_PACKED, dest, 0, comm);

    delete[] buf;

    return bs;
}

int Dataset::recvStructure(Structure* const structure, int src)
{
    unsigned char*   buf = 0;         // Receive buffer.
    int              bs  = 0;         // Receive buffer size.
    int              p   = 0;         // Receive buffer position.
    int              ts  = 0;         // Size for temporary stuff.
    Structure* const s   = structure; // Shortcut for structure.
    MPI_Status       ms;

    // Receive buffer size and extract source.
    MPI_Recv(&bs, 1, MPI_INT, src, 0, comm, &ms);
    src = ms.MPI_SOURCE;

    buf = new unsigned char[bs];

    MPI_Recv(buf, bs, MPI_PACKED, src, 0, comm, &ms);

    // Structure
    MPI_Unpack(buf, bs, &p, &(s->isPeriodic                    ), 1, MPI_CHAR  , comm);
    MPI_Unpack(buf, bs, &p, &(s->isTriclinic                   ), 1, MPI_CHAR  , comm);
    MPI_Unpack(buf, bs, &p, &(s->hasNeighborList               ), 1, MPI_CHAR  , comm);
    MPI_Unpack(buf, bs, &p, &(s->hasSymmetryFunctions          ), 1, MPI_CHAR  , comm);
    MPI_Unpack(buf, bs, &p, &(s->hasSymmetryFunctionDerivatives), 1, MPI_CHAR  , comm);
    MPI_Unpack(buf, bs, &p, &(s->index                         ), 1, MPI_SIZE_T, comm);
    MPI_Unpack(buf, bs, &p, &(s->numAtoms                      ), 1, MPI_SIZE_T, comm);
    MPI_Unpack(buf, bs, &p, &(s->numElements                   ), 1, MPI_SIZE_T, comm);
    MPI_Unpack(buf, bs, &p, &(s->numElementsPresent            ), 1, MPI_SIZE_T, comm);
    MPI_Unpack(buf, bs, &p, &(s->pbc                           ), 3, MPI_INT   , comm);
    MPI_Unpack(buf, bs, &p, &(s->energy                        ), 1, MPI_DOUBLE, comm);
    MPI_Unpack(buf, bs, &p, &(s->energyRef                     ), 1, MPI_DOUBLE, comm);
    MPI_Unpack(buf, bs, &p, &(s->chargeRef                     ), 1, MPI_DOUBLE, comm);
    MPI_Unpack(buf, bs, &p, &(s->volume                        ), 1, MPI_DOUBLE, comm);
    MPI_Unpack(buf, bs, &p, &(s->sampleType                    ), 1, MPI_INT   , comm);

    // Strucuture.comment
    ts = 0;
    MPI_Unpack(buf, bs, &p, &ts, 1, MPI_SIZE_T, comm);
    char* comment = new char[ts];
    MPI_Unpack(buf, bs, &p, comment, ts, MPI_CHAR, comm);
    s->comment = comment;
    delete[] comment;

    // Structure.box
    for (size_t i = 0; i < 3; ++i)
    {
        MPI_Unpack(buf, bs, &p, s->box[i].r, 3, MPI_DOUBLE, comm);
    }

    // Structure.invbox
    for (size_t i = 0; i < 3; ++i)
    {
        MPI_Unpack(buf, bs, &p, s->invbox[i].r, 3, MPI_DOUBLE, comm);
    }

    // Structure.numAtomsPerElement
    ts = 0;
    MPI_Unpack(buf, bs, &p, &ts, 1, MPI_SIZE_T, comm);
    if (ts > 0)
    {
        s->numAtomsPerElement.clear();
        s->numAtomsPerElement.resize(ts, 0);
        MPI_Unpack(buf, bs, &p, &(s->numAtomsPerElement.front()), ts, MPI_SIZE_T, comm);
    }

    // Structure.atoms
    ts = 0;
    MPI_Unpack(buf, bs, &p, &ts, 1, MPI_SIZE_T, comm);
    if (ts > 0)
    {
        s->atoms.clear();
        s->atoms.resize(ts);
        for (vector<Atom>::iterator it = s->atoms.begin();
             it != s->atoms.end(); ++it)
        {
            // Atom
            MPI_Unpack(buf, bs, &p, &(it->hasNeighborList               ), 1, MPI_CHAR  , comm);
            MPI_Unpack(buf, bs, &p, &(it->hasSymmetryFunctions          ), 1, MPI_CHAR  , comm);
            MPI_Unpack(buf, bs, &p, &(it->hasSymmetryFunctionDerivatives), 1, MPI_CHAR  , comm);
            MPI_Unpack(buf, bs, &p, &(it->index                         ), 1, MPI_SIZE_T, comm);
            MPI_Unpack(buf, bs, &p, &(it->indexStructure                ), 1, MPI_SIZE_T, comm);
            MPI_Unpack(buf, bs, &p, &(it->tag                           ), 1, MPI_SIZE_T, comm);
            MPI_Unpack(buf, bs, &p, &(it->element                       ), 1, MPI_SIZE_T, comm);
            MPI_Unpack(buf, bs, &p, &(it->numNeighbors                  ), 1, MPI_SIZE_T, comm);
            MPI_Unpack(buf, bs, &p, &(it->numNeighborsUnique            ), 1, MPI_SIZE_T, comm);
            MPI_Unpack(buf, bs, &p, &(it->numSymmetryFunctions          ), 1, MPI_SIZE_T, comm);
            MPI_Unpack(buf, bs, &p, &(it->energy                        ), 1, MPI_DOUBLE, comm);
            MPI_Unpack(buf, bs, &p, &(it->charge                        ), 1, MPI_DOUBLE, comm);
            MPI_Unpack(buf, bs, &p, &(it->r.r                           ), 3, MPI_DOUBLE, comm);
            MPI_Unpack(buf, bs, &p, &(it->f.r                           ), 3, MPI_DOUBLE, comm);
            MPI_Unpack(buf, bs, &p, &(it->fRef.r                        ), 3, MPI_DOUBLE, comm);

            // Atom.neighborsUnique
            size_t ts2 = 0;
            MPI_Unpack(buf, bs, &p, &ts2, 1, MPI_SIZE_T, comm);
            if (ts2 > 0)
            {
                it->neighborsUnique.clear();
                it->neighborsUnique.resize(ts2, 0);
                MPI_Unpack(buf, bs, &p, &(it->neighborsUnique.front()), ts2, MPI_SIZE_T, comm);
            }

            // Atom.numNeighborsPerElement
            ts2 = 0;
            MPI_Unpack(buf, bs, &p, &ts2, 1, MPI_SIZE_T, comm);
            if (ts2 > 0)
            {
                it->numNeighborsPerElement.clear();
                it->numNeighborsPerElement.resize(ts2, 0);
                MPI_Unpack(buf, bs, &p, &(it->numNeighborsPerElement.front()), ts2, MPI_SIZE_T, comm);
            }

            // Atom.G
            ts2 = 0;
            MPI_Unpack(buf, bs, &p, &ts2, 1, MPI_SIZE_T, comm);
            if (ts2 > 0)
            {
                it->G.clear();
                it->G.resize(ts2, 0.0);
                MPI_Unpack(buf, bs, &p, &(it->G.front()), ts2, MPI_DOUBLE, comm);
            }

            // Atom.dEdG
            ts2 = 0;
            MPI_Unpack(buf, bs, &p, &ts2, 1, MPI_SIZE_T, comm);
            if (ts2 > 0)
            {
                it->dEdG.clear();
                it->dEdG.resize(ts2, 0.0);
                MPI_Unpack(buf, bs, &p, &(it->dEdG.front()), ts2, MPI_DOUBLE, comm);
            }

            // Atom.dGdxia
            ts2 = 0;
            MPI_Unpack(buf, bs, &p, &ts2, 1, MPI_SIZE_T, comm);
            if (ts2 > 0)
            {
                it->dGdxia.clear();
                it->dGdxia.resize(ts2, 0.0);
                MPI_Unpack(buf, bs, &p, &(it->dGdxia.front()), ts2, MPI_DOUBLE, comm);
            }

            // Atom.dGdr
            ts2 = 0;
            MPI_Unpack(buf, bs, &p, &ts2, 1, MPI_SIZE_T, comm);
            if (ts2 > 0)
            {
                it->dGdr.clear();
                it->dGdr.resize(ts2);
                for (vector<Vec3D>::iterator it2 = it->dGdr.begin();
                     it2 != it->dGdr.end(); ++it2)
                {
                    MPI_Unpack(buf, bs, &p, it2->r, 3, MPI_DOUBLE, comm);
                }
            }

            // Atom.neighbors
            ts2 = 0;
            MPI_Unpack(buf, bs, &p, &ts2, 1, MPI_SIZE_T, comm);
            if (ts2 > 0)
            {
                it->neighbors.clear();
                it->neighbors.resize(ts2);
                for (vector<Atom::Neighbor>::iterator it2 =
                     it->neighbors.begin(); it2 != it->neighbors.end(); ++it2)
                {
                    // Neighbor
                    MPI_Unpack(buf, bs, &p, &(it2->index      ), 1, MPI_SIZE_T, comm);
                    MPI_Unpack(buf, bs, &p, &(it2->tag        ), 1, MPI_SIZE_T, comm);
                    MPI_Unpack(buf, bs, &p, &(it2->element    ), 1, MPI_SIZE_T, comm);
                    MPI_Unpack(buf, bs, &p, &(it2->d          ), 1, MPI_DOUBLE, comm);
                    MPI_Unpack(buf, bs, &p, &(it2->fc         ), 1, MPI_DOUBLE, comm);
                    MPI_Unpack(buf, bs, &p, &(it2->dfc        ), 1, MPI_DOUBLE, comm);
                    MPI_Unpack(buf, bs, &p, &(it2->rc         ), 1, MPI_DOUBLE, comm);
                    MPI_Unpack(buf, bs, &p, &(it2->cutoffAlpha), 1, MPI_DOUBLE, comm);
                    MPI_Unpack(buf, bs, &p, &(it2->cutoffType ), 1, MPI_INT   , comm);
                    MPI_Unpack(buf, bs, &p,   it2->dr.r        , 3, MPI_DOUBLE, comm);

                    // Neighbor.dGdr
                    size_t ts3 = 0;
                    MPI_Unpack(buf, bs, &p, &ts3, 1, MPI_SIZE_T, comm);
                    if (ts3 > 0)
                    {
                        it2->dGdr.clear();
                        it2->dGdr.resize(ts3);
                        for (vector<Vec3D>::iterator it3 = it2->dGdr.begin();
                             it3 != it2->dGdr.end(); ++it3)
                        {
                            MPI_Unpack(buf, bs, &p, it3->r, 3, MPI_DOUBLE, comm);
                        }
                    }
                }
            }
        }
    }

    delete[] buf;

    return bs;
}

size_t Dataset::getNumStructures(ifstream& dataFile)
{
    size_t count = 0;
    string line;
    vector<string> splitLine;

    while (getline(dataFile, line))
    {
        splitLine = split(reduce(line));
        if (splitLine.at(0) == "begin") count++;
    }

    return count;
}

int Dataset::distributeStructures(bool          randomize,
                                  bool          excludeRank0,
                                  string const& fileName)
{
    log << "\n";
    log << "*** STRUCTURE DISTRIBUTION **************"
           "**************************************\n";
    log << "\n";

    ifstream dataFile;
    vector<size_t> numStructuresPerProc;

    if (excludeRank0)
    {
        log << "No structures will be distributed to rank 0.\n";
        if (numProcs == 1)
        {
            throw runtime_error("ERROR: Can not distribute structures, "
                                "at least 2 MPI tasks are required.\n");
        }
    }
    size_t numReceivers = numProcs;
    if (excludeRank0) numReceivers--;

    if (myRank == 0)
    {
        log << strpr("Reading configurations from data file: %s.\n",
                     fileName.c_str());
        dataFile.open(fileName.c_str());
        numStructures = getNumStructures(dataFile);
        log << strpr("Total number of structures: %zu\n", numStructures);
        dataFile.clear();
        dataFile.seekg(0);
    }
    MPI_Bcast(&numStructures, 1, MPI_SIZE_T, 0, comm);
    if (numStructures < numReceivers)
    {
        throw runtime_error("ERROR: More receiving processors than "
                            "structures.\n");
    }

    numStructuresPerProc.resize(numProcs, 0);
    if (myRank == 0)
    {
        size_t quotient = numStructures / numReceivers;
        size_t remainder = numStructures % numReceivers;
        for (size_t i = 0; i < (size_t) numProcs; i++)
        {
            if (i != 0 || (!excludeRank0))
            {
                numStructuresPerProc.at(i) = quotient;
                if (remainder > 0 && i > 0 && i <= remainder)
                {
                    numStructuresPerProc.at(i)++;
                }
            }
        }
        if (remainder == 0)
        {
            log << strpr("Number of structures per processor: %d\n", quotient);
        }
        else
        {
            log << strpr("Number of structures per"
                         " processor: %d (%d) or %d (%d)\n",
                         quotient,
                         numReceivers - remainder,
                         quotient + 1,
                         remainder);
        }
    }
    MPI_Bcast(&(numStructuresPerProc.front()), numProcs, MPI_SIZE_T, 0, comm);

    int bytesTransferred = 0;
    size_t numMyStructures = numStructuresPerProc.at(myRank);
    if (myRank == 0)
    {
        size_t countStructures = 0;
        vector<size_t> countStructuresPerProc;

        countStructuresPerProc.resize(numProcs, 0);
        
        if (randomize)
        {
            while (countStructures < numStructures)
            {
                int proc = gsl_rng_uniform_int(rng, numProcs);
                if (countStructuresPerProc.at(proc)
                    < numStructuresPerProc.at(proc))
                {
                    if (proc == 0)
                    {
                        structures.push_back(Structure());
                        structures.back().setElementMap(elementMap);
                        structures.back().index = countStructures;
                        structures.back().readFromFile(dataFile);
                        removeEnergyOffset(structures.back());
                    }
                    else
                    {
                        Structure tmpStructure;
                        tmpStructure.setElementMap(elementMap);
                        tmpStructure.index = countStructures;
                        tmpStructure.readFromFile(dataFile);
                        removeEnergyOffset(tmpStructure);
                        bytesTransferred += sendStructure(tmpStructure, proc);
                    }
                    countStructuresPerProc.at(proc)++;
                    countStructures++;
                }
            }
        }
        else
        {
            for (int proc = 0; proc < numProcs; ++proc)
            {
                for (size_t i = 0; i < numStructuresPerProc.at(proc); ++i)
                {
                    if (proc == 0)
                    {
                        structures.push_back(Structure());
                        structures.back().setElementMap(elementMap);
                        structures.back().index = countStructures;
                        structures.back().readFromFile(dataFile);
                        removeEnergyOffset(structures.back());
                    }
                    else
                    {
                        Structure tmpStructure;
                        tmpStructure.setElementMap(elementMap);
                        tmpStructure.index = countStructures;
                        tmpStructure.readFromFile(dataFile);
                        removeEnergyOffset(tmpStructure);
                        bytesTransferred += sendStructure(tmpStructure, proc);
                    }
                    countStructuresPerProc.at(proc)++;
                    countStructures++;
                }
            }
        }
        dataFile.close();
    }
    else
    {
        for (size_t i = 0; i < numMyStructures; i++)
        {
            structures.push_back(Structure());
            structures.back().setElementMap(elementMap);
            bytesTransferred += recvStructure(&(structures.back()), 0);
        }
    }

    log << strpr("Distributed %zu structures,"
                 " %d bytes (%.2f MiB) transferred.\n",
                 numStructures,
                 bytesTransferred,
                 bytesTransferred / 1024. / 1024.);
    log << strpr("Number of local structures: %zu\n", structures.size());
    log << "*****************************************"
           "**************************************\n";

    return bytesTransferred;
}

void Dataset::toNormalizedUnits()
{
    for (vector<Structure>::iterator it = structures.begin();
         it != structures.end(); ++it)
    {
        it->toNormalizedUnits(meanEnergy, convEnergy, convLength);
    }
    
    return;
}

void Dataset::toPhysicalUnits()
{
    for (vector<Structure>::iterator it = structures.begin();
         it != structures.end(); ++it)
    {
        it->toPhysicalUnits(meanEnergy, convEnergy, convLength);
    }
    
    return;
}

void Dataset::collectSymmetryFunctionStatistics()
{
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        for (size_t i = 0; i < it->numSymmetryFunctions(); ++i)
        {
            SymmetryFunctionStatistics::
            Container& c = it->statistics.data.at(i);
            MPI_Allreduce(MPI_IN_PLACE, &(c.count), 1, MPI_SIZE_T, MPI_SUM, comm);
            MPI_Allreduce(MPI_IN_PLACE, &(c.min  ), 1, MPI_DOUBLE, MPI_MIN, comm);
            MPI_Allreduce(MPI_IN_PLACE, &(c.max  ), 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(MPI_IN_PLACE, &(c.sum  ), 1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(MPI_IN_PLACE, &(c.sum2 ), 1, MPI_DOUBLE, MPI_SUM, comm);
        }
    }

    return;
}

void Dataset::writeSymmetryFunctionScaling(string const& fileName)
{
    log << "\n";
    log << "*** SYMMETRY FUNCTION SCALING ***********"
           "**************************************\n";
    log << "\n";

    if (myRank == 0)
    {
        log << strpr("Writing symmetry function scaling file: %s.\n",
                     fileName.c_str());
        ofstream sFile;
        sFile.open(fileName.c_str());

        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back("Symmetry function scaling data.");
        colSize.push_back(10);
        colName.push_back("e_index");
        colInfo.push_back("Element index.");
        colSize.push_back(10);
        colName.push_back("sf_index");
        colInfo.push_back("Symmetry function index.");
        colSize.push_back(24);
        colName.push_back("sf_min");
        colInfo.push_back("Symmetry function minimum.");
        colSize.push_back(24);
        colName.push_back("sf_max");
        colInfo.push_back("Symmetry function maximum.");
        colSize.push_back(24);
        colName.push_back("sf_mean");
        colInfo.push_back("Symmetry function mean.");
        colSize.push_back(24);
        colName.push_back("sf_sigma");
        colInfo.push_back("Symmetry function sigma.");
        appendLinesToFile(sFile,
                          createFileHeader(title, colSize, colName, colInfo));

        for (vector<Element>::const_iterator it = elements.begin();
             it != elements.end(); ++it)
        {
            for (size_t i = 0; i < it->numSymmetryFunctions(); ++i)
            {
                SymmetryFunctionStatistics::
                Container const& c = it->statistics.data.at(i);
                size_t n = c.count;
                sFile << strpr("%10d %10d %24.16E %24.16E %24.16E %24.16E\n",
                               it->getIndex() + 1,
                               i + 1,
                               c.min,
                               c.max,
                               c.sum / n,
                               sqrt((c.sum2 - c.sum * c.sum / n) / (n - 1)));
            }
        }
        // Finally decided to remove this legacy line...
        //sFile << strpr("%f %f\n", 0.0, 0.0);
        sFile.close();
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Dataset::writeSymmetryFunctionHistograms(size_t numBins,
                                              string fileNameFormat)
{
    log << "\n";
    log << "*** SYMMETRY FUNCTION HISTOGRAMS ********"
           "**************************************\n";
    log << "\n";

    // Initialize histograms.
    numBins--;
    vector<vector<gsl_histogram*> > histograms;
    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        histograms.push_back(vector<gsl_histogram*>());
        for (size_t i = 0; i < it->numSymmetryFunctions(); ++i)
        {
            double l = safeFind(it->statistics.data, i).min;
            double h = safeFind(it->statistics.data, i).max;
            // Add an extra bin at the end to cover complete range.
            h += (h - l) / numBins;
            histograms.back().push_back(gsl_histogram_alloc(numBins + 1));
            gsl_histogram_set_ranges_uniform(histograms.back().back(), l, h);
        }
    }

    // Increment histograms with symmetry function values.
    for (vector<Structure>::const_iterator it = structures.begin();
         it != structures.end(); ++it)
    {
        for (vector<Atom>::const_iterator it2 = it->atoms.begin();
             it2 != it->atoms.end(); ++it2)
        {
            size_t e = it2->element;
            vector<gsl_histogram*>& h = histograms.at(e);
            for (size_t s = 0; s < it2->G.size(); ++s)
            {
                gsl_histogram_increment(h.at(s), it2->G.at(s));
            }
        }
    }

    // Collect histograms from all processes.
    for (vector<vector<gsl_histogram*> >::const_iterator it =
         histograms.begin(); it != histograms.end(); ++it)
    {
        for (vector<gsl_histogram*>::const_iterator it2 = it->begin();
             it2 != it->end(); ++it2)
        {
            MPI_Allreduce(MPI_IN_PLACE, (*it2)->bin, numBins + 1, MPI_DOUBLE, MPI_SUM, comm);
        }
    }

    // Write histogram to file.
    if (myRank == 0)
    {
        log << strpr("Writing histograms with %zu bins.\n", numBins + 1);
        for (size_t e = 0; e < elements.size(); ++e)
        {
            for (size_t s = 0; s < elements.at(e).numSymmetryFunctions(); ++s)
            {
                FILE* fp = 0;
                string fileName = strpr(fileNameFormat.c_str(),
                                        elementMap.atomicNumber(e),
                                        s + 1);
                fp = fopen(fileName.c_str(), "w");
                if (fp == 0)
                {
                    throw runtime_error(strpr("ERROR: Could not open file:"
                                              " %s.\n", fileName.c_str()));
                }
                gsl_histogram*& h = histograms.at(e).at(s);
                vector<string> info = elements.at(e).infoSymmetryFunction(s);
                for (vector<string>::const_iterator it = info.begin();
                     it != info.end(); ++it)
                {
                    fprintf(fp, "#SFINFO %s\n", it->c_str());
                }
                SymmetryFunctionStatistics::
                Container const& c = elements.at(e).statistics.data.at(s);
                size_t n = c.count;
                fprintf(fp, "#SFINFO min         %15.8E\n", c.min);
                fprintf(fp, "#SFINFO max         %15.8E\n", c.max);
                fprintf(fp, "#SFINFO mean        %15.8E\n", c.sum / n);
                fprintf(fp, "#SFINFO sigma       %15.8E\n",
                        sqrt(1.0 / (n - 1) * (c.sum2 - c.sum * c.sum / n)));

                // File header.
                vector<string> title;
                vector<string> colName;
                vector<string> colInfo;
                vector<size_t> colSize;
                title.push_back("Symmetry function histogram.");
                colSize.push_back(16);
                colName.push_back("symfunc_l");
                colInfo.push_back("Symmetry function value, left bin limit.");
                colSize.push_back(16);
                colName.push_back("symfunc_r");
                colInfo.push_back("Symmetry function value, right bin limit.");
                colSize.push_back(16);
                colName.push_back("hist");
                colInfo.push_back("Histogram count.");
                appendLinesToFile(fp,
                                  createFileHeader(title,
                                                   colSize,
                                                   colName,
                                                   colInfo));

                gsl_histogram_fprintf(fp, h, "%16.8E", "%16.8E");
                fflush(fp);
                fclose(fp);
                fp = 0;
            }
        }
    }

    for (vector<vector<gsl_histogram*> >::const_iterator it =
         histograms.begin(); it != histograms.end(); ++it)
    {
        for (vector<gsl_histogram*>::const_iterator it2 = it->begin();
             it2 != it->end(); ++it2)
        {
            gsl_histogram_free(*it2);
        }
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Dataset::writeSymmetryFunctionFile(string fileName)
{
    log << "\n";
    log << "*** SYMMETRY FUNCTION FILE **************"
           "**************************************\n";
    log << "\n";

    // Create empty file.
    log << strpr("Writing symmetry functions to file: %s\n", fileName.c_str());
    if (myRank == 0)
    {
        ofstream file;
        file.open(fileName.c_str());
        file.close();
    }
    MPI_Barrier(comm);

    // Prepare structure iterator.
    vector<Structure>::const_iterator it = structures.begin();
    // Loop over all structures (on each proc the local structures are stored
    // with increasing index).
    for (size_t i = 0; i < numStructures; ++i)
    {
        // If this proc holds structure with matching index,
        // open file and write symmetry functions.
        if (i == it->index)
        {
            ofstream file;
            file.open(fileName.c_str(), ios_base::app);
            file << strpr("%6zu\n", it->numAtoms);
            // Loop over atoms.
            for (vector<Atom>::const_iterator it2 = it->atoms.begin();
                 it2 != it->atoms.end(); ++it2)
            {
                // Loop over symmetry functions.
                file << strpr("%3zu ", elementMap.atomicNumber(it2->element));
                for (vector<double>::const_iterator it3 = it2->G.begin();
                     it3 != it2->G.end(); ++it3)
                {
                    file << strpr(" %14.10f", *it3);
                }
                file << '\n';
            }
            // There is no charge NN, so first and last entry is zero.
            double energy = 0.0;
            if (normalize) energy = physicalEnergy(*it);
            else energy = it->energyRef;
            energy += getEnergyOffset(*it);
            energy /= it->numAtoms;
            file << strpr(" %19.10f %19.10f %19.10f %19.10f\n",
                          0.0, energy, energy, 0.0);
            file.flush();
            file.close();
            // Iterate to next structure.
            ++it;
        }
        MPI_Barrier(comm);
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

size_t Dataset::writeNeighborHistogram(string const& fileName)
{
    log << "\n";
    log << "*** NEIGHBOR HISTOGRAMS *****************"
           "**************************************\n";
    log << "\n";

    // Determine maximum number of neighbors.
    size_t numAtoms = 0;
    size_t minNeighbors = numeric_limits<size_t>::max();
    size_t maxNeighbors = 0;
    double meanNeighbors = 0.0;
    for (vector<Structure>::const_iterator it = structures.begin();
         it != structures.end(); ++it)
    {
        numAtoms += it->numAtoms;
        for (vector<Atom>::const_iterator it2 = it->atoms.begin();
             it2 != it->atoms.end(); ++it2)
        {
            size_t const n = it2->numNeighbors;
            minNeighbors = min(minNeighbors, n);
            maxNeighbors = max(maxNeighbors, n);
            meanNeighbors += n;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &numAtoms     , 1, MPI_SIZE_T, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &minNeighbors , 1, MPI_SIZE_T, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maxNeighbors , 1, MPI_SIZE_T, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &meanNeighbors, 1, MPI_DOUBLE, MPI_SUM, comm);
    log << strpr("Minimum number of neighbors: %d\n", minNeighbors);
    log << strpr("Mean    number of neighbors: %.1f\n",
                 meanNeighbors / numAtoms);
    log << strpr("Maximum number of neighbors: %d\n", maxNeighbors);

    // Calculate neighbor histogram.
    gsl_histogram* histNeighbors = NULL;
    histNeighbors = gsl_histogram_alloc(maxNeighbors + 1);
    gsl_histogram_set_ranges_uniform(histNeighbors,
                                     -0.5,
                                     maxNeighbors + 0.5);
    for (vector<Structure>::const_iterator it = structures.begin();
         it != structures.end(); ++it)
    {
        for (vector<Atom>::const_iterator it2 = it->atoms.begin();
             it2 != it->atoms.end(); ++it2)
        {
            gsl_histogram_increment(histNeighbors, it2->numNeighbors);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, histNeighbors->bin, maxNeighbors + 1, MPI_DOUBLE, MPI_SUM, comm);

    // Write histogram to file.
    if (myRank == 0)
    {
        log << strpr("Neighbor histogram file: %s.\n", fileName.c_str());
        FILE* fp = 0;
        fp = fopen(fileName.c_str(), "w");
        if (fp == 0)
        {
            throw runtime_error(strpr("ERROR: Could not open file: %s.\n",
                                      fileName.c_str()));
        }

        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back("Symmetry function histogram.");
        colSize.push_back(9);
        colName.push_back("neigh_l");
        colInfo.push_back("Number of neighbors, left bin limit.");
        colSize.push_back(9);
        colName.push_back("neigh_r");
        colInfo.push_back("Number of neighbors, right bin limit.");
        colSize.push_back(16);
        colName.push_back("hist");
        colInfo.push_back("Histogram count.");
        appendLinesToFile(fp,
                          createFileHeader(title, colSize, colName, colInfo));

        gsl_histogram_fprintf(fp, histNeighbors, "%9.1f", "%16.8E");
        fflush(fp);
        fclose(fp);
        fp = 0;
    }

    gsl_histogram_free(histNeighbors);

    log << "*****************************************"
           "**************************************\n";

    return maxNeighbors;
}

void Dataset::averageRmse(double& rmse, size_t& count) const
{
    MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_SIZE_T, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &rmse , 1, MPI_DOUBLE, MPI_SUM, comm);
    rmse = sqrt(rmse / count);

    return;
}

void Dataset::combineFiles(string filePrefix) const
{
    ofstream combinedFile(filePrefix.c_str(), ios::binary);
    if (!combinedFile.is_open())
    {
        throw runtime_error(strpr("ERROR: Could not open file: %s.\n",
                                  filePrefix.c_str()));
    }
    for (int i = 0; i < numProcs; ++i)
    {
        string procFileName = strpr("%s.%04d", filePrefix.c_str(), i);
        ifstream procFile(procFileName.c_str(), ios::binary);
        if (!procFile.is_open())
        {
            throw runtime_error(strpr("ERROR: Could not open file: %s.\n",
                                      procFileName.c_str()));
        }
        // If file is empty, do not get rdbuf, otherwise combined file will be
        // closed!
        if (procFile.peek() != ifstream::traits_type::eof())
        {
            combinedFile << procFile.rdbuf();
        }
        procFile.close();
        remove(procFileName.c_str());
    }
    combinedFile.close();

    return;
}

