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

#ifndef DATASET_H
#define DATASET_H

#include <mpi.h>
#include "Mode.h"
#include "Structure.h"
#include <cstddef> // std::size_t
#include <fstream> // std::ifstream
#include <map>     // std::map
#include <string>  // std::string
#include <vector>  // std::vector
#include <gsl/gsl_rng.h>

namespace nnp
{

///Collect and process large data sets.
class Dataset : public Mode
{
public:
    /** Constructor, initialize members.
     */
    Dataset();
    /** Destructor.
     */
    ~Dataset();
    /** Initialize MPI with MPI_COMM_WORLD.
     */
    void        setupMPI();
    /** Initialize MPI with given communicator.
     *
     * @param[in] communicator Provided communicator which should be used.
     */
    void        setupMPI(MPI_Comm* communicator);
    /** Initialize random number generator.
     *
     * CAUTION: MPI communicator required, call #setupMPI() before!
     */
    void        setupRandomNumberGenerator();
    /** Get number of structures in data file.
     *
     * @param[inout] dataFile Data file name.
     *
     * @return Number of structures.
     */
    std::size_t getNumStructures(std::ifstream& dataFile);
    /** Calculate buffer size required to communicate structure via MPI
     *
     * @param[in] structure Input structure.
     *
     * @return Buffer size in bytes.
     */
    int         calculateBufferSize(Structure const& structure) const;
    /** Send one structure to destination process.
     *
     * @param[in] structure Source structure.
     * @param[in] dest MPI rank of destination process.
     *
     * @return Number of bytes sent.
     */
    int         sendStructure(Structure const& structure, int dest) const;
    /** Receive one structure from source process.
     *
     * @param[in] structure Source structure.
     * @param[in] src MPI rank of source process.
     *
     * @return Number of bytes received.
     */
    int         recvStructure(Structure* structure, int src);
    /** Read data file and distribute structures among processors.
     *
     * @param[in] randomize If `true` randomly distribute structures, otherwise
     *                      they are distributed in order.
     * @param[in] excludeRank0 If `true` no structures are distributed to MPI
     *                         process with rank 0.
     * @param[in] fileName Data file name, e.g. "input.data".
     *
     * @return Number of bytes transferred.
     */
    int         distributeStructures(bool               randomize,
                                     bool               excludeRank0 = false,
                                     std::string const& fileName
                                         = "input.data");
    /** Switch all structures to normalized units.
     */
    void        toNormalizedUnits();
    /** Switch all structures to physical units.
     */
    void        toPhysicalUnits();
    /** Collect symmetry function statistics from all processors.
     */
    void        collectSymmetryFunctionStatistics();
    /** Write symmetry function scaling values to file.
     *
     * @param[in] fileName Scaling data file name (e.g. "scaling.data").
     */
    void        writeSymmetryFunctionScaling(std::string const& fileName
                                                 = "scaling.data");
    /** Calculate and write symmetry function histograms.
     *
     * @param[in] numBins Number of histogram bins used.
     * @param[in] fileNameFormat Format for histogram file names, must include
     *                           placeholders for element and symmetry function
     *                           number.
     */
    void        writeSymmetryFunctionHistograms(std::size_t numBins,
                                                std::string fileNameFormat
                                                    = "sf.%03zu.%04zu.histo");
    /** Write symmetry function legacy file ("function.data").
     *
     * @param[in] fileName File name for symmetry function file.
     */
    void        writeSymmetryFunctionFile(std::string fileName
                                              = "function.data");
    /** Calculate and write neighbor histogram.
     *
     * @param[in] fileName Name of histogram file.
     *
     * @return Maximum number of neighbors.
     */
    std::size_t writeNeighborHistogram(std::string const& fileName
                                           = "neighbors.histo");
    /** Sort all neighbor lists according to element and distance.
     */
    void        sortNeighborLists();
    /** Write neighbor list file.
     *
     * @param[in] fileName Name for neighbor list file.
     */
    void        writeNeighborLists(std::string const& fileName
                                       = "neighbor-list.data");
    /** Write atomic environment file.
     *
     * @param[in] neighCutoff Maximum number of neighbor to consider (for each
     *                        element combination).
     * @param[in] derivatives If true, write separate files for derivates.
     * @param[in] fileNamePrefix Prefix for atomic environment files.
     *
     * This file is used for symmetry function clustering analysis.
     */
    void        writeAtomicEnvironmentFile(
                                        std::vector<std::vector<
                                        std::size_t> >           neighCutoff,
                                        bool                     derivatives,
                                        std::string const &      fileNamePrefix
                                             = "atomic-env");
    /** Collect error metrics of a property over all MPI procs.
     *
     * @param[in] property One of "energy", "force" or "charge".
     * @param[in,out] error Metric sums of this proc (in), global metric (out).
     * @param[in,out] count Count for this proc (in), global count (out).
     */
    void        collectError(std::string const&             property,
                             std::map<std::string, double>& error,
                             std::size_t&                   count) const;
    /** Combine individual MPI proc files to one.
     *
     * @param[in] filePrefix File prefix without the ".0001" suffix.
     *
     * CAUTION: Make sure files are completely written and closed.
     */
    void        combineFiles(std::string filePrefix) const;

    /// All structures in this dataset.
    std::vector<Structure> structures;

protected:
    /// My process ID.
    int         myRank;
    /// Total number of MPI processors.
    int         numProcs;
    /// Total number of structures in dataset.
    std::size_t numStructures;
    /// My processor name.
    std::string myName;
    /// Global MPI communicator.
    MPI_Comm    comm;
    /// GSL random number generator (different seed for each MPI process).
    gsl_rng*    rng;
    /// Global GSL random number generator (equal seed for each MPI process).
    gsl_rng*    rngGlobal;
};

}

#endif
