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

#ifndef INTERFACELAMMPS_H
#define INTERFACELAMMPS_H

#include "Mode.h"
#include "Structure.h"
#include <map>     // std::map
#include <cstddef> // std::size_t
#include <cstdint> // int64_t
#include <string>  // std::string

namespace nnp
{

class InterfaceLammps : public Mode
{
public:
    InterfaceLammps();

    /** Initialize the LAMMPS interface.
     *
     * @param[in] directory Directory containing NNP data files (weights,
     *                      scaling, settings).
     * @param[in] emap Element mapping from LAMMPS to n2p2.
     * @param[in] showew If detailed extrapolation warnings for all atoms are
     *                   shown.
     * @param[in] resetew If extrapolation warnings counter is reset every
     *                    timestep.
     * @param[in] showewsum Show number of warnings every this many timesteps.
     * @param[in] maxew Abort simulation if more than this many warnings are
     *                  encountered.
     * @param[in] cflength Length unit conversion factor.
     * @param[in] cfenergy Energy unit conversion factor.
     * @param[in] lammpsCutoff Cutoff radius from LAMMPS (via pair_coeff).
     * @param[in] lammpsNtypes Number of atom types in LAMMPS.
     * @param[in] myRank MPI process rank (passed on to structure index).
     */
    void   initialize(char const* const& directory,
                      char const* const& emap,
                      bool               showew,
                      bool               resetew,
                      int                showewsum,
                      int                maxew,
                      double             cflength,
                      double             cfenergy,
                      double             lammpsCutoff,
                      int                lammpsNtypes,
                      int                myRank);
    /** Specify whether n2p2 knows about global structure or only local
     * structure.
     * @param[in] status true if n2p2 has global structure.
     */
    void    setGlobalStructureStatus(bool const status);
    /// Check if n2p2 knows about global structure.
    bool    getGlobalStructureStatus();
    /** (Re)set #structure to contain only local LAMMPS atoms.
     *
     * @param[in] numAtomsLocal Number of local atoms.
     * @param[in] atomType LAMMPS atom type.
     */
    void   setLocalAtoms(int              numAtomsLocal,
                         int const* const atomType);
    /** Set absolute atom positions from LAMMPS (nnp/develop only).
     *
     * @param[in] atomPos Atom coordinate array in LAMMPS units.
     */
    void   setLocalAtomPositions(double const* const* const atomPos);
    /** Set atom tags (int version, -DLAMMPS_SMALLBIG).
     *
     * @param[in] atomTag LAMMPS atom tag.
     */
    void   setLocalTags(int const* const atomTag);
    /** Set atom tags (int64_t version, -DLAMMPS_BIGBIG).
     *
     * @param[in] atomTag LAMMPS atom tag.
     */
    void   setLocalTags(int64_t const* const atomTag);
    /** Set box vectors of structure stored in LAMMPS (nnp/develop only).
     *
     * @param[in] boxlo Array containing coordinates of origin xlo, ylo, zlo.
     * @param[in] boxhi Array containing coordinates xhi, yhi, zhi.
     * @param[in] xy Tilt factor for box vector b.
     * @param[in] xz First tilt factor for box vector c.
     * @param[in] yz Second tilt factor for box vector c.
     */
    void   setBoxVectors(double const* boxlo,
                         double const* boxhi,
                         double const  xy,
                         double const  xz,
                         double const  yz);
    /** Allocate neighbor lists.
     *
     * @param[in] numneigh Array containing number of neighbors for each local atom.
     */
    void   allocateNeighborlists(int const* const numneigh);
    /** Add one neighbor to atom (int64_t version, -DLAMMPS_BIGBIG).
     *
     * @param[in] i Local atom index.
     * @param[in] j Neighbor atom index.
     * @param[in] tag Neighbor atom tag.
     * @param[in] type Neighbor atom type.
     * @param[in] dx Neighbor atom distance in x direction.
     * @param[in] dy Neighbor atom distance in y direction.
     * @param[in] dz Neighbor atom distance in z direction.
     * @param[in] d2 Square of neighbor atom distance.
     *
     * If -DLAMMPS_SMALLBIG implicit conversion is applied for tag.
     */
    void   addNeighbor(int     i,
                       int     j,
                       int64_t tag,
                       int     type,
                       double  dx,
                       double  dy,
                       double  dz,
                       double  d2);
    /** Sorts neighbor list and creates cutoff map if necessary. If structure is
     * periodic, this function needs to be called after setBoxVectors!
     */
    void   finalizeNeighborList();
    /** Calculate symmetry functions, atomic neural networks and sum of local
     * energy contributions.
     */
    void   process();
    /** Return sum of local energy contributions.
     *
     * @return Sum of local energy contributions.
     */
    double getEnergy() const;
    /** Return energy contribution of one atom.
     *
     * @param[in] index Atom index.
     *
     * @return energy contribution of atom with given index.
     *
     * @attention These atomic contributions are not physical!
     */
    double getAtomicEnergy(int index) const;
    /** Calculate forces and add to LAMMPS atomic force arrays.
     *
     * @param[in,out] atomF LAMMPS force array for local and ghost atoms.
     */
    void   getForces(double* const* const& atomF) const;
    /** Transfer charges (in units of e) to LAMMPS atomic charge vector. Call
     *  after getAtomicEnergy().
     *
     * @param[in,out] atomQ LAMMPS charge vector.
     */
    void   getCharges(double* const& atomQ) const;
    /** Check if this interface is correctly initialized.
     *
     * @return `True` if initialized, `False` otherwise.
     */
    bool   isInitialized() const;
    /** Get largest cutoff of symmetry functions.
     *
     * @return Largest cutoff of all symmetry functions.
     */
    double getMaxCutoffRadius() const;
    /** Get largest cutoff including structure specific cutoff and screening
     *                  cutoff
     *
     * @return Largest cutoff of all symmetry functions and structure specific
     *                  cutoff and screening cutoff.
     */
    double getMaxCutoffRadiusOverall();
    /** Calculate buffer size for extrapolation warning communication.
     *
     * @return Buffer size.
     */
    long   getEWBufferSize() const;
    /** Fill provided buffer with extrapolation warning entries.
     *
     * @param[in,out] buf Communication buffer to fill.
     * @param[in] bs Buffer size.
     */
    void   fillEWBuffer(char* const& buf, int bs) const;
    /** Extract given buffer to symmetry function statistics class.
     *
     * @param[in] buf Buffer with extrapolation warnings data.
     * @param[in] bs Buffer size.
     */
    void   extractEWBuffer(char const* const& buf, int bs);
    /** Write extrapolation warnings to log.
     */
    void   writeExtrapolationWarnings();
    /** Clear extrapolation warnings storage.
     */
    void   clearExtrapolationWarnings();
    /** Write current structure to file in units used in training data.
     *
     * @param fileName File name of the output structure file.
     * @param append true if structure should be appended to existing file.
     */
    void   writeToFile(std::string const fileName,
                       bool const        append);
    /** Add a Vec3D vector to a 3D array in place.
     *
     * @param[in,out] arr Array which is edited in place.
     * @param[in] v Vector which is added to arr.
     */
    void   add3DVecToArray(double *const & arr, Vec3D const& v) const;

protected:
    /// Process rank.
    int                        myRank;
    /// Initialization state.
    bool                       initialized;
    /// Whether n2p2 knows about the global structure or only a local part.
    bool                       hasGlobalStructure;
    /// Corresponds to LAMMPS `showew` keyword.
    bool                       showew;
    /// Corresponds to LAMMPS `resetew` keyword.
    bool                       resetew;
    /// Corresponds to LAMMPS `showewsum` keyword.
    int                        showewsum;
    /// Corresponds to LAMMPS `maxew` keyword.
    int                        maxew;
    /// Corresponds to LAMMPS `cflength` keyword.
    double                     cflength;
    /// Corresponds to LAMMPS `cfenergy` keyword.
    double                     cfenergy;
    /// Corresponds to LAMMPS `map` keyword.
    std::string                emap;
    /// Map from LAMMPS index to n2p2 atom index.
    std::vector<size_t>        indexMap;
    /// True if atoms of this LAMMPS type will be ignored.
    std::map<int, bool>        ignoreType;
    /// Map from LAMMPS type to n2p2 element index.
    std::map<int, std::size_t> mapTypeToElement;
    /// Map from n2p2 element index to LAMMPS type.
    std::map<std::size_t, int> mapElementToType;
    /// Structure containing local atoms.
    Structure                  structure;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool InterfaceLammps::isInitialized() const
{
    return initialized;
}

}

#endif
