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
    void   initialize(char* const& directory,
                      char* const& emap,
                      bool         showew,
                      bool         resetew,
                      int          showewsum,
                      int          maxew,
                      double       cflength,
                      double       cfenergy,
                      double       lammpsCutoff,
                      int          lammpsNtypes,
                      int          myRank);
    /** (Re)set #structure to contain only local LAMMPS atoms.
     *
     * @param[in] numAtomsLocal Number of local atoms.
     * @param[in] atomTag LAMMPS atom tag.
     * @param[in] atomType LAMMPS atom type.
     */
    void   setLocalAtoms(int              numAtomsLocal,
                         int const* const atomTag,
                         int const* const atomType);
    /** Add one neighbor to atom.
     *
     * @param[in] i Local atom index.
     * @param[in] j Neighbor atom index.
     * @param[in] tag Neighbor atom tag.
     * @param[in] type Neighbor atom type.
     * @param[in] dx Neighbor atom distance in x direction.
     * @param[in] dy Neighbor atom distance in y direction.
     * @param[in] dz Neighbor atom distance in z direction.
     * @param[in] d2 Square of neighbor atom distance.
     */
    void   addNeighbor(int    i,
                       int    j,
                       int    tag,
                       int    type,
                       double dx,
                       double dy,
                       double dz,
                       double d2);
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
    /** Check if this interface is correctly initialized.
     *
     * @return `True` if initialized, `False` otherwise.
     */
    bool   isInitialized() const;
    /** Get largest cutoff.
     *
     * @return Largest cutoff of all symmetry functions.
     */
    double getMaxCutoffRadius() const;
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

protected:
    /// Process rank.
    int                        myRank;
    /// Initialization state.
    bool                       initialized;
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
