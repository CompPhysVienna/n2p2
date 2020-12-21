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

#ifndef STRUCTURE_H
#define STRUCTURE_H

#include "Atom.h"
#include "ElementMap.h"
#include "Vec3D.h"
#include <cstddef> // std::size_t
#include <fstream> // std::ofstream
#include <map>     // std::map
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/// Storage for one atomic configuration.
struct Structure
{
    /** Enumerates different sample types (e.g. training or test set).
     */
    enum SampleType
    {
        /** Sample type not assigned yet.
         */
        ST_UNKNOWN,
        /** Structure is part of the training set.
         */
        ST_TRAINING,
        /** Structure is part of validation set (currently unused).
         */
        ST_VALIDATION,
        /** Structure is part of the test set.
         */
        ST_TEST
    };

    /// Copy of element map provided as constructor argument.
    ElementMap               elementMap;
    /// If periodic boundary conditions apply.
    bool                     isPeriodic;
    /// If the simulation box is triclinic.
    bool                     isTriclinic;
    /// If the neighbor list has been calculated.
    bool                     hasNeighborList;
    /// If symmetry function values are saved for each atom.
    bool                     hasSymmetryFunctions;
    /// If symmetry function derivatives are saved for each atom.
    bool                     hasSymmetryFunctionDerivatives;
    /// Index number of this structure.
    std::size_t              index;
    /// Total number of atoms present in this structure.
    std::size_t              numAtoms;
    /// Global number of elements (all structures).
    std::size_t              numElements;
    /// Number of elements present in this structure.
    std::size_t              numElementsPresent;
    /// Number of PBC images necessary in each direction.
    int                      pbc[3];
    /// Potential energy determined by neural network.
    double                   energy;
    /// Reference potential energy.
    double                   energyRef;
    /// Charge determined by neural network potential.
    double                   charge;
    /// Reference charge.
    double                   chargeRef;
    /// Simulation box volume.
    double                   volume;
    /// Sample type (training or test set).
    SampleType               sampleType;
    /// Structure comment.
    std::string              comment;
    /// Simulation box vectors.
    Vec3D                    box[3];
    /// Inverse simulation box vectors.
    Vec3D                    invbox[3];
    /// Number of atoms of each element in this structure.
    std::vector<std::size_t> numAtomsPerElement;
    /// Vector of all atoms in this structure.
    std::vector<Atom>        atoms;

    /** Constructor, initializes to zero.
     */
    Structure();
    /** Set element map of structure.
     *
     * @param[in] elementMap Reference to a map containing all possible
     *                       (symbol, index)-pairs (see ElementMap).
     */
    void                     setElementMap(ElementMap const& elementMap);
    /** Add a single atom to structure.
     *
     * @param[in] atom Atom to insert.
     * @param[in] element Element string of new atom.
     *
     * @note Be sure to set the element map properly before adding atoms. This
     * function will only keep the atom's coordinates, energy, charge, tag and
     * forces, all other members will be cleared or reset (in particular, the
     * neighbor list and symmetry function data will be deleted).
     */
    void                     addAtom(Atom const&        atom,
                                     std::string const& element);
    /** Read configuration from file.
     *
     * @param[in] fileName Input file name.
     *
     * Reads the first configuration found in the input file.
     */
    void                     readFromFile(std::string const fileName
                                                            = "input.data");
    /** Read configuration from file.
     *
     * @param[in] file Input file stream (already opened).
     *
     * Expects that a file with configurations is open, first keyword on first
     * line should be `begin`. Reads until keyword is `end`.
     */
    void                     readFromFile(std::ifstream& file);
    /** Read configuration from lines.
     *
     * @param[in] lines One configuration in form of a vector of strings.
     *
     * Read the configuration from a vector of strings.
     */
    void                     readFromLines(std::vector<
                                           std::string> const& lines);
    /** Calculate neighbor list for all atoms.
     *
     * @param[in] cutoffRadius Atoms are neighbors if there distance is smaller
     *                         than the cutoff radius.
     */
    void                     calculateNeighborList(double cutoffRadius);
    /** Calculate required PBC copies.
     *
     * @param[in] cutoffRadius Cutoff radius for neighbor list.
     *
     * Called by #calculateNeighborList().
     */
    void                     calculatePbcCopies(double cutoffRadius);
    /** Calculate inverse box.
     *
     * Simulation box looks like this:
     *
     * @f[
     * h =
     * \begin{pmatrix}
     * \phantom{a_x} & \phantom{b_x} & \phantom{c_x} \\
     * \vec{\mathbf{a}} & \vec{\mathbf{b}} & \vec{\mathbf{c}} \\
     * \phantom{a_z} & \phantom{b_z} & \phantom{c_z} \\
     * \end{pmatrix} =
     * \begin{pmatrix}
     * a_x & b_x & c_x \\
     * a_y & b_y & c_y \\
     * a_z & b_z & c_z \\
     * \end{pmatrix},
     * @f]
     *
     * where @f$\vec{\mathbf{a}} = @f$ `box[0]`, @f$\vec{\mathbf{b}} = @f$
     * `box[1]` and @f$\vec{\mathbf{c}} = @f$ `box[2]`. Thus, indices are
     * column first, row second:
     *
     * @f[
     * h =
     * \begin{pmatrix}
     * \texttt{box[0][0]} & \texttt{box[1][0]} & \texttt{box[2][0]} \\
     * \texttt{box[0][1]} & \texttt{box[1][1]} & \texttt{box[2][1]} \\
     * \texttt{box[0][2]} & \texttt{box[1][2]} & \texttt{box[2][2]} \\
     * \end{pmatrix}.
     * @f]
     *
     * The inverse box matrix (same scheme as above but with `invbox`) can be
     * used to calculate fractional coordinates:
     *
     * @f[
     * \begin{pmatrix}
     * f_0 \\ f_1 \\ f_2
     * \end{pmatrix} = h^{-1} \; \vec{\mathbf{r}}.
     * @f]
     */
    void                     calculateInverseBox();
    /** Calculate volume from box vectors.
     */
    void                     calculateVolume();
    /** Translate atom back into box if outside.
     *
     * @param[in,out] atom Atom to be remapped.
     */
    void                     remap(Atom& atom);
    /** Normalize structure, shift energy and change energy and length unit.
     *
     * @param[in] meanEnergy Mean energy per atom (in old units).
     * @param[in] convEnergy Multiplicative energy unit conversion factor.
     * @param[in] convLength Multiplicative length unit conversion factor.
     */
    void                     toNormalizedUnits(double meanEnergy,
                                               double convEnergy,
                                               double convLength);
    /** Switch to physical units, shift energy and change energy and length unit.
     *
     * @param[in] meanEnergy Mean energy per atom (in old units).
     * @param[in] convEnergy Multiplicative energy unit conversion factor.
     * @param[in] convLength Multiplicative length unit conversion factor.
     */
    void                     toPhysicalUnits(double meanEnergy,
                                             double convEnergy,
                                             double convLength);
    /** Find maximum number of neighbors.
     *
     * @return Maximum numbor of neighbors of all atoms in this structure.
     */
    std::size_t              getMaxNumNeighbors() const;
    /** Free symmetry function memory for all atoms, see free() in Atom class.
     *
     * @param[in] all See description in Atom.
     */
    void                     freeAtoms(bool all);
    /** Reset everything but #elementMap.
     */
    void                     reset();
    /** Clear neighbor list of all atoms.
     */
    void                     clearNeighborList();
    /** Update property error metrics with this structure.
     *
     * @param[in] property One of "energy", "force" or "charge".
     * @param[in,out] error Input error metric map to be updated.
     * @param[in,out] count Input counter to be updated.
     *
     * The "energy" error metric map stores temporary sums for the following
     * metrics:
     *
     * key "RMSEpa": RMSE of energy per atom
     * key "RMSE"  : RMSE of energy
     * key "MAEpa" : MAE  of energy per atom
     * key "MAE"   : MAE  of energy
     *
     * The "force" error metric map stores temporary sums for the following
     * metrics:
     *
     * key "RMSE"  : RMSE of forces
     * key "MAE"   : MAE  of forces
     *
     * The "charge" error metric map stores temporary sums for the following
     * metrics:
     *
     * key "RMSE"  : RMSE of charges
     * key "MAE"   : MAE  of charges
     */
    void                     updateError(
                                   std::string const&             property,
                                   std::map<std::string, double>& error,
                                   std::size_t&                   count) const;
    /** Get reference and NN energy.
     *
     * @return String with #index, #energyRef and #energy values.
     */
    std::string              getEnergyLine() const;
    /** Get reference and NN forces for all atoms.
     *
     * @return Vector of strings with force comparison.
     */
    std::vector<std::string> getForcesLines() const;
    /** Get reference and NN charges for all atoms.
     *
     * @return Vector of strings with charge comparison.
     */
    std::vector<std::string> getChargesLines() const;
    /** Write configuration to file.
     *
     * @param[in,out] fileName Ouptut file name.
     * @param[in] ref If true, write reference energy and forces, if false,
     *                write NNP results instead.
     * @param[in] append If true, append to existing file.
     */
    void                     writeToFile(
                                     std::string const fileName ="output.data",
                                     bool const        ref = true,
                                     bool const        append = false) const;
    /** Write configuration to file.
     *
     * @param[in,out] file Ouptut file.
     * @param[in] ref If true, write reference energy and forces, if false,
     *                write NNP results instead.
     */
    void                     writeToFile(
                                       std::ofstream* const& file,
                                       bool const            ref = true) const;
    /** Write configuration to xyz file.
     *
     * @param[in,out] file xyz output file.
     */
    void                     writeToFileXyz(std::ofstream* const& file) const;
    /** Write configuration to POSCAR file.
     *
     * @param[in,out] file POSCAR output file.
     *
     * @warning Elements in POTCAR file must be ordered according to
     *          periodic table.
     */
    void                     writeToFilePoscar(
                                             std::ofstream* const& file) const;
    /** Write configuration to POSCAR file.
     *
     * @param[in,out] file POSCAR output file.
     * @param[in,out] elements User-defined order of elements, e.g. "Zn O Cu".
     */
    void                     writeToFilePoscar(
                                         std::ofstream* const& file,
                                         std::string const     elements) const;
    /** Get structure information as a vector of strings.
     *
     * @return Lines with structure information.
     */
    std::vector<std::string> info() const;
};

}

#endif
