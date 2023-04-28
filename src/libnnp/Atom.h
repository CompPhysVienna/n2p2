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

#ifndef ATOM_H
#define ATOM_H

#include "Vec3D.h"
#include <cstddef>          // std::size_t
#include <cstdint>          // int64_t
#include <map>              // std::map
#include <unordered_map>    // std::unordered_map
#include <string>           // std::string
#include <vector>           // std::vector

namespace nnp
{

/// Storage for a single atom.
struct Atom
{
    /// Struct to store information on neighbor atoms.
    struct Neighbor
    {
        /// Index of neighbor atom.
        std::size_t                index;
        /// Tag of neighbor atom.
        int64_t                    tag;
        /// %Element index of neighbor atom.
        std::size_t                element;
        /// Distance to neighbor atom.
        double                     d;
        /// Distance vector to neighbor atom.
        Vec3D                      dr;
#ifndef N2P2_NO_SF_CACHE
        /// Symmetry function cache (e.g. for cutoffs, compact functions).
        std::vector<double>        cache;
#endif
        /** Derivatives of symmetry functions with respect to neighbor
         * coordinates.
         *
         * May be empty, only filled when needed. Contains
         * @f$ \frac{\partial G_i}{\partial \alpha_j} @f$, where @f$G_i@f$ is
         * a symmetry function of atom @f$i@f$ and @f$\alpha_j@f$ is a
         * coordinate @f$ \alpha = x,y,z @f$ of neighbor atom @f$ j @f$,
         * necessary for force calculation.
         */
        std::vector<Vec3D>         dGdr;

        /** Neighbor constructor, initialize to zero.
         */
        Neighbor();
        /** Overload == operator.
         */
        bool                     operator==(Neighbor const& rhs) const;
        /** Overload != operator.
         */
        bool                     operator!=(Neighbor const& rhs) const;
        /** Overload < operator.
         */
        bool                     operator<(Neighbor const& rhs) const;
        /** Overload > operator.
         */
        bool                     operator>(Neighbor const& rhs) const;
        /** Overload <= operator.
         */
        bool                     operator<=(Neighbor const& rhs) const;
        /** Overload >= operator.
         */
        bool                     operator>=(Neighbor const& rhs) const;
        /** Get atom information as a vector of strings.
         *
         * @return Lines with atom information.
         */
        std::vector<std::string> info() const;
    };

    /// If the neighbor list has been calculated for this atom.
    bool                     hasNeighborList;
    /// If the neighbor list is sorted by distance.
    bool                     NeighborListIsSorted;
    /// If symmetry function values are saved for this atom.
    bool                     hasSymmetryFunctions;
    /// If symmetry function derivatives are saved for this atom.
    bool                     hasSymmetryFunctionDerivatives;
    /// If an additional charge neuron in the short-range NN is present.
    bool                     useChargeNeuron;
    /// Index number of this atom.
    std::size_t              index;
    /// Index number of structure this atom belongs to.
    std::size_t              indexStructure;
    /// Tag number of this atom.
    int64_t                  tag;
    /// %Element index of this atom.
    std::size_t              element;
    /// Total number of neighbors.
    std::size_t              numNeighbors;
    /// Number of unique neighbor indices (don't count multiple PBC images).
    std::size_t              numNeighborsUnique;
    /// Number of symmetry functions used to describe the atom environment.
    std::size_t              numSymmetryFunctions;
    /// Atomic energy determined by neural network.
    double                   energy;
    /// Derivative of electrostatic energy with respect to this atom's charge.
    double                   dEelecdQ;
    /// Atomic electronegativity determined by neural network.
    double                   chi;
    /// Atomic charge determined by neural network.
    double                   charge;
    /// Atomic reference charge.
    double                   chargeRef;
    /// Cartesian coordinates
    Vec3D                    r;
    /// Force vector calculated by neural network.
    Vec3D                    f;
    /// Force vector resulting from electrostatics.
    Vec3D                    fElec;
    /// Reference force vector from data set.
    Vec3D                    fRef;
    /// Partial derivative of electrostatic energy with respect to this atom's
    /// coordinates.
    Vec3D                    pEelecpr;
    /// List of unique neighbor indices (don't count multiple PBC images).
    std::vector<std::size_t> neighborsUnique;
    /// Number of neighbors per element.
    std::vector<std::size_t> numNeighborsPerElement;
    /// Number of neighbor atom symmetry function derivatives per element.
    std::vector<std::size_t> numSymmetryFunctionDerivatives;
#ifndef N2P2_NO_SF_CACHE
    /// Cache size for each element.
    std::vector<std::size_t> cacheSizePerElement;
#endif
    /// Symmetry function values
    std::vector<double>      G;
    /// Derivative of atomic energy with respect to symmetry functions. Also
    /// contains dEdQ in the last element if HDNNP-4G is used.
    std::vector<double>      dEdG;
    /// Derivative of atomic charge with respect to symmetry functions.
    std::vector<double>      dQdG;
    /// Derivative of electronegativity with respect to symmetry functions.
    std::vector<double>      dChidG;
#ifdef N2P2_FULL_SFD_MEMORY
    /// Derivative of symmetry functions with respect to one specific atom
    /// coordinate.
    std::vector<double>      dGdxia;
#endif
    /// Derivative of symmetry functions with respect to this atom's
    /// coordinates.
    std::vector<Vec3D>       dGdr;
    /// Derivative of charges with respect to this atom's coordinates.
    std::vector<Vec3D>       dQdr;
    /// If dQdr has been calculated for respective components.
    //bool                    hasdQdr[3];
    /// Derivative of A-matrix with respect to this atom's coordinates
    /// contracted with the charges.
    std::vector<Vec3D>       dAdrQ;
    /// Neighbor array (maximum number defined in macros.h.
    std::vector<Neighbor>    neighbors;
    /// Map stores number of neighbors needed for the corresponding cut-off.
    std::unordered_map<double, size_t> neighborCutoffs;

    /** Atom constructor, initialize to zero.
     */
    Atom();
#ifdef N2P2_FULL_SFD_MEMORY
    /** Collect derivative of symmetry functions with repect to one atom's
     * coordinate.
     *
     * @param[in] indexAtom The index @f$i@f$ of the atom requested.
     * @param[in] indexComponent The component @f$\alpha@f$ of the atom
     *                           requested.
     * @param[in] maxCutoffRadius Maximum symmetry function cutoff.
     *
     * This calculates an array of derivatives
     * @f[
     *   \left(\frac{\partial G_1}{\partial x_{i,\alpha}}, \ldots,
     *   \frac{\partial G_n}{\partial x_{i,\alpha}}\right),
     *
     * @f]
     * where @f$\{G_j\}_{j=1,\ldots,n}@f$ are the symmetry functions for this
     * atom and @f$x_{i,\alpha}@f$ is the @f$\alpha@f$-component of the
     * position of atom @f$i@f$. The result is stored in #dGdxia.
     */
    void                     collectDGdxia(std::size_t indexAtom,
                                           std::size_t indexComponent,
                                           double maxCutoffRadius);
#endif
    /** Switch to normalized length, energy and charge units.
     *
     * @param[in] convEnergy Multiplicative energy unit conversion factor.
     * @param[in] convLength Multiplicative length unit conversion factor.
     * @param[in] convCharge Multiplicative charge unit conversion factor.
     */
    void                     toNormalizedUnits(double convEnergy,
                                               double convLength,
                                               double convCharge);
    /** Switch to physical length, energy and charge units.
     *
     * @param[in] convEnergy Multiplicative energy unit conversion factor.
     * @param[in] convLength Multiplicative length unit conversion factor.
     * @param[in] convCharge Multiplicative charge unit conversion factor.
     */
    void                     toPhysicalUnits(double convEnergy,
                                             double convLength,
                                             double convCharge);
    /** Allocate vectors related to symmetry functions (#G, #dEdG).
     *
     * @param[in] all If `true` allocate also vectors corresponding to
     *                derivatives of symmetry functions (#dEdG, #dGdr, #dGdxia
     *                and Neighbor::dGdr, neighbors must be present). If
     *                `false` allocate only #G.
     * @param[in] maxCutoffRadius Maximum cutoff radius of symmetry functions.
     *
     * Warning: #numSymmetryFunctions and #numSymmetryFunctionDerivatives need
     * to be set first (the latter only in case of argument all == true.
     */
    void                     allocate(bool         all,
                                      double const maxCutoffRadius = 0.0);
    /** Free vectors related to symmetry functions, opposite of #allocate().
     *
     * @param[in] all If `true` free all vectors (#G, #dEdG, #dGdr, #dGdxia
     *                and Neighbor::dGdr) otherwise free only #dEdG, #dGdr,
     *                #dGdxia and Neighbor::dGdr.
     * @param[in] maxCutoffRadius Maximum cutoff radius of symmetry functions.
     */
    void                     free(bool all, double const maxCutoffRadius = 0.0);
    /** Clear neighbor list.
     *
     * @note Also clears symmetry function data.
     */
    void                     clearNeighborList();
    /** Clear neighbor list and change number of elements.
     *
     * @param[in] numElements Number of elements present.
     *
     * @note Also clears symmetry function data. The number of elements is
     * necessary to allocate the #numNeighborsPerElement vector. Use the
     * overloaded version clearNeighborList() if the number of elements stays
     * the same.
     */
    void                     clearNeighborList(std::size_t const numElements);
    /** Calculate number of neighbors for a given cutoff radius.
     *
     * @param[in] cutoffRadius Desired cutoff radius.
     *
     * This function assumes that the neighbor list has already been calculated
     * for a large cutoff radius and now the number of neighbor atoms for a
     * smaller cutoff is requested.
     */
    std::size_t              calculateNumNeighbors(double const cutoffRadius) const;
    /** Return needed number of neighbors for a given cutoff radius from
     *                  neighborCutoffs map. If it isn't setup, return number
     *                  of all neighbors in list.
     * @param[in]   cutoffRadius Desired cutoff radius.
     * @return Integer with the number of neighbors corresponding to the cutoff
     *                  radius.
     */
    std::size_t              getStoredMinNumNeighbors(
                                             double const cutoffRadius) const;
    /** Return whether atom is a neighbor.
     *
     * @param[in] index Index of atom in question.
     *
     * @return `True` if atom is neighbor of this atom.
     */
    bool                     isNeighbor(std::size_t index) const;
    /** Update property error metrics with data from this atom.
     *
     * @param[in] property One of "force" or "charge".
     * @param[in,out] error Input error metric map to be updated.
     * @param[in,out] count Input counter to be updated.
     */
    void                     updateError(
                                   std::string const&             property,
                                   std::map<std::string, double>& error,
                                   std::size_t&                   count) const;
    /** Calculate force resulting from gradient of this atom's
     *  (short-ranged) energy contribution with respect to this atom's
     *  coordinate.
     *  @return calculated force vector.
     */
    Vec3D                    calculateSelfForceShort() const;
     /** Calculate force resulting from gradient of this atom's (short-ranged)
     *  energy contribution with respect to neighbor's coordinate.
     *
     * @param[in] neighbor Neighbor of which the gradient is considered.
     * @param[in] tableFull Pointer to ymmetry function table if it is used.
     * @return calculated force vector.
     */
    Vec3D                    calculatePairForceShort(
                                    Neighbor const&  neighbor,
                                    std::vector<std::vector <std::size_t> >
                                    const *const     tableFull = nullptr) const;
    /** Calculate dChi/dr of this atom's Chi with respect to the coordinates
     *  of the given atom.
     *
     * @param[in] atomIndexOfR Index of the atom corresponding to the coordinates r.
     * @param[in] maxCutoffRadius Largest cut-off of the symmetry functions.
     * @param[in] tableFull Pointer to symmetry function table if it is used.
     * @return 3D Vector containing the derivative dChi/dr.
     */
    Vec3D                    calculateDChidr(
                                    size_t const     atomIndexOfR,
                                    double const     maxCutoffRadius,
                                    std::vector<std::vector<size_t> >
                                    const *const     tableFull = nullptr) const;
    /** Get reference and NN forces for this atoms.
     *
     * @return Vector of strings with #indexStructure, #index, #fRef, #f
     * values.
     */
    std::vector<std::string> getForcesLines() const;
    /** Get reference and NN charge for this atoms.
     *
     * @return Line with #indexStructure, #index, #chargeRef, #charge values.
     */
    std::string              getChargeLine() const;
    /** Get atom information as a vector of strings.
     *
     * @return Lines with atom information.
     */
    std::vector<std::string> info() const;
};

inline bool Atom::Neighbor::operator!=(Atom::Neighbor const& rhs) const
{
    return !((*this) == rhs);
}

inline bool Atom::Neighbor::operator>(Atom::Neighbor const& rhs) const
{
    return rhs < (*this);
}

inline bool Atom::Neighbor::operator<=(Atom::Neighbor const& rhs) const
{
    return !((*this) > rhs);
}

inline bool Atom::Neighbor::operator>=(Atom::Neighbor const& rhs) const
{
    return !((*this) < rhs);
}

}

#endif
