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
#include <cstddef> // std::size_t
#include <map>     // std::map
#include <string>  // std::string
#include <vector>  // std::vector

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
        std::size_t                tag;
        /// %Element index of neighbor atom.
        std::size_t                element;
        /// Distance to neighbor atom.
        double                     d;
        /// Distance vector to neighbor atom.
        Vec3D                      dr;
#ifndef NNP_NO_SF_CACHE
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
    std::size_t              tag;
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
    /// Atomic charge determined by neural network.
    double                   charge;
    /// Atomic reference charge.
    double                   chargeRef;
    /// Cartesian coordinates
    Vec3D                    r;
    /// Force vector calculated by neural network.
    Vec3D                    f;
    /// Reference force vector from data set.
    Vec3D                    fRef;
    /// List of unique neighbor indices (don't count multiple PBC images).
    std::vector<std::size_t> neighborsUnique;
    /// Number of neighbors per element.
    std::vector<std::size_t> numNeighborsPerElement;
    /// Number of neighbor atom symmetry function derivatives per element.
    std::vector<std::size_t> numSymmetryFunctionDerivatives;
#ifndef NNP_NO_SF_CACHE
    /// Cache size for each element.
    std::vector<std::size_t> cacheSizePerElement;
#endif
    /// Symmetry function values
    std::vector<double>      G;
    /// Derivative of atomic energy with respect to symmetry functions.
    std::vector<double>      dEdG;
    /// Derivative of atomic charge with respect to symmetry functions.
    std::vector<double>      dQdG;
#ifdef NNP_FULL_SFD_MEMORY
    /// Derivative of symmetry functions with respect to one specific atom
    /// coordinate.
    std::vector<double>      dGdxia;
#endif
    /// Derivative of symmetry functions with respect to this atom's
    /// coordinates.
    std::vector<Vec3D>       dGdr;
    /// Neighbor array (maximum number defined in macros.h.
    std::vector<Neighbor>    neighbors;

    /** Atom constructor, initialize to zero.
     */
    Atom();
#ifdef NNP_FULL_SFD_MEMORY
    /** Collect derivative of symmetry functions with repect to one atom's
     * coordinate.
     *
     * @param[in] indexAtom The index @f$i@f$ of the atom requested.
     * @param[in] indexComponent The component @f$\alpha@f$ of the atom
     *                           requested.
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
                                           std::size_t indexComponent);
#endif
    /** Switch to normalized length and energy units.
     *
     * @param[in] convEnergy Multiplicative energy unit conversion factor.
     * @param[in] convLength Multiplicative length unit conversion factor.
     */
    void                     toNormalizedUnits(double convEnergy,
                                               double convLength);
    /** Switch to physical length and energy units.
     *
     * @param[in] convEnergy Multiplicative energy unit conversion factor.
     * @param[in] convLength Multiplicative length unit conversion factor.
     */
    void                     toPhysicalUnits(double convEnergy,
                                             double convLength);
    /** Allocate vectors related to symmetry functions (#G, #dEdG).
     *
     * @param[in] all If `true` allocate also vectors corresponding to
     *                derivatives of symmetry functions (#dEdG, #dGdr, #dGdxia
     *                and Neighbor::dGdr, neighbors must be present). If
     *                `false` allocate only #G.
     *
     * Warning: #numSymmetryFunctions and #numSymmetryFunctionDerivatives need
     * to be set first (the latter only in case of argument all == true.
     */
    void                     allocate(bool all);
    /** Free vectors related to symmetry functions, opposite of #allocate().
     *
     * @param[in] all If `true` free all vectors (#G, #dEdG, #dGdr, #dGdxia
     *                and Neighbor::dGdr) otherwise free only #dEdG, #dGdr,
     *                #dGdxia and Neighbor::dGdr.
     */
    void                     free(bool all);
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
    std::size_t              getNumNeighbors(double cutoffRadius) const;
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
