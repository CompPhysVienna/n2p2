// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
// Copyright (C) 2020 Saaketh Desai and Sam Reeve
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

#ifndef MODE_CABANA_H
#define MODE_CABANA_H

#include "ModeKokkos.h"

#include <Kokkos_Core.hpp>
#include <Cabana_Core.hpp>

#include "CutoffFunction.h"
#include "Log.h"
#include "Mode.h"
#include "Settings.h"

#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/** Derived Cabana main NNP class.
 *
 * The main n2p2 functions for computing energies and forces are replaced
 * to use the Kokkos and Cabana libraries. Only the main compute kernels
 * differ from the Kokkos version.
 */
template <class t_device>
class ModeCabana : public ModeKokkos<t_device>
{

 public:
    using ModeKokkos<t_device>::ModeKokkos;
    using exe_space = typename ModeKokkos<t_device>::exe_space;

    // Symmetry function Kokkos::Views
    using ModeKokkos<t_device>::d_SF;
    using ModeKokkos<t_device>::d_SFGmemberlist;
    using ModeKokkos<t_device>::d_SFscaling;

    // Per type Kokkos::Views
    using ModeKokkos<t_device>::numSFGperElem;

    using ModeKokkos<t_device>::cutoffAlpha;
    using ModeKokkos<t_device>::convEnergy;
    using ModeKokkos<t_device>::convLength;
    using ModeKokkos<t_device>::maxSFperElem;
    using ModeKokkos<t_device>::cutoffType;

    /** Calculate forces for all atoms in given structure.
     *
     * @param[in] x Cabana slice of atom positions.
     * @param[in] f Cabana slice of atom forces.
     * @param[in] type Cabana slice of atom types.
     * @param[in] dEdG Cabana slice of the derivative of energy with respect
                       to symmetry functions per atom.
     * @param[in] N_local Number of atoms.
     * @param[in] neigh_op Cabana tag for neighbor parallelism.
     * @param[in] angle_op Cabana tag for angular parallelism.
     *
     * Combine intermediate results from symmetry function and neural network
     * computation to atomic forces. Results are stored in f.
     */
    template <class t_slice_x, class t_slice_f, class t_slice_type,
              class t_slice_dEdG, class t_neigh_list, class t_neigh_parallel,
              class t_angle_parallel>
    void calculateForces(t_slice_x x, t_slice_f f, t_slice_type type,
                         t_slice_dEdG dEdG, t_neigh_list neigh_list, int N_local,
                         t_neigh_parallel neigh_op, t_angle_parallel angle_op);

    /** Calculate all symmetry function groups for all atoms in given
     * structure.
     *
     * @param[in] x Cabana slice of atom positions.
     * @param[in] type Cabana slice of atom types.
     * @param[in] G Cabana slice of symmetry functions per atom.
     * @param[in] neigh_list neighbor list.
     * @param[in] N_local Number of atoms.
     * @param[in] neigh_op Cabana tag for neighbor parallelism.
     * @param[in] angle_op Cabana tag for angular parallelism.
     *
     * Note there is no calculateSymmetryFunctions() within this derived class.
     * Results are stored in G.
     */
    template <class t_slice_x, class t_slice_type, class t_slice_G,
              class t_neigh_list, class t_neigh_parallel, class t_angle_parallel>
    void calculateSymmetryFunctionGroups(t_slice_x x, t_slice_type type,
                                         t_slice_G G, t_neigh_list neigh_list,
                                         int N_local, t_neigh_parallel neigh_op,
                                         t_angle_parallel angle_op);
};

}

#include "ModeCabana_impl.h"

#endif
