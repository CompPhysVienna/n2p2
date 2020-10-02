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

#include "./ElementCabana.h"
#include "typesCabana.h"

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

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
 * to use the Kokkos and Cabana libraries. Most setup functions are
 * overridden; some are replaced as needed to work within device kernels.
 */
template <class t_device>
class ModeCabana : public Mode
{

 public:
    using Mode::Mode;

    // Kokkos settings
    using device_type = t_device;
    using exe_space = typename device_type::execution_space;
    using memory_space = typename device_type::memory_space;
    typedef typename exe_space::array_layout layout;
    using host_space = Kokkos::HostSpace;

    // Per-type Kokkos::Views
    using d_t_mass = Kokkos::View<T_V_FLOAT *, memory_space>;
    using h_t_mass = Kokkos::View<T_V_FLOAT *, layout, host_space>;
    using d_t_int = Kokkos::View<T_INT *, memory_space>;
    using h_t_int = Kokkos::View<T_INT *, layout, host_space>;

    // SymmetryFunctionTypes Kokkos::Views
    using d_t_SF = Kokkos::View<T_FLOAT * * [15], memory_space>;
    using t_SF = Kokkos::View<T_FLOAT * * [15], layout, host_space>;
    using d_t_SFscaling = Kokkos::View<T_FLOAT * * [8], memory_space>;
    using t_SFscaling = Kokkos::View<T_FLOAT * * [8], layout, host_space>;
    using d_t_SFGmemberlist = Kokkos::View<T_INT ***, memory_space>;
    using t_SFGmemberlist = Kokkos::View<T_INT ***, layout, host_space>;

    // NN Kokkos::Views
    using d_t_bias = Kokkos::View<T_FLOAT ***, memory_space>;
    using t_bias = Kokkos::View<T_FLOAT ***, layout, host_space>;
    using d_t_weights = Kokkos::View<T_FLOAT ****, memory_space>;
    using t_weights = Kokkos::View<T_FLOAT ****, layout, host_space>;
    using d_t_NN = Kokkos::View<T_FLOAT ***, memory_space>;

    /** Set up the element map.
     *
     * Uses keyword `elements`. This function should follow immediately after
     * settings are loaded via loadSettingsFile().
     */
    void setupElementMap() override;
    /** Set up all Element instances.
     *
     * Uses keywords `number_of_elements` and `atom_energy`. This function
     * should follow immediately after setupElementMap().
     */
    void setupElements() override;
    /** Set up all symmetry functions.
     *
     * Uses keyword `symfunction_short`. Reads all symmetry functions from
     * settings and automatically assigns them to the correct element.
     */
    void setupSymmetryFunctions() override;
    /** Set up symmetry function scaling from file.
     *
     * @param[in] fileName Scaling file name.
     *
     * Uses keywords `scale_symmetry_functions`, `center_symmetry_functions`,
     * `scale_symmetry_functions_sigma`, `scale_min_short` and
     * `scale_max_short`. Reads in scaling information and sets correct scaling
     * behavior for all symmetry functions. Call after
     * setupSymmetryFunctions().
     */
    void setupSymmetryFunctionScaling(
             std::string const &fileName = "scaling.data") override;
    /** Set up symmetry function groups.
     *
     * Does not use any keywords. Call after setupSymmetryFunctions() and
     * ensure that correct scaling behavior has already been set.
     */
    void setupSymmetryFunctionGroups() override;
    /** Set up neural networks for all elements.
     *
     * Uses keywords `global_hidden_layers_short`, `global_nodes_short`,
     * `global_activation_short`, `normalize_nodes`. Call after
     * setupSymmetryFunctions(), only then the number of input layer neurons is
     * known.
     */
    void setupNeuralNetwork() override;
    /** Set up neural network weights from files.
     *
     * @param[in] fileNameFormat Format for weights file name. The string must
     *                           contain one placeholder for the atomic number.
     *
     * Does not use any keywords. The weight files should contain one weight
     * per line, see NeuralNetwork::setConnections() for the correct order.
     */
    void setupNeuralNetworkWeights(
        std::string const &fileNameFormat = "weights.%03zu.data") override;

    /* Compute cutoff for a single atom pair.
     *
     * @param[in] cutoffType Cutoff type.
     * @param[in] fc Cutoff function value.
     * @param[in] dfc Derivative of cutoff function.
     * @param[in] r Atom separation distance.
     * @param[in] rc Cutoff radius.
     * @param[in] derivative If the cutoff derivative should be computed.
     *
     * Callable within device kernel.
     */
    KOKKOS_INLINE_FUNCTION
    void compute_cutoff(CutoffFunction::CutoffType cutoffType, double cutoffAlpha,
                        double &fc, double &dfc, double r, double rc,
                        bool derivative) const;
    /*
     * Scale one symmetry function.
     *
     * @param[in] attype Atom type.
     * @param[in] value Scaled symmetry function value.
     * @param[in] k Symmetry function index.
     * @param[in] SFscaling Kokkos host View of symmetry function scaling.
     *
     * Callable within device kernel.
     */
    KOKKOS_INLINE_FUNCTION
    double scale(int attype, double value, int k, d_t_SFscaling SFscaling) const;

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

    /** Calculate atomic neural networks for all atoms in given structure.
     *
     * @param[in] type Cabana slice of atom types.
     * @param[in] G Cabana slice of symmetry functions per atom.
     * @param[in] dEdG Cabana slice of the derivative of energy with respect
                       to symmetry functions per atom.
     * @param[in] E Cabana slice of energy per atom.
     * @param[in] N_local Number of atoms.
     *
     * The atomic energy is stored in E.
     */
    template <class t_slice_type, class t_slice_G, class t_slice_dEdG,
              class t_slice_E>
    void calculateAtomicNeuralNetworks(t_slice_type type, t_slice_G G,
                                       t_slice_dEdG dEdG, t_slice_E E,
                                       int N_local);

    /** Calculate all symmetry function groups for all atoms in given
     * structure.
     *
     * @param[in] x Cabana slice of atom positions.
     * @param[in] type Cabana slice of atom types.
     * @param[in] G Cabana slice of symmetry functions per atom.
     * @param[in] neigh_list Cabana neighbor list.
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

    using Mode::log;

    /// list of element symbols in order of periodic table
    // (duplicated from ElementMap)
    std::vector<std::string> knownElements = {
        "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg",
        "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr",
        "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
        "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
        "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm",
        "Bk", "Cf", "Es", "Fm", "Md", "No"};

    // Symmetry function Kokkos::Views
    d_t_SF d_SF;
    t_SF SF;
    d_t_SFGmemberlist d_SFGmemberlist;
    t_SFGmemberlist SFGmemberlist;
    d_t_SFscaling d_SFscaling;
    t_SFscaling SFscaling;

    // NN Kokkos::Views
    d_t_bias bias;
    d_t_weights weights;
    t_bias h_bias;
    t_weights h_weights;
    int numLayers, numHiddenLayers, maxNeurons;
    d_t_int numNeuronsPerLayer;
    h_t_int h_numNeuronsPerLayer;
    d_t_int AF;
    h_t_int h_AF;

    // Per type Kokkos::Views
    h_t_mass atomicEnergyOffset;
    h_t_int h_numSFperElem;
    d_t_int numSFperElem;
    h_t_int h_numSFGperElem;
    d_t_int numSFGperElem;
    int maxSFperElem;

    using Mode::numElements;
    using Mode::minNeighbors;
    using Mode::minCutoffRadius;
    using Mode::maxCutoffRadius;
    using Mode::cutoffAlpha;
    double meanEnergy;
    using Mode::convEnergy;
    using Mode::convLength;
    ScalingType scalingType;
    using Mode::settings;
    using Mode::cutoffType;
    std::vector<ElementCabana> elements;
    std::vector<std::string> elementStrings;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

template <class t_device>
KOKKOS_INLINE_FUNCTION void
ModeCabana<t_device>::compute_cutoff(CutoffFunction::CutoffType cutoffType,
                                     double cutoffAlpha, double &fc, double &dfc,
                                     double r, double rc, bool derivative) const
{
    double temp;
    if (cutoffType == CutoffFunction::CT_TANHU) {
        temp = tanh(1.0 - r / rc);
        fc = temp * temp * temp;
        if (derivative)
            dfc = 3.0 * temp * temp * (temp * temp - 1.0) / rc;
    }

    if (cutoffType == CutoffFunction::CT_COS) {

        double rci = rc * cutoffAlpha;
        double iw = 1.0 / (rc - rci);
        double PI = 4.0 * atan(1.0);
        if (r < rci) {
            fc = 1.0;
            dfc = 0.0;
        } else {
            temp = cos(PI * (r - rci) * iw);
            fc = 0.5 * (temp + 1.0);
            if (derivative)
                dfc = -0.5 * iw * PI * sqrt(1.0 - temp * temp);
        }
    }
}

template <class t_device>
KOKKOS_INLINE_FUNCTION double
ModeCabana<t_device>::scale(int attype, double value, int k,
                            d_t_SFscaling SFscaling_) const
{
    double scalingType = SFscaling_(attype, k, 7);
    double scalingFactor = SFscaling_(attype, k, 6);
    double Gmin = SFscaling_(attype, k, 0);
    // double Gmax = SFscaling_(attype,k,1);
    double Gmean = SFscaling_(attype, k, 2);
    // double Gsigma = SFscaling_(attype,k,3);
    double Smin = SFscaling_(attype, k, 4);
    // double Smax = SFscaling_(attype,k,5);

    if (scalingType == 0.0) {
        return value;
    } else if (scalingType == 1.0) {
        return Smin + scalingFactor * (value - Gmin);
    } else if (scalingType == 2.0) {
        return value - Gmean;
    } else if (scalingType == 3.0) {
        return Smin + scalingFactor * (value - Gmean);
    } else if (scalingType == 4.0) {
        return Smin + scalingFactor * (value - Gmean);
    } else {
        return 0.0;
    }
}

}

#include "ModeCabana_impl.h"

#endif
