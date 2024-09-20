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

#ifndef EWALD_H
#define EWALD_H

#include "Vec3D.h"
#include <string>
#include <vector>


namespace nnp {
enum class EWALDTruncMethod {
    /// Method 0: Used by RuNNer (DOI: 10.1080/08927028808080944).
    JACKSON_CATLOW,
    /// Method 1: Optimized in n2p2 (DOI: 10.1080/08927029208049126).
    KOLAFA_PERRAM
};

/// Setup data for Ewald summation.
struct EwaldSetup {
    /// Method for determining real and k space cutoffs.
    EWALDTruncMethod truncMethod;
    /// Precision of the Ewald summation (interpretation is dependent on
    /// truncMethod).
    double precision;
    /// Multiplicative constant
    ///         @f$ \text{fourPiEps} = 4 \pi \varepsilon_0 @f$.
    ///         Value depends on unit system (e.g. normalization).
    double fourPiEps;
    /// Maximum expected charge, needed for error estimate in method
    /// KOLAFA_PERRAM.
    double maxCharge;

    /// Maximum charge width.
    double maxQsigma;
    /// Cell volume of structure.
    double volume;
    /// Width of the gaussian screening charges.
    double eta;
    /// intermediate value that is needed for calculation of kCut and rCut.
    double s;
    /// Cutoff in real space.
    double rCut;
    /// Cutoff in reciprocal space.
    double kCut;

    /// Default constructor.
    EwaldSetup();
    /** Setup parameters from argument vector.
     *
     * @param[in] args Vector containing arguments of input file.
     * @param[in] maxQ maximum of absolute value of charges in training
     *                      set, used as default.
     */
    void readFromArgs(std::vector<std::string> const& args,
                      double const maxQ);
    /** Setter for maximum width of charges.
     *
     * @param maxWidth Maximum width of gaussian charges.
     */
    void setMaxQSigma(double const maxWidth);
    /** Convert cutoff parameters to normalized units.
     *
     * @param convEnergy Conversion factor for energy.
     * @param convLength Conversion factor for length.
     * @param convCharge Conversion factor for charge.
     */
    void toNormalizedUnits(double const convEnergy,
                           double const convLength,
                           double const convCharge);
    /** Compute eta, rCut and kCut.
    *
    * @param[in] newVolume Volume of the real space cell.
    * @param[in] numAtoms Number of atoms in system.
    */
    void calculateParameters(double const newVolume, size_t const numAtoms);
    /** Compute the width eta of the gaussian screening charges.
    *
    * @param[in] newVolume Volume of the real space cell.
    * @param[in] numAtoms Number of atoms in system. Optional, if provided
    *                     will use "regular" Ewald optimal eta, otherwise use
    *                     "matrix" version of eta.
    *
    * @return Eta.
    */
    double calculateEta(double const newVolume, size_t const numAtoms);
    /// Compute cutoff in real space for Ewald summation.
    double calculateRCut();
    /// Compute cutoff in reciprocal space for Ewald summation.
    double calculateKCut();
    /// Compute intermediate value s.
    double calculateS(size_t const numAtoms);
};

inline void EwaldSetup::setMaxQSigma(double const maxWidth)
{
    maxQsigma = maxWidth;
}

}
#endif //EWALD_H