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

#ifndef EWALDSETUP_H
#define EWALDSETUP_H

//#include "Vec3D.h"
#include "IEwaldTrunc.h"
#include "Log.h"
#include <string>
#include <vector>
#include <memory>


namespace nnp {
enum class EWALDTruncMethod {
    /// Method 0: Used by RuNNer (DOI: 10.1080/08927028808080944).
    JACKSON_CATLOW,
    /// Method 1: Optimized in n2p2 (DOI: 10.1080/08927029208049126).
    KOLAFA_PERRAM
};

/// Setup data for Ewald summation.
class EwaldSetup {
public:
    EwaldParameters params;
    /// Default constructor.
    EwaldSetup();
    void setTruncMethod(EWALDTruncMethod const m);
    EWALDTruncMethod getTruncMethod() const {return truncMethod;};
    double getMaxCharge() const {return GlobSett.maxCharge;};
    double getPrecision() const {return GlobSett.precision;};

    /** Setter for maximum width of charges.
     *
     * @param maxWidth Maximum width of gaussian charges.
     */
    void setMaxQSigma(double const maxWidth);
    /** Setup parameters from argument vector.
     *
     * @param[in] args Vector containing arguments of input file.
     */
    void readFromArgs(std::vector<std::string> const& args);

    /** Convert cutoff parameters to normalized units.
     *
     * @param convEnergy Conversion factor for energy.
     * @param convLength Conversion factor for length.
     */
    void toNormalizedUnits(double const convEnergy,
                           double const convLength);
    /** Compute eta, rCut and kCut.
    *
    * @param[in] newVolume Volume of the real space cell.
    * @param[in] newNumAtoms Number of atoms in system.
    */
    void calculateParameters(double const newVolume, size_t const newNumAtoms);
    /** Use after Ewald summation!
     *
     * @return
     */
    void logEwaldCutoffs(Log& log, double const lengthConversion) const;

private:
    /// Method for determining real and k space cutoffs.
    EWALDTruncMethod truncMethod;
    EwaldGlobalSettings GlobSett;

    std::unique_ptr<IEwaldTrunc> truncImpl;

    bool publishedNewCutoffs() const;
    bool isEstimateReliable() const;
};

inline void EwaldSetup::setMaxQSigma(double const maxWidth)
{
    GlobSett.maxQSigma = maxWidth;
}

inline void EwaldSetup::setTruncMethod(EWALDTruncMethod const m)
{
    truncMethod = m;
}

}
#endif //EWALDSETUP_H
