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

#include "Ewald.h"
#include <stdexcept> // std::runtime_error
#include <iostream>  // std::cout

using namespace nnp;
using namespace std;

// Ratio of computing times for one real space and k space iteration.
constexpr double TrOverTk = 3.676;

EwaldSetup::EwaldSetup() : truncMethod(EWALDTruncMethod::JACKSON_CATLOW),
                           precision  (1.0E-6                          ),
                           maxCharge  (0.0                             ),
                           maxQsigma  (0.0                             ),
                           volume     (0.0                             ),
                           eta        (0.0                             ),
                           s          (0.0                             ),
                           rCut       (0.0                             ),
                           kCut       (0.0                             )
{
}

void EwaldSetup::readFromArgs(vector<string> const& args, double const maxWidth)
{
    bool argConsistent = false;
    if (truncMethod ==
        EWALDTruncMethod::JACKSON_CATLOW)
        argConsistent = (args.size() == 1);
    else
        argConsistent = (args.size() == 2);

    if (!argConsistent)
        throw runtime_error("ERROR: Number of arguments for ewald_prec is "
                            "not consistent with "
                            "ewald_truncation_error_method.");
    precision = atof(args.at(0).c_str());
    if (truncMethod == EWALDTruncMethod::KOLAFA_PERRAM)
        maxCharge = atof(args.at(1).c_str());
    maxQsigma = maxWidth;
}

void EwaldSetup::calculateParameters(double const newVolume, size_t const numAtoms)
{
    calculateEta(newVolume, numAtoms);
    if (truncMethod == EWALDTruncMethod::KOLAFA_PERRAM)
        calculateS(numAtoms);
    calculateRCut();
    calculateKCut();
}


double EwaldSetup::calculateEta(double const newVolume, size_t const numAtoms)
{
    if (truncMethod == EWALDTruncMethod::JACKSON_CATLOW)
    {
        volume = newVolume;
        eta = 1.0 / sqrt(2.0 * M_PI);
        // Regular Ewald eta.
        //if (numAtoms != 0) eta *= pow(volume * volume / numAtoms, 1.0 / 6.0);
            // Matrix version eta.
        //else eta *= pow(volume, 1.0 / 3.0);
        eta *= pow(volume, 1.0 / 3.0);
        //TODO: in RuNNer they take eta = max(eta, maxval(sigma))
    }
    else
    {
        if (newVolume == volume) return eta;
        else volume = newVolume;

        eta = pow(1 / TrOverTk * pow(volume, 2) / pow(2 * M_PI, 3),
                  1.0 / 6.0);

        eta = max(eta, maxQsigma);
    }

    return eta;
}

double EwaldSetup::calculateRCut()
{
    if (eta == 0.0) throw runtime_error("ERROR: Width of screening charges "
                                        "can't be 0.");

    if (truncMethod == EWALDTruncMethod::JACKSON_CATLOW)
        rCut = sqrt(-2.0 * log(precision)) * eta;
    else
        rCut = sqrt(2) * eta * s;
    return rCut;
}

double EwaldSetup::calculateKCut()
{
    if (eta == 0.0) throw runtime_error("ERROR: Width of screening charges "
                                        "can't be 0.");
    if (truncMethod == EWALDTruncMethod::JACKSON_CATLOW)
        kCut = sqrt(-2.0 * log(precision)) / eta;
    else
        kCut = sqrt(2) * s / eta;
    return kCut;
}

double EwaldSetup::calculateS(size_t const numAtoms)
{
    double y = precision * sqrt(eta / sqrt(2));
    y /= 2 * sqrt(numAtoms * 1.0 / volume) * pow(maxCharge, 2);

    s = 1.0;
    double step = 1.0;
    while (step > 0.001)
    {
        step = 2 * s / (4 * pow(s,2) + 1);
        step *= 1 -  sqrt(s) * y / exp(-pow(s,2));
        s += step;
    }
    //cout << "intermediate value s: " << s << endl;
    return s;
}