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

#include "EwaldSetup.h"
#include "EwaldTruncKolafaFixR.h"
#include "EwaldTruncKolafaOptEta.h"
#include "EwaldTruncJackson.h"
#include "utility.h"
#include <stdexcept> // std::runtime_error
#include <memory>    // std::make_unique

using namespace nnp;
using namespace std;


EwaldSetup::EwaldSetup() : params     (                                ),
                           truncMethod(EWALDTruncMethod::JACKSON_CATLOW),
                           GlobSett   (                                ),
                           truncImpl  (nullptr                         )
{
}

void EwaldSetup::readFromArgs(vector<string> const& args)
{
    bool argConsistent = false;
    if (truncMethod ==
        EWALDTruncMethod::JACKSON_CATLOW)
        argConsistent = (args.size() == 1);
    else
        argConsistent = (args.size() >= 1 && args.size() <= 3);

    if (!argConsistent)
        throw runtime_error("ERROR: Number of arguments for ewald_prec is "
                            "not consistent with "
                            "ewald_truncation_error_method.");

    GlobSett.precision = atof(args.at(0).c_str());
    if (GlobSett.precision <= 0.0)
        throw runtime_error("ERROR: Ewald precision must be positive.");

    if (truncMethod == EWALDTruncMethod::JACKSON_CATLOW)
    {
        truncImpl.reset(new EwaldTruncJackson{});
    }
    else
    {
        if (args.size() >= 2)
            GlobSett.maxCharge = atof(args.at(1).c_str());
        if (args.size() == 2)
        {
            truncImpl.reset(new EwaldTruncKolafaOptEta{});
        }
        if (args.size() == 3)
        {
            params.rCut = atof(args.at(2).c_str());
            truncImpl.reset(new EwaldTruncKolafaFixR{});
        }
    }
}

void EwaldSetup::toNormalizedUnits(double const convEnergy,
                                   double const convLength)
{
    if (truncMethod == EWALDTruncMethod::KOLAFA_PERRAM)
    {
        // in KOLAFA_PERRAM method precision has unit of a force.
        GlobSett.precision *= convEnergy / convLength;
        GlobSett.fourPiEps /= convLength * convEnergy;
        // GlobSett.maxQSigma is already normalized
        params = params.toNormalizedUnits(convLength);
        // Other variables are already normalized.
    }
}

void EwaldSetup::calculateParameters(double const volume, size_t const numAtoms)
{
    if (truncImpl == nullptr)
        throw runtime_error("ERROR: Undefined Ewald truncation method.");
    EwaldStructureData sData{volume, numAtoms};
    truncImpl->calculateParameters(GlobSett, sData, params);
}
void EwaldSetup::logEwaldCutoffs(Log& log, double const lengthConversion) const
{
    if (!publishedNewCutoffs())
    {
        EwaldParameters ewaldParams = params.toPhysicalUnits(lengthConversion);
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << "Ewald parameters for this structure changed:\n";
        log << strpr("Real space cutoff:         %16.8E\n",
                     ewaldParams.rCut);
        log << strpr("Reciprocal space cutoff:   %16.8E\n",
                     ewaldParams.kCut);
        log << strpr("Ewald screening parameter: %16.8E\n",
                     ewaldParams.eta);
        if (!isEstimateReliable())
            log << strpr("WARNING: Ewald truncation error estimate may not be "
                         "reliable, better compare it\n"
                         "         with high accuracy settings.\n");
    }
}
bool EwaldSetup::publishedNewCutoffs() const
{
    return truncImpl->publishedNewCutoffs();
}
bool EwaldSetup::isEstimateReliable() const
{
    return truncImpl->isEstimateReliable(GlobSett, params);
}
