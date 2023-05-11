//
// Created by philipp on 2/17/23.
//

#include "EwaldTruncJackson.h"
#include <cmath>
#include <stdexcept>

namespace nnp
{
    void
    EwaldTruncJackson::calculateParameters(EwaldGlobalSettings const& settings,
                                           EwaldStructureData const& sData,
                                           EwaldParameters &params)
    {
        double V = sData.getVolume();
        newCutoffs = (V != volume);
        if (!newCutoffs) return;
        newCutoffsWerePublished = false;
        volume = V;

        if ( settings.precision >= 1.0 )
            throw std::runtime_error("ERROR: Ewald truncation method 0 "
                                     "(Jackson) requires precision < 1.0");
        params.eta = calculateEta();
        if ( params.eta <= 0.0 )
            throw std::runtime_error("ERROR: Ewald screening parameter eta is "
                                     "not positive, is the unit cell volume "
                                     "correct?");
        params.rCut = calculateRCut(params.eta, settings.precision);
        params.kCut = calculateKCut(params.eta, settings.precision);
        return;
    }

    bool EwaldTruncJackson::publishedNewCutoffs()
    {
        bool answer = newCutoffsWerePublished;
        newCutoffsWerePublished = true;
        return answer;
    }

    double EwaldTruncJackson::calculateEta() const
    {
        // Matrix version of eta.
        return 1.0 / sqrt(2.0 * M_PI) * pow(volume, 1.0 / 3.0);
    }

    double EwaldTruncJackson::calculateRCut(double const eta,
                                            double const prec) const
    {
        return sqrt(-2.0 * log(prec)) * eta;
    }

    double EwaldTruncJackson::calculateKCut(double const eta,
                                            double const prec) const
    {
        return sqrt(-2.0 * log(prec)) / eta;
    }
} // nnp