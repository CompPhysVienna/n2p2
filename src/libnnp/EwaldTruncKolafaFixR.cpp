//
// Created by philipp on 2/17/23.
//

#include "EwaldTruncKolafaFixR.h"
#include <stdexcept>

using namespace std;

namespace nnp
{
    void EwaldTruncKolafaFixR::calculateParameters(EwaldGlobalSettings const& settings,
                                                   EwaldStructureData const& sData,
                                                   EwaldParameters &params)
    {
        size_t N = sData.getNumAtoms();
        double V = sData.getVolume();

        newCutoffs = (V != volume || N != numAtoms);
        if (!newCutoffs) return;
        newCutoffsWerePublished = false;

        volume = V;
        fourPiEps = settings.fourPiEps;
        numAtoms = N;
        double rCut = params.rCut;
        if (rCut == 0.0)
            throw std::runtime_error("ERROR: Real space cutoff of Ewald "
                                     "summation cannot be zero.");

        calculateC(settings.maxCharge);
        params.eta = calculateEta(rCut, settings.precision);
        params.kCut = calculateKCut(params.eta, settings.precision);
        return;
    }
    bool EwaldTruncKolafaFixR::publishedNewCutoffs()
    {
        bool answer = newCutoffsWerePublished;
        newCutoffsWerePublished = true;
        return answer;
    }
    bool EwaldTruncKolafaFixR::isEstimateReliable(
            EwaldGlobalSettings const& settings,
            EwaldParameters const& params) const
    {
        return params.eta >= settings.maxQSigma;
    }

    double EwaldTruncKolafaFixR::calculateEta(double const rCut,
                                              double const prec) const
    {
        double x = sqrt(rCut/2) * prec / C;
        if (x >= 1.0)
            throw runtime_error("ERROR: Bad Ewald precision settings, try a "
                                "smaller value.");
        return rCut / sqrt(-2 * log(x));
    }

    double EwaldTruncKolafaFixR::calculateKCut(double const eta,
                                               double const prec) const
    {
        double constexpr acceptError = 1.e-10;
        double relXError = 1.0;
        double relYError = 1.0;
        double y = eta * prec / (sqrt(2) * C);

        double kCut = 1.0/eta;
        double kCutOld;
        while (relXError > acceptError || relYError > acceptError)
        {
            kCutOld = kCut;
            kCut -= 2 * kCut / (1 + 2*pow(kCut * eta,2.0))
                    * (-1 + sqrt(kCut) * exp(pow(kCut * eta,2.0)/2) * y);
            // If kCut becomes negative, start with smaller value.
            if (kCut <= 0)
            {
                kCut = kCutOld / 2;
                continue;
            }

            relXError = abs((kCut - kCutOld) / kCut);
            relYError = abs((1/sqrt(kCut)
                        * exp(-pow(kCut * eta,2.0) / 2) - y) / y);
        }
        return kCut;
    }

    void EwaldTruncKolafaFixR::calculateC(const double qMax)
    {
        C = 2 * pow(qMax,2.0) * sqrt(numAtoms/volume) / fourPiEps;
    }

} // nnp