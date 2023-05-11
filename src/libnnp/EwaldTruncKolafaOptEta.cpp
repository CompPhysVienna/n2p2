//
// Created by philipp on 2/17/23.
//

#include "EwaldTruncKolafaOptEta.h"
#include <cmath>
#include <algorithm>

using namespace std;

namespace nnp
{
    // Ratio of computing times for one real space and k space iteration.
    double constexpr TrOverTk = 3.676;

    void EwaldTruncKolafaOptEta::calculateParameters(
            const EwaldGlobalSettings &settings,
            const EwaldStructureData &sData, EwaldParameters &params)
    {
        size_t N = sData.getNumAtoms();
        double V = sData.getVolume();
        newCutoffs = (volume != V || numAtoms != N);
        if (!newCutoffs) return;
        newCutoffsWerePublished = false;

        volume = V;
        numAtoms = N;
        prec = settings.precision;
        fourPiEps = settings.fourPiEps;

        calculateC(settings.maxCharge);
        calculateEta();

        eta = std::max(eta,settings.maxQSigma);
        params.eta = eta;
        params.rCut = calculateRCut();
        params.kCut = calculateKCut();
    }
    bool EwaldTruncKolafaOptEta::publishedNewCutoffs()
    {
        bool answer = newCutoffsWerePublished;
        newCutoffsWerePublished = true;
        return answer;
    }

    void EwaldTruncKolafaOptEta::calculateEta()
    {
        double constexpr acceptError = 1.e-10;

        // Initial approximation
        double eta0 = pow(1 / TrOverTk * pow(volume, 2.0)
                    / pow(2 * M_PI, 3.0), 1.0 / 6.0);

        // Selfconsistent calculation of eta
        eta = eta0;
        double oldEta;
        double relError = 1.0;
        s = 1.0;
        while (relError > acceptError)
        {
            calculateS();
            oldEta = eta;
            eta = eta0 * pow((1 + 1 / (2 * pow(s, 2))), 1.0 / 6.0);
            relError = abs(oldEta - eta) / eta;
        }
    }
    double EwaldTruncKolafaOptEta::calculateRCut()
    {
        return sqrt(2)*eta*s;
    }

    double EwaldTruncKolafaOptEta::calculateKCut()
    {
        return sqrt(2)*s / eta;
    }

    void EwaldTruncKolafaOptEta::calculateS()
    {
        double constexpr acceptError = 1.e-10;
        double relXError = 1.0;
        double relYError = 1.0;
        double y = prec / C * sqrt(eta/2);

        if (s <= 0.0) s = 0.5;

        double step;
        while (relXError > acceptError || relYError > acceptError)
        {
            step = 2*s / (4*pow(s,2.0) + 1)
                   * (1 - sqrt(s) / exp(-pow(s,2.0)) * y);

            // If s would become negative, try smaller start value.
            if (s <= -step)
            {
                s /= 2;
                step = 1.0;
                continue;
            }
            s += step;
            relYError = abs((exp(-pow(s,2.0)) / sqrt(s) - y) / y);
            relXError = abs(step/s);
        }
    }

    void EwaldTruncKolafaOptEta::calculateC(double const qMax)
    {
        C = pow(2, 3.0/4) * pow(qMax,2.0) * sqrt(numAtoms/volume) / fourPiEps;
    }
} // nnp
