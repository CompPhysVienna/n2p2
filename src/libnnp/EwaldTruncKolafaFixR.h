//
// Created by philipp on 2/17/23.
//

#ifndef N2P2_EWALDTRUNCKOLAFAFIXR_H
#define N2P2_EWALDTRUNCKOLAFAFIXR_H

#include "IEwaldTrunc.h"
#include <cmath>

namespace nnp
{
    class EwaldTruncKolafaFixR : public IEwaldTrunc
    {
    public:
        void calculateParameters(EwaldGlobalSettings const& settings,
                                 EwaldStructureData const& sData,
                                 EwaldParameters &params) override;
        bool publishedNewCutoffs() override;
        bool isEstimateReliable(
                EwaldGlobalSettings const& settings,
                EwaldParameters const& params) const override;

    private:
        double volume = 0.0;
        double fourPiEps = 1.0;
        std::size_t numAtoms = 0;
        bool newCutoffs = true;
        bool newCutoffsWerePublished = false;

        double C = 0.0;

        double calculateEta(double const rCut, double const prec) const;
        double calculateKCut(double const eta, double const prec) const;
        void calculateC(double const qMax);
    };


} // nnp

#endif //N2P2_EWALDTRUNCKOLAFAFIXR_H
