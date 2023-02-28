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
        bool cutoffsChanged() const override {return newCutoffs;};
        bool isEstimateReliable(
                EwaldGlobalSettings const& settings,
                EwaldParameters const& params) const override;

    private:
        double volume = 0.0;
        std::size_t numAtoms = 0;
        bool newCutoffs = true;

        double C = 0.0;

        double calculateEta(double const rCut, double const prec) const;
        double calculateKCut(double const eta, double const prec) const;
        void calculateC(double const qMax);
    };


} // nnp

#endif //N2P2_EWALDTRUNCKOLAFAFIXR_H
