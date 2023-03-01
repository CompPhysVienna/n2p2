//
// Created by philipp on 2/17/23.
//

#ifndef N2P2_EWALDTRUNCJACKSON_H
#define N2P2_EWALDTRUNCJACKSON_H

#include "IEwaldTrunc.h"
namespace nnp
{
    class EwaldTruncJackson : public IEwaldTrunc
    {
    public:
        void calculateParameters(EwaldGlobalSettings const& settings,
                                 EwaldStructureData const& sData,
                                 EwaldParameters &params) override;
        bool publishedNewCutoffs() override;
        virtual bool isEstimateReliable(
                EwaldGlobalSettings const&,
                EwaldParameters const&) const override {return true;};
    private:
        bool newCutoffs = true;
        bool newCutoffsWerePublished = false;
        double volume = 0.0;

        double calculateEta() const;
        double calculateRCut(double const eta, double const prec) const;
        double calculateKCut(double const eta, double const prec) const;

    };

} // nnp

#endif //N2P2_EWALDTRUNCJACKSON_H
