//
// Created by philipp on 2/17/23.
//

#ifndef N2P2_EWALDTRUNCKOLAFAOPTETA_H
#define N2P2_EWALDTRUNCKOLAFAOPTETA_H

#include "IEwaldTrunc.h"

namespace nnp
{

    class EwaldTruncKolafaOptEta : public IEwaldTrunc
    {
    public:
        void calculateParameters(EwaldGlobalSettings const& settings,
                                 EwaldStructureData const& sData,
                                 EwaldParameters &params) override;
        bool publishedNewCutoffs() override;
        virtual bool isEstimateReliable(
                EwaldGlobalSettings const& ,
                EwaldParameters const& ) const override { return true; };
    private:
        bool newCutoffs = true;
        bool newCutoffsWerePublished = false;
        double C = 0.0;
        double s = 1.0;
        double eta = 1.0;
        double prec = 1.e-6;
        double volume = 0.0;
        double fourPiEps = 1.0;
        std::size_t numAtoms = 0;

        void calculateEta();
        double calculateRCut();
        double calculateKCut();
        void calculateS();
        void calculateC(double const qMax);
    };

} // nnp

#endif //N2P2_EWALDTRUNCKOLAFAOPTETA_H
