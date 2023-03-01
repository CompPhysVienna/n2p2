//
// Created by philipp on 2/17/23.
//

#ifndef N2P2_IEWALDTRUNC_H
#define N2P2_IEWALDTRUNC_H

#include <cstddef>        // std::size_t

namespace nnp
{
    struct EwaldGlobalSettings
    {
        double precision = 1.0E-6;
        double maxCharge = 1.0;
        double maxQSigma = 1.0;
        /// Multiplicative constant \f$ 4 \pi \varepsilon_0 \f$.
        /// Value depends on unit system (e.g. normalization).
        double fourPiEps = 1.0;
    };

    class EwaldStructureData
    {
    public:
        EwaldStructureData(double const V, std::size_t const N)
                : volume{V}, numAtoms{N}
        {};
        ~EwaldStructureData() = default;

        double getVolume() const { return volume; };
        std::size_t getNumAtoms() const { return numAtoms; };
    private:
        double volume = 0.0;
        std::size_t numAtoms = 0;
    };

    struct EwaldParameters
    {
        /// Width of the gaussian screening charges.
        double eta = 0.0;
        /// Cutoff in real space.
        double rCut = 0.0;
        /// Cutoff in reciprocal space.
        double kCut = 0.0;

        EwaldParameters() = default;
        EwaldParameters(double oEta, double oRCut, double oKCut)
                : eta{oEta}, rCut{oRCut}, kCut{oKCut}
        {}

        EwaldParameters toPhysicalUnits(double const convLength) const
        {
            return EwaldParameters{eta / convLength,
                                   rCut / convLength,
                                   kCut * convLength};
        }
        EwaldParameters toNormalizedUnits(double const convLength) const
        {
            return toPhysicalUnits(1/convLength);
        }
    };

    class IEwaldTrunc
    {
    public:
        virtual void calculateParameters(EwaldGlobalSettings const& settings,
                                         EwaldStructureData const& sData,
                                         EwaldParameters &params) = 0;
        virtual bool publishedNewCutoffs() = 0;
        virtual bool isEstimateReliable(
                EwaldGlobalSettings const& settings,
                EwaldParameters const& params) const = 0;
        virtual ~IEwaldTrunc() = default;
    };
}
#endif //N2P2_IEWALDTRUNC_H
