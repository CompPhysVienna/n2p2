#ifndef EXAMPLE_NNP_TRAIN_H
#define EXAMPLE_NNP_TRAIN_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"

#include <cstddef> // std::size_t
#include <limits>  // std::numeric_limits
#include <vector>  // std::vector

// One row of learning-curve.out: RMSEs of charges (stage 1 of a 4G-HDNNP
// training) or of energies and forces (everything else).
struct LearningCurveRow
{
    std::size_t epoch;
    double      rmseTrain;
    double      rmseTest;
    double      rmseForcesTrain;
    double      rmseForcesTest;
};

struct Example_nnp_train : public Example_nnp
{
    std::size_t lastEpoch;
    double      rmseChargesTrain;
    double      rmseChargesTest;
    double      rmseEnergyTrain;
    double      rmseEnergyTest;
    double      rmseForcesTrain;
    double      rmseForcesTest;
    double      accuracy;
    // Epochs 0 (untrained network, a pure evaluation) and 1 (after the first
    // epoch of updates). Unlike the last epoch, these stay comparable when
    // the summation order changes (compiler, parallelization): there,
    // last-digit differences are amplified epoch after epoch by the
    // selection of update candidates until the trained networks differ
    // visibly. Checking them localizes a regression to the evaluation
    // (epoch 0) or the first updates (epoch 1).
    std::vector<LearningCurveRow> earlyEpochs;
    // Relative tolerance for the last epoch, in addition to accuracy. 0 for
    // examples whose last epoch is reproducible across machines; a sanity
    // bound for those where the amplification above is large.
    double      accuracyLastEpochRelative;

    Example_nnp_train(std::string name) :
        Example_nnp("nnp-train", name),
        accuracyLastEpochRelative(0.0) {};
};

template<>
void BoostDataContainer<Example_nnp_train>::setup()
{
    Example_nnp_train* e = nullptr;

    examples.push_back(Example_nnp_train("LJ"));
    e = &(examples.back());
    e->earlyEpochs = {{0, 2.94929187E-01, 3.04351093E-01, 2.01914883E+00, 4.76367591E+00},
                      {1, 5.72511237E-02, 5.65587888E-02, 3.59515825E-01, 1.40104723E+00}};
    e->lastEpoch = 10;
    e->rmseEnergyTrain = 1.70594742E-03;
    e->rmseEnergyTest  = 1.22642208E-03;
    e->rmseForcesTrain = 3.77302629E-02;
    e->rmseForcesTest  = 2.25135617E-01;
    e->accuracy        = 10.0 * std::numeric_limits<double>::epsilon();

    examples.push_back(Example_nnp_train("H2O_RPBE-D3"));
    e = &(examples.back());
    e->earlyEpochs = {{0, 1.17658233E-03, 1.09663051E-03, 2.72051656E-02, 1.64585427E-02},
                      {1, 9.73051110E-04, 9.86836069E-04, 2.42340155E-02, 1.16747598E-02}};
    e->lastEpoch = 2;
    e->rmseEnergyTrain = 7.19539183E-04;
    e->rmseEnergyTest  = 3.57910403E-04;
    e->rmseForcesTrain = 2.46334310E-02;
    e->rmseForcesTest  = 1.22544569E-02;
    e->accuracy        = 1.0E-8;

    examples.push_back(Example_nnp_train("H2O_RPBE-D3_norm-force"));
    e = &(examples.back());
    e->earlyEpochs = {{0, 2.61033391E-03, 1.43716083E-03, 4.19486080E-02, 3.69378470E-02},
                      {1, 5.86058543E-04, 1.02174122E-03, 9.60525638E-03, 8.68088582E-03}};
    e->lastEpoch = 2;
    e->rmseEnergyTrain = 3.02402814E-04;
    e->rmseEnergyTest  = 1.01753064E-04;
    e->rmseForcesTrain = 6.08971134E-03;
    e->rmseForcesTest  = 4.60947053E-03;
    e->accuracy        = 100.0 * std::numeric_limits<double>::epsilon();

    examples.push_back(Example_nnp_train("H2O_RPBE-D3_4G"));
    e = &(examples.back());
    e->args = "1";
    e->earlyEpochs = {{0, 1.53886213E-01, 1.56144377E-01, 0.0, 0.0},
                      {1, 1.98168103E-03, 3.29080752E-03, 0.0, 0.0}};
    e->lastEpoch = 10;
    e->rmseChargesTrain = 3.60544679E-04;
    e->rmseChargesTest  = 4.60549641E-04;
    e->accuracy        = 100.0 * std::numeric_limits<double>::epsilon();

    examples.push_back(Example_nnp_train("H2O_RPBE-D3_4G"));
    e = &(examples.back());
    e->args = "2";
    e->earlyEpochs = {{0, 2.54963983E+01, 2.54883959E+01, 6.39401966E-01, 6.42712759E-01},
                      {1, 2.46483134E-04, 2.33868144E-04, 7.59729026E-03, 7.42592511E-03}};
    e->lastEpoch = 10;
    e->rmseEnergyTrain = 2.15640267E-05;
    e->rmseEnergyTest  = 1.30155783E-05;
    e->rmseForcesTrain = 2.07286126E-04;
    e->rmseForcesTest  = 2.20371908E-04;
    e->accuracy        = 1E-13;

    // Periodic solid-state system (Cu2S), no long-range electrostatics --
    // complements the molecular/4G examples above with a plain short-range
    // 2G training run on a different chemistry, widening the golden-master
    // net this test provides.
    // Reference values from the current code, epoch 10 of
    // examples/nnp-train/Cu2S_PBE/input.nn (deterministic: fixed
    // random_seed).
    examples.push_back(Example_nnp_train("Cu2S_PBE"));
    e = &(examples.back());
    e->earlyEpochs = {{0, 2.01284122E-02, 6.72142235E-03, 3.30885002E-01, 4.21540979E-01},
                      {1, 9.49509887E-03, 1.39112676E-02, 1.43674283E-01, 1.87872840E-01}};
    e->lastEpoch = 10;
    e->rmseEnergyTrain = 1.59193717E-04;
    e->rmseEnergyTest  = 2.64525224E-04;
    e->rmseForcesTrain = 3.12362540E-02;
    e->rmseForcesTest  = 4.03303534E-02;
    e->accuracy        = 1.0E-8;
    // Last-digit differences grow about x10 per epoch here (1E-7 at epoch 2,
    // 34% at epoch 7), so on another machine (CPU, compiler flags, BLAS) the
    // epoch-10 RMSEs differ visibly (e.g. +36% in CI). Epochs 0 and 1 are
    // checked to accuracy; epoch 10 only within a factor of two.
    e->accuracyLastEpochRelative = 1.0;

    return;
}

#endif
