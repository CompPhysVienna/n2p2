#ifndef EXAMPLE_NNP_TRAIN_H
#define EXAMPLE_NNP_TRAIN_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"

#include <limits>  // std::numeric_limits

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

    Example_nnp_train(std::string name) : Example_nnp("nnp-train", name) {};
};

template<>
void BoostDataContainer<Example_nnp_train>::setup()
{
    Example_nnp_train* e = nullptr;

    examples.push_back(Example_nnp_train("LJ"));
    e = &(examples.back());
    e->lastEpoch = 10;
    e->rmseEnergyTrain = 1.70594742E-03;
    e->rmseEnergyTest  = 1.22642208E-03;
    e->rmseForcesTrain = 3.77302629E-02;
    e->rmseForcesTest  = 2.25135617E-01;
    e->accuracy        = 10.0 * std::numeric_limits<double>::epsilon();

    examples.push_back(Example_nnp_train("H2O_RPBE-D3"));
    e = &(examples.back());
    e->lastEpoch = 2;
    e->rmseEnergyTrain = 7.19539183E-04;
    e->rmseEnergyTest  = 3.57910403E-04;
    e->rmseForcesTrain = 2.46334310E-02;
    e->rmseForcesTest  = 1.22544569E-02;
    e->accuracy        = 1.0E-8;

    examples.push_back(Example_nnp_train("H2O_RPBE-D3_norm-force"));
    e = &(examples.back());
    e->lastEpoch = 2;
    e->rmseEnergyTrain = 3.02402814E-04;
    e->rmseEnergyTest  = 1.01753064E-04;
    e->rmseForcesTrain = 6.08971134E-03;
    e->rmseForcesTest  = 4.60947053E-03;
    e->accuracy        = 100.0 * std::numeric_limits<double>::epsilon();

    examples.push_back(Example_nnp_train("H2O_RPBE-D3_4G"));
    e = &(examples.back());
    e->args = "1";
    e->lastEpoch = 10;
    e->rmseChargesTrain = 3.60544679E-04;
    e->rmseChargesTest  = 4.60549641E-04;
    e->accuracy        = 100.0 * std::numeric_limits<double>::epsilon();

    examples.push_back(Example_nnp_train("H2O_RPBE-D3_4G"));
    e = &(examples.back());
    e->args = "2";
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
    e->lastEpoch = 10;
    e->rmseEnergyTrain = 1.59193717E-04;
    e->rmseEnergyTest  = 2.64525224E-04;
    e->rmseForcesTrain = 3.12362540E-02;
    e->rmseForcesTest  = 4.03303534E-02;
    e->accuracy        = 1.0E-8;

    return;
}

#endif
