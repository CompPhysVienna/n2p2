#ifndef EXAMPLE_NNP_TRAIN_H
#define EXAMPLE_NNP_TRAIN_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"

struct Example_nnp_train : public Example_nnp
{
    std::size_t lastEpoch;
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
    e->accuracy        = 10.0 * numeric_limits<double>::epsilon();

    examples.push_back(Example_nnp_train("H2O_RPBE-D3"));
    e = &(examples.back());
    e->lastEpoch = 2;
    e->rmseEnergyTrain = 7.19538940E-04;
    e->rmseEnergyTest  = 3.57909502E-04;
    e->rmseForcesTrain = 2.46334317E-02;
    e->rmseForcesTest  = 1.22544616E-02;
    e->accuracy        = 1.0E-9;

    examples.push_back(Example_nnp_train("H2O_RPBE-D3_norm-force"));
    e = &(examples.back());
    e->lastEpoch = 2;
    e->rmseEnergyTrain = 3.02402814E-04;
    e->rmseEnergyTest  = 1.01753064E-04;
    e->rmseForcesTrain = 6.08971134E-03;
    e->rmseForcesTest  = 4.60947053E-03;
    e->accuracy        = 100.0 * numeric_limits<double>::epsilon();

    return;
}

#endif
