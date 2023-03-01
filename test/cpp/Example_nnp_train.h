#ifndef EXAMPLE_NNP_TRAIN_H
#define EXAMPLE_NNP_TRAIN_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"

struct Example_nnp_train : public Example_nnp
{
    std::size_t lastEpoch;
    double      rmseChargesTrain;
    double      rmseChargesTest;
    double      rmseEnergyTrain;
    double      rmseEnergyTest;
    double      rmseForcesTrain;
    double      rmseForcesTest;

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

    examples.push_back(Example_nnp_train("H2O_RPBE-D3_4G"));
    e = &(examples.back());
    e->args = "1";
    e->lastEpoch = 10;
    e->rmseChargesTrain = 3.60544679E-04;
    e->rmseChargesTest  = 4.60549641E-04;

    examples.push_back(Example_nnp_train("H2O_RPBE-D3_4G"));
    e = &(examples.back());
    e->args = "2";
    e->lastEpoch = 10;
    e->rmseEnergyTrain = 2.15640267E-05;
    e->rmseEnergyTest  = 1.30155783E-05;
    e->rmseForcesTrain = 2.07286126E-04;
    e->rmseForcesTest  = 2.20371908E-04;

    //examples.push_back(Example_nnp_train("H2O_RPBE-D3"));
    //e = &(examples.back());
    //e->lastEpoch = 10;
    //e->rmseEnergyTrain = 1.02312914E-04;
    //e->rmseEnergyTest  = 5.28837597E-04;
    //e->rmseForcesTrain = 1.67069652E-02;
    //e->rmseForcesTest  = 1.07708383E-02;

    return;
}

#endif
