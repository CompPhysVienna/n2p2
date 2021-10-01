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

    examples.push_back(Example_nnp_train("H2O_RPBE-D3"));
    e = &(examples.back());
    e->lastEpoch = 5;
    e->rmseEnergyTrain = 5.19563489E-04;
    e->rmseEnergyTest  = 4.19002577E-04;
    e->rmseForcesTrain = 2.41271658E-02;
    e->rmseForcesTest  = 1.59314285E-02;

    return;
}

#endif
