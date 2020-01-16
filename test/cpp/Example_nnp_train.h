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

    return;
}

#endif
