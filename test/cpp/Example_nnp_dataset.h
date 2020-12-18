#ifndef EXAMPLE_NNP_DATASET_H
#define EXAMPLE_NNP_DATASET_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"

struct Example_nnp_dataset : public Example_nnp
{
    Example_nnp_dataset(std::string name)
        : Example_nnp("nnp-dataset", name) {};
};

template<>
void BoostDataContainer<Example_nnp_dataset>::setup()
{
    Example_nnp_dataset* e = nullptr;

    examples.push_back(Example_nnp_dataset("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "0 ";

    examples.push_back(Example_nnp_dataset("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "0 ";

    return;
}

#endif
