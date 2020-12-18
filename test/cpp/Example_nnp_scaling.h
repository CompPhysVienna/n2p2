#ifndef EXAMPLE_NNP_SCALING_H
#define EXAMPLE_NNP_SCALING_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"

struct Example_nnp_scaling : public Example_nnp
{
    Example_nnp_scaling(std::string name)
        : Example_nnp("nnp-scaling", name) {};
};

template<>
void BoostDataContainer<Example_nnp_scaling>::setup()
{
    Example_nnp_scaling* e = nullptr;

    examples.push_back(Example_nnp_scaling("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "100 ";

    examples.push_back(Example_nnp_scaling("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "100 ";

    examples.push_back(Example_nnp_scaling("LJ"));
    e = &(examples.back());
    e->args = "100 ";

    return;
}

#endif
