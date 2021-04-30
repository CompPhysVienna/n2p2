#ifndef EXAMPLE_NNP_CHECKF_H
#define EXAMPLE_NNP_CHECKF_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"

struct Example_nnp_checkf : public Example_nnp
{
    Example_nnp_checkf(std::string name)
        : Example_nnp("nnp-checkf", name) {};
};

template<>
void BoostDataContainer<Example_nnp_checkf>::setup()
{
    Example_nnp_checkf* e = nullptr;

    examples.push_back(Example_nnp_checkf("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "";

    examples.push_back(Example_nnp_checkf("LJ"));
    e = &(examples.back());
    e->args = "";

    return;
}

#endif
