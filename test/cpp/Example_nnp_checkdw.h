#ifndef EXAMPLE_NNP_CHECKDW_H
#define EXAMPLE_NNP_CHECKDW_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"

struct Example_nnp_checkdw : public Example_nnp
{
    Example_nnp_checkdw(std::string name)
        : Example_nnp("nnp-checkdw", name) {};
};

template<>
void BoostDataContainer<Example_nnp_checkdw>::setup()
{
    Example_nnp_checkdw* e = nullptr;

    // Way too slow for CI test.
    //examples.push_back(Example_nnp_checkdw("H2O_RPBE-D3"));
    //e = &(examples.back());
    //e->args = "0 ";

    examples.push_back(Example_nnp_checkdw("LJ"));
    e = &(examples.back());
    e->args = "0 ";

    return;
}

#endif
