#ifndef EXAMPLE_NNP_SFCLUST_H
#define EXAMPLE_NNP_SFCLUST_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"

struct Example_nnp_sfclust : public Example_nnp
{
    Example_nnp_sfclust(std::string name)
        : Example_nnp("nnp-sfclust", name) {};
};

template<>
void BoostDataContainer<Example_nnp_sfclust>::setup()
{
    Example_nnp_sfclust* e = nullptr;

    examples.push_back(Example_nnp_sfclust("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "100 6 2 4 4 ";

    examples.push_back(Example_nnp_sfclust("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "100 12 6 3 6 ";

    return;
}

#endif
