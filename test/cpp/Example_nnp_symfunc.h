#ifndef EXAMPLE_NNP_DIST_H
#define EXAMPLE_NNP_DIST_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"
#include <vector>

struct Example_nnp_symfunc : public Example_nnp
{
    std::vector<std::string> createdFiles;

    Example_nnp_symfunc(std::string name)
        : Example_nnp("nnp-symfunc", name) {};
};

template<>
void BoostDataContainer<Example_nnp_symfunc>::setup()
{
    Example_nnp_symfunc* e = nullptr;

    examples.push_back(Example_nnp_symfunc("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "200 ";
    e->createdFiles.push_back("sf.001.0001.out");
    e->createdFiles.push_back("sf.001.0027.out");
    e->createdFiles.push_back("sf.008.0001.out");
    e->createdFiles.push_back("sf.008.0030.out");

    return;
}

#endif
