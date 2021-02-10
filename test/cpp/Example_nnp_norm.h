#ifndef EXAMPLE_NNP_DIST_H
#define EXAMPLE_NNP_DIST_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"
#include <vector>

struct Example_nnp_norm : public Example_nnp
{
    std::vector<std::string> createdFiles;

    Example_nnp_norm(std::string name)
        : Example_nnp("nnp-norm", name) {};
};

template<>
void BoostDataContainer<Example_nnp_norm>::setup()
{
    Example_nnp_norm* e = nullptr;

    examples.push_back(Example_nnp_norm("H2O_RPBE-D3"));
    e = &(examples.back());
    e->createdFiles.push_back("evsv.dat");
    e->createdFiles.push_back("input.nn.bak");
    e->createdFiles.push_back("output.data");

    examples.push_back(Example_nnp_norm("Cu2S_PBE"));
    e = &(examples.back());
    e->createdFiles.push_back("evsv.dat");
    e->createdFiles.push_back("input.nn.bak");
    e->createdFiles.push_back("output.data");

    return;
}

#endif
