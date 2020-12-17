#ifndef EXAMPLE_NNP_DIST_H
#define EXAMPLE_NNP_DIST_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"
#include <vector>

struct Example_nnp_prune : public Example_nnp
{
    std::vector<std::string> createdFiles;

    Example_nnp_prune(std::string name)
        : Example_nnp("nnp-prune", name) {};
};

template<>
void BoostDataContainer<Example_nnp_prune>::setup()
{
    Example_nnp_prune* e = nullptr;

    examples.push_back(Example_nnp_prune("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "range 1.0E-1 ";
    e->createdFiles.push_back("output-prune-range.nn");

    examples.push_back(Example_nnp_prune("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "sensitivity 0.5 max ";
    e->createdFiles.push_back("output-prune-sensitivity.nn");

    examples.push_back(Example_nnp_prune("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "range 1.0E-1 ";
    e->createdFiles.push_back("output-prune-range.nn");

    examples.push_back(Example_nnp_prune("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "sensitivity 0.5 max ";
    e->createdFiles.push_back("output-prune-sensitivity.nn");

    return;
}

#endif
