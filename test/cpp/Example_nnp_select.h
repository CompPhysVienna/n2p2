#ifndef EXAMPLE_NNP_DIST_H
#define EXAMPLE_NNP_DIST_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"
#include <vector>

struct Example_nnp_select : public Example_nnp
{
    std::vector<std::string> createdFiles;

    Example_nnp_select(std::string name)
        : Example_nnp("nnp-select", name) {};
};

template<>
void BoostDataContainer<Example_nnp_select>::setup()
{
    Example_nnp_select* e = nullptr;

    examples.push_back(Example_nnp_select("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "interval 5 ";
    e->createdFiles.push_back("output.data");
    e->createdFiles.push_back("reject.data");

    examples.push_back(Example_nnp_select("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "random 0.5 123 ";
    e->createdFiles.push_back("output.data");
    e->createdFiles.push_back("reject.data");

    return;
}

#endif
