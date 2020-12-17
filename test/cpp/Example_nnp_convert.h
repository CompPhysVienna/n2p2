#ifndef EXAMPLE_NNP_CONVERT_H
#define EXAMPLE_NNP_CONVERT_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"
#include <vector>

struct Example_nnp_convert : public Example_nnp
{
    std::vector<std::string> createdFiles;

    Example_nnp_convert(std::string name)
        : Example_nnp("nnp-convert", name) {};
};

template<>
void BoostDataContainer<Example_nnp_convert>::setup()
{
    Example_nnp_convert* e = nullptr;

    examples.push_back(Example_nnp_convert("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "xyz S Cu ";
    e->createdFiles.push_back("input.xyz");

    examples.push_back(Example_nnp_convert("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "poscar S Cu ";
    e->createdFiles.push_back("POSCAR_1");
    e->createdFiles.push_back("POSCAR_20");

    return;
}

#endif
