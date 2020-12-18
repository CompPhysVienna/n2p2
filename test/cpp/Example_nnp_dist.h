#ifndef EXAMPLE_NNP_DIST_H
#define EXAMPLE_NNP_DIST_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"
#include <vector>

struct Example_nnp_dist : public Example_nnp
{
    std::vector<std::string> createdFiles;

    Example_nnp_dist(std::string name)
        : Example_nnp("nnp-dist", name) {};
};

template<>
void BoostDataContainer<Example_nnp_dist>::setup()
{
    Example_nnp_dist* e = nullptr;

    examples.push_back(Example_nnp_dist("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "12.0 200 1 H O ";
    e->createdFiles.push_back("rdf_H_H.out");
    e->createdFiles.push_back("rdf_H_O.out");
    e->createdFiles.push_back("rdf_O_O.out");
    e->createdFiles.push_back("adf_H_H_H.out");
    e->createdFiles.push_back("adf_H_H_O.out");
    e->createdFiles.push_back("adf_H_O_O.out");
    e->createdFiles.push_back("adf_O_H_H.out");
    e->createdFiles.push_back("adf_O_H_O.out");
    e->createdFiles.push_back("adf_O_O_O.out");

    examples.push_back(Example_nnp_dist("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "6.0 200 1 S Cu ";
    e->createdFiles.push_back("rdf_S_S.out");
    e->createdFiles.push_back("rdf_S_Cu.out");
    e->createdFiles.push_back("rdf_Cu_Cu.out");
    e->createdFiles.push_back("adf_S_S_S.out");
    e->createdFiles.push_back("adf_S_S_Cu.out");
    e->createdFiles.push_back("adf_S_Cu_Cu.out");
    e->createdFiles.push_back("adf_Cu_S_S.out");
    e->createdFiles.push_back("adf_Cu_S_Cu.out");
    e->createdFiles.push_back("adf_Cu_Cu_Cu.out");

    return;
}

#endif
