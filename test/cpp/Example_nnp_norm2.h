#ifndef EXAMPLE_NNP_NORM2_H
#define EXAMPLE_NNP_NORM2_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"
#include <vector>

struct Example_nnp_norm2 : public Example_nnp
{
    std::string              logFile;
    std::vector<std::string> createdFiles;
    std::vector<std::string> notCreatedFiles;

    Example_nnp_norm2(std::string name)
        : Example_nnp("nnp-norm2", name),
          logFile("nnp-norm2.log.0000") {};
};

template<>
void BoostDataContainer<Example_nnp_norm2>::setup()
{
    Example_nnp_norm2* e = nullptr;

    // Elements H (Z=1), O (Z=8).
    examples.push_back(Example_nnp_norm2("H2O_RPBE-D3"));
    e = &(examples.back());
    e->createdFiles.push_back("evsv.dat");
    e->createdFiles.push_back("input.nn.bak");
    e->createdFiles.push_back("weights.001.norm");
    e->createdFiles.push_back("weights.008.norm");

    // Elements S (Z=16), Cu (Z=29).
    examples.push_back(Example_nnp_norm2("Cu2S_PBE"));
    e = &(examples.back());
    e->createdFiles.push_back("evsv.dat");
    e->createdFiles.push_back("input.nn.bak");
    e->createdFiles.push_back("weights.016.norm");
    e->createdFiles.push_back("weights.029.norm");

    // 4G-HDNNP stage 2 (files of the nnp-train example): normalization is not
    // implemented for NNP types with charges, nnp-norm2 has to stop before
    // writing anything.
    examples.push_back(Example_nnp_norm2("H2O_RPBE-D3_4G"));
    e = &(examples.back());
    e->pathData = "../../examples/nnp-train/H2O_RPBE-D3_4G";
    e->args = "2 ";
    e->logFile = "nnp-norm2.log.0000.stage-2";
    e->notCreatedFiles.push_back("evsv.dat");
    e->notCreatedFiles.push_back("input.nn.bak");
    e->notCreatedFiles.push_back("weights.001.norm");
    e->notCreatedFiles.push_back("weights.008.norm");

    return;
}

#endif
