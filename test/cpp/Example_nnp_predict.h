#ifndef EXAMPLE_NNP_DIST_H
#define EXAMPLE_NNP_DIST_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"
#include <vector>

struct Example_nnp_predict : public Example_nnp
{
    double                   energy;
    std::vector<std::string> createdFiles;

    Example_nnp_predict(std::string name)
        : Example_nnp("nnp-predict", name) {};
};

template<>
void BoostDataContainer<Example_nnp_predict>::setup()
{
    Example_nnp_predict* e = nullptr;

    examples.push_back(Example_nnp_predict("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "0 ";
    e->energy = -2.7564547347815904E+04;
    e->createdFiles.push_back("energy.out");
    e->createdFiles.push_back("nnatoms.out");
    e->createdFiles.push_back("nnforces.out");
    e->createdFiles.push_back("output.data");

    examples.push_back(Example_nnp_predict("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "0 ";
    e->energy = -5.7365603183874578E+02;
    e->createdFiles.push_back("energy.out");
    e->createdFiles.push_back("nnatoms.out");
    e->createdFiles.push_back("nnforces.out");
    e->createdFiles.push_back("output.data");

    examples.push_back(Example_nnp_predict("Anisole_SCAN"));
    e = &(examples.back());
    e->args = "0 ";
    e->energy = -9.6749614724733078E+02;
    e->createdFiles.push_back("energy.out");
    e->createdFiles.push_back("nnatoms.out");
    e->createdFiles.push_back("nnforces.out");
    e->createdFiles.push_back("output.data");

    examples.push_back(Example_nnp_predict("DMABN_SCAN"));
    e = &(examples.back());
    e->args = "0 ";
    e->energy = -7.6864264790108564E+01;
    e->createdFiles.push_back("energy.out");
    e->createdFiles.push_back("nnatoms.out");
    e->createdFiles.push_back("nnforces.out");
    e->createdFiles.push_back("output.data");

    examples.push_back(Example_nnp_predict("Ethylbenzene_SCAN"));
    e = &(examples.back());
    e->args = "0 ";
    e->energy = -8.2198742206281679E+02;
    e->createdFiles.push_back("energy.out");
    e->createdFiles.push_back("nnatoms.out");
    e->createdFiles.push_back("nnforces.out");
    e->createdFiles.push_back("output.data");

    return;
}

#endif
