#ifndef EXAMPLE_NNP_COMP2_H
#define EXAMPLE_NNP_COMP2_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"

struct Example_nnp_comp2 : public Example_nnp
{
    Example_nnp_comp2(std::string name)
        : Example_nnp("nnp-comp2", name) {};
};

template<>
void BoostDataContainer<Example_nnp_comp2>::setup()
{
    Example_nnp_comp2* e = nullptr;

    // "compare" mode requires an even number of MPI processes (one half
    // evaluates NNP 1, the other NNP 2), so the plain application binary
    // must be launched through mpirun here (every other regression test in
    // this suite runs single-process and relies on OpenMPI's singleton
    // mode, which is not an option for nnp-comp2's "compare" mode).
    // --oversubscribe guards against environments that otherwise refuse to
    // place 2 ranks (e.g. CI runners reporting a single usable slot).
    examples.push_back(Example_nnp_comp2("Cu2S_PBE"));
    e = &(examples.back());
    e->command = "mpirun --oversubscribe -np 2 " + e->command;
    e->args = "compare";

    return;
}

#endif
