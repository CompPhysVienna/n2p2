#ifndef EXAMPLE_NNP_FPS_H
#define EXAMPLE_NNP_FPS_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"

struct Example_nnp_fps : public Example_nnp
{
    std::vector<int> chosenStructures;

    Example_nnp_fps(std::string name) : Example_nnp("nnp-fps", name) {};
};

template<>
void BoostDataContainer<Example_nnp_fps>::setup()
{
    Example_nnp_fps* e = nullptr;

    examples.push_back(Example_nnp_fps("HBeW"));
    e = &(examples.back());
    e->args = "20 0";
    e->chosenStructures.push_back(0);
    e->chosenStructures.push_back(9);
    e->chosenStructures.push_back(16);
    e->chosenStructures.push_back(22);
    e->chosenStructures.push_back(35);
    e->chosenStructures.push_back(41);
    e->chosenStructures.push_back(47);
    e->chosenStructures.push_back(53);
    e->chosenStructures.push_back(58);
    e->chosenStructures.push_back(63);
    e->chosenStructures.push_back(68);
    e->chosenStructures.push_back(77);
    e->chosenStructures.push_back(86);
    e->chosenStructures.push_back(94);
    e->chosenStructures.push_back(103);
    e->chosenStructures.push_back(111);
    e->chosenStructures.push_back(119);
    e->chosenStructures.push_back(126);
    e->chosenStructures.push_back(133);
    e->chosenStructures.push_back(139);

    return;
}

#endif
