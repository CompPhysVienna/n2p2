#ifndef EXAMPLE_NNP_ATOMENV_H
#define EXAMPLE_NNP_ATOMENV_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"
#include <map>
#include <string>
#include <utility>

struct Example_nnp_atomenv : public Example_nnp
{
    std::vector<std::string>                 elements;
    std::map<std::string, std::size_t>       numSF;
    std::map<
        std::pair<std::string, std::string>,
        std::pair<std::size_t, std::size_t>> neighCut;

    Example_nnp_atomenv(std::string name)
        : Example_nnp("nnp-atomenv", name) {};
};

template<>
void BoostDataContainer<Example_nnp_atomenv>::setup()
{
    Example_nnp_atomenv* e = nullptr;

    examples.push_back(Example_nnp_atomenv("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "100 6 2 4 4 ";
    e->elements.push_back("H");
    e->elements.push_back("O");
    e->numSF["H"] = 27;
    e->numSF["O"] = 30;
    e->neighCut[std::make_pair("H", "H")] = std::make_pair(6, 15);
    e->neighCut[std::make_pair("H", "O")] = std::make_pair(2, 19);
    e->neighCut[std::make_pair("O", "H")] = std::make_pair(4, 18);
    e->neighCut[std::make_pair("O", "O")] = std::make_pair(4, 16);

    examples.push_back(Example_nnp_atomenv("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "100 12 6 3 6 ";
    e->elements.push_back("S");
    e->elements.push_back("Cu");
    e->numSF["S"]  = 72;
    e->numSF["Cu"] = 66;
    e->neighCut[std::make_pair("S" , "S" )] = std::make_pair(12, 48);
    e->neighCut[std::make_pair("S" , "Cu")] = std::make_pair( 6, 47);
    e->neighCut[std::make_pair("Cu", "S" )] = std::make_pair( 3, 45);
    e->neighCut[std::make_pair("Cu", "Cu")] = std::make_pair( 6, 38);

    return;
}

#endif
