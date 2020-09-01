#ifndef EXAMPLE_LAMMPS_NNP_H
#define EXAMPLE_LAMMPS_NNP_H

#include "Example.h"
#include "BoostDataContainer.h"

struct Example_lammps_nnp : public Example
{
    std::string tool;
    std::string command;
    std::string args;
    std::string pathBin;
    std::string pathData;

    std::size_t lastTimeStep;
    double      potentialEnergy;
    double      totalEnergy;

    Example_lammps_nnp(std::string name) : 
        pathBin("../../../bin"),
        pathData("../../examples/interface-LAMMPS/")
    {
        this->name = name;
        this->tool = "lmp_mpi"; 
        this->description = std::string("Test example \"")
                          + this->name
                          + "\" with tool \""
                          + this->tool + "\"";
        this->command = pathBin + "/" + this->tool;
        this->pathData += "/" + this->name;
    }
};

template<>
void BoostDataContainer<Example_lammps_nnp>::setup()
{
    Example_lammps_nnp* e = nullptr;

    examples.push_back(Example_lammps_nnp("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "-in md.lmp ";
    e->lastTimeStep = 5;
    e->potentialEnergy = -6000559.4;
    e->totalEnergy = -6000220.2;

    examples.push_back(Example_lammps_nnp("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "-in md.lmp ";
    e->lastTimeStep = 100;
    e->potentialEnergy = -574.21185;
    e->totalEnergy = -573.6561;

    return;
}

#endif
