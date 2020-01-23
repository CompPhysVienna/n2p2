#ifndef EXAMPLE_NNP_H
#define EXAMPLE_NNP_H

#include "Example.h"

struct Example_nnp : public Example
{
    std::string tool;
    std::string command;
    std::string args;
    std::string pathBin;
    std::string pathData;

    Example_nnp();

    Example_nnp(std::string tool, std::string name) : 
        pathBin("../../../bin"),
        pathData("../../examples")
    {
        this->name = name;
        this->tool = tool; 
        this->description = std::string("Test example \"")
                          + this->name
                          + "\" with tool \""
                          + this->tool + "\"";
        this->command = pathBin + "/" + this->tool;
        this->pathData += "/" + this->tool + "/" + this->name;
    }
};

#endif
