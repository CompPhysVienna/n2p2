#ifndef EXAMPLENEURALNETWORX_H
#define EXAMPLENEURALNETWORX_H

#include <vector>
#include "Example.h"
#include "BoostDataContainer.h"
#include "NeuralNetworx.h"

struct ExampleNeuralNetworx : public Example
{
    std::size_t                                 numConnections;
    std::size_t                                 numWeights;
    std::size_t                                 numBiases;
    std::vector<std::size_t>                    numNeuronsPerLayer;
    std::vector<nnp::NeuralNetworx::Activation> activationPerLayer;
    std::vector<std::string>                    activationStringPerLayer;

    ExampleNeuralNetworx();

    ExampleNeuralNetworx(std::string name) : numConnections(0),
                                             numWeights    (0),
                                             numBiases     (0)
    {
        this->name = name;
        this->description = std::string("NeuralNetworx example \"")
                          + this->name
                          + "\"";
    }
};

template<>
void BoostDataContainer<ExampleNeuralNetworx>::setup()
{
    ExampleNeuralNetworx* e = nullptr;

    examples.push_back(ExampleNeuralNetworx("4-layer, single output"));
    e = &(examples.back());
    e->numNeuronsPerLayer = {10, 20, 15, 1};
    e->activationStringPerLayer = {"l", "t", "p", "l"};
    e->activationPerLayer = {nnp::NeuralNetworx::Activation::IDENTITY,
                             nnp::NeuralNetworx::Activation::TANH,
                             nnp::NeuralNetworx::Activation::SOFTPLUS,
                             nnp::NeuralNetworx::Activation::IDENTITY};
    e->numConnections = 551;
    e->numWeights = 515;
    e->numBiases = 36;

    examples.push_back(ExampleNeuralNetworx("10-layer, multiple output"));
    e = &(examples.back());
    e->numNeuronsPerLayer = {25, 20, 15, 30, 10, 4, 12, 13, 7, 5};
    e->activationStringPerLayer = {"l", "c", "e", "g", "h",
                                   "s", "r", "S", "p", "t"};
    e->activationPerLayer = {nnp::NeuralNetworx::Activation::IDENTITY,
                             nnp::NeuralNetworx::Activation::COS,
                             nnp::NeuralNetworx::Activation::EXP,
                             nnp::NeuralNetworx::Activation::GAUSSIAN,
                             nnp::NeuralNetworx::Activation::HARMONIC,
                             nnp::NeuralNetworx::Activation::LOGISTIC,
                             nnp::NeuralNetworx::Activation::RELU,
                             nnp::NeuralNetworx::Activation::REVLOGISTIC,
                             nnp::NeuralNetworx::Activation::SOFTPLUS,
                             nnp::NeuralNetworx::Activation::TANH};
    e->numConnections = 2036;
    e->numWeights = 1920;
    e->numBiases = 116;

    return;
}

#endif
