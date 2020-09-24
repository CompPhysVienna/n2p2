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

struct ExampleActivation : public Example
{
    std::string          activation;
    std::vector<double>  x;
    std::vector<double>  f;

    ExampleActivation();

    ExampleActivation(std::string name)
    {
        this->name = name;
        this->description = std::string("Activation example \"")
                          + this->name
                          + "\"";
    }
};

template<>
void BoostDataContainer<ExampleActivation>::setup()
{
    ExampleActivation* e = nullptr;

    examples.push_back(ExampleActivation("Linear \"l\""));
    e = &(examples.back());
    e->activation = "l";
    e->x = {-1.5, -0.7, 0.0, 0.7, 1.5};
    e->f = {-1.5, -0.7, 0.0, 0.7, 1.5};

    examples.push_back(ExampleActivation("Hyperbolic Tangent \"t\""));
    e = &(examples.back());
    e->activation = "t";
    e->x = {-1.5, -0.7, 0.0, 0.7, 1.5};
    e->f = {-9.0514825364486640E-01, -6.0436777711716361E-01,  0.0,  6.0436777711716361E-01,  9.0514825364486640E-01};

    examples.push_back(ExampleActivation("Logistic \"s\""));
    e = &(examples.back());
    e->activation = "s";
    e->x = {-1.5, -0.7, 0.0, 0.7, 1.5};
    e->f = {1.8242552380635635E-01,  3.3181222783183389E-01,  5.0E-01,  6.6818777216816616E-01,  8.1757447619364365E-01};

    examples.push_back(ExampleActivation("Softplus \"p\""));
    e = &(examples.back());
    e->activation = "p";
    e->x = {-1.5, -0.7, 0.0, 0.7, 1.5};
    e->f = {2.0141327798275246E-01,  4.0318604888545784E-01,  6.9314718055994529E-01,  1.1031860488854579E+00,  1.7014132779827524E+00};

    examples.push_back(ExampleActivation("ReLU \"r\""));
    e = &(examples.back());
    e->activation = "r";
    e->x = {-1.5, -0.7, -0.1, 0.1, 0.7, 1.5};
    e->f = {0.0, 0.0, 0.0, 0.1, 0.7, 1.5};

    examples.push_back(ExampleActivation("Gaussian \"g\""));
    e = &(examples.back());
    e->activation = "g";
    e->x = {-1.5, -0.7, 0.0, 0.7, 1.5};
    e->f = {3.2465246735834974E-01,  7.8270453824186814E-01,  1.0E+00,  7.8270453824186814E-01,  3.2465246735834974E-01};

    examples.push_back(ExampleActivation("Reverse Logistic \"S\""));
    e = &(examples.back());
    e->activation = "S";
    e->x = {-1.5, -0.7, 0.0, 0.7, 1.5};
    e->f = {8.1757447619364365E-01,  6.6818777216816616E-01,  5.0E-01,  3.3181222783183384E-01,  1.8242552380635635E-01};

    examples.push_back(ExampleActivation("Exponential \"e\""));
    e = &(examples.back());
    e->activation = "e";
    e->x = {-1.5, -0.7, 0.0, 0.7, 1.5};
    e->f = {4.4816890703380645E+00,  2.0137527074704766E+00,  1.0E+00,  4.9658530379140953E-01,  2.2313016014842982E-01};

    examples.push_back(ExampleActivation("Harmonic \"h\""));
    e = &(examples.back());
    e->activation = "h";
    e->x = {-1.5, -0.7, 0.0, 0.7, 1.5};
    e->f = {2.25E+00,  4.8999999999999994E-01,  0.0,  4.8999999999999994E-01,  2.25E+00};

    return;
}

#endif
