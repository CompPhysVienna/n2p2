// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "NeuralNetworx.h"
#include "utility.h"
#include <algorithm> // std::min, std::max
#include <iostream>
#include <stdexcept>

#define EXP_LIMIT 35.0

using namespace std;
using namespace nnp;

NeuralNetworx::NeuralNetworx(vector<size_t>     numNeuronsPerLayer,
                             vector<Activation> activationPerLayer)
{
    if (numNeuronsPerLayer.size() < 2)
    {
        throw runtime_error(strpr("ERROR: Neural network must have at least "
                                  "two layers (intput = %zu)\n.",
                                  numNeuronsPerLayer.size()));
    }
    if (numNeuronsPerLayer.size() != activationPerLayer.size())
    {
        throw runtime_error("ERROR: Number of layers is inconsistent.\n");
    }

    // Input layer has no neurons from previous layer.
    numNeuronsPerLayer.insert(numNeuronsPerLayer.begin(), 0);
    for (size_t i = 0; i < activationPerLayer.size(); ++i)
    {
        if (numNeuronsPerLayer.at(i + 1) < 1)
        {
            throw runtime_error("ERROR: Layer needs at least one neuron.\n");
        }
        layers.push_back(Layer(numNeuronsPerLayer.at(i + 1),
                               numNeuronsPerLayer.at(i),
                               activationPerLayer.at(i)));
    } 
}

NeuralNetworx::NeuralNetworx(vector<size_t> numNeuronsPerLayer,
                             vector<string> activationStringPerLayer)
{
    // Set irrelevant input layer activation to linear.
    activationStringPerLayer.at(0) = "l";
    vector<Activation> activationPerLayer;
    for (auto a : activationStringPerLayer) 
    {
        activationPerLayer.push_back(activationFromString(a));
    }

    NeuralNetworx(numNeuronsPerLayer, activationPerLayer);
}

NeuralNetworx::Layer::Layer(size_t numNeurons,
                            size_t numNeuronsPrevLayer,
                            Activation activation)
{
    this->fa = activation;
    // Input layer has no previous layer.
    if (numNeuronsPrevLayer > 0)
    {
        w.resize(numNeurons, numNeuronsPrevLayer);
        b.resize(numNeurons);
        x.resize(numNeurons);
    }
    y.resize(numNeurons);
}

vector<string> NeuralNetworx::info() const
{
    vector<string> v;
    size_t maxNeurons = 0;

    //v.push_back(strpr("Number of weights    : %6zu\n", numWeights));
    //v.push_back(strpr("Number of biases     : %6zu\n", numBiases));
    //v.push_back(strpr("Number of connections: %6zu\n", numConnections));
    //v.push_back(strpr("Architecture    "));
    v.push_back(strpr("Number of layers     : %6zu\n", layers.size()));
    for (size_t i = 0; i < layers.size(); ++i)
    {
        maxNeurons = max(static_cast<size_t>(layers.at(i).y.size()),
                         maxNeurons);
        v.push_back(strpr(" %4d", layers.at(i).y.size()));
        v.push_back(strpr("%zu\n", maxNeurons));
        v.push_back(strpr("%zu\n", static_cast<size_t>(layers.at(i).y.size())));
    }
    v.push_back("\n");
    v.push_back("-----------------------------------------"
                "--------------------------------------\n");

    for (size_t i = 0; i < maxNeurons; ++i)
    {
        v.push_back(strpr("%4d", i + 1));
        string s = "";
        for (size_t j = 0; j < layers.size(); ++j)
        {
            if (static_cast<int>(i) < layers.at(j).y.size())
            {
                if (j == 0)
                {
                    s += strpr(" %3s", "G");
                }
                else
                {
                    s += strpr(" %3s",
                               stringFromActivation(layers.at(j).fa).c_str());
                }
            }
            else
            {
                s += "    ";
            }
        }
        v.push_back(s += "\n");
    }

    return v;
}


string nnp::stringFromActivation(NeuralNetworx::Activation a)
{
    string c = "";

    if      (a == NeuralNetworx::Activation::IDENTITY)    c = "l";
    else if (a == NeuralNetworx::Activation::TANH)        c = "t";
    else if (a == NeuralNetworx::Activation::LOGISTIC)    c = "s";
    else if (a == NeuralNetworx::Activation::SOFTPLUS)    c = "p";
    else if (a == NeuralNetworx::Activation::RELU)        c = "r";
    else if (a == NeuralNetworx::Activation::GAUSSIAN)    c = "g";
    else if (a == NeuralNetworx::Activation::COS)         c = "c";
    else if (a == NeuralNetworx::Activation::REVLOGISTIC) c = "S";
    else if (a == NeuralNetworx::Activation::EXP)         c = "e";
    else if (a == NeuralNetworx::Activation::HARMONIC)    c = "h";
    else
    {
        throw runtime_error("ERROR: Unknown activation function.\n");
    }

    return c;
}

NeuralNetworx::Activation nnp::activationFromString(string c)
{
    NeuralNetworx::Activation a;

    if      (c == "l") a = NeuralNetworx::Activation::IDENTITY;
    else if (c == "t") a = NeuralNetworx::Activation::TANH;
    else if (c == "s") a = NeuralNetworx::Activation::LOGISTIC;
    else if (c == "p") a = NeuralNetworx::Activation::SOFTPLUS;
    else if (c == "r") a = NeuralNetworx::Activation::RELU;
    else if (c == "g") a = NeuralNetworx::Activation::GAUSSIAN;
    else if (c == "c") a = NeuralNetworx::Activation::COS;
    else if (c == "S") a = NeuralNetworx::Activation::REVLOGISTIC;
    else if (c == "e") a = NeuralNetworx::Activation::EXP;
    else if (c == "h") a = NeuralNetworx::Activation::HARMONIC;
    else
    {
        throw runtime_error("ERROR: Unknown activation function.\n");
    }

    return a;
}
