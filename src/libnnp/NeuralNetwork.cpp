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

#include "NeuralNetwork.h"
#include "utility.h"
#include <algorithm> // std::min, std::max
#include <cmath>     // sqrt, pow, tanh
#include <cstdio>    // fprintf, stderr
#include <cstdlib>   // exit, EXIT_FAILURE, rand, srand
#include <limits>    // std::numeric_limits

#define EXP_LIMIT 35.0

using namespace std;
using namespace nnp;

NeuralNetwork::
NeuralNetwork(int                              numLayers,
              int const* const&                numNeuronsPerLayer,
              ActivationFunction const* const& activationFunctionsPerLayer)
{
    // check number of layers
    this->numLayers = numLayers;
    if (numLayers < 3)
    {
        fprintf(stderr,
                "ERROR: Neural network must have at least three layers");
        exit(EXIT_FAILURE);
    }
    numHiddenLayers = numLayers - 2;

    // do not normalize neurons by default
    normalizeNeurons = false;

    // allocate layers and populate with neurons
    layers = new Layer[numLayers];
    inputLayer = &layers[0];
    outputLayer = &layers[numLayers-1];
    allocateLayer(*inputLayer,
                  0,
                  numNeuronsPerLayer[0],
                  activationFunctionsPerLayer[0]);
    for (int i = 1; i < numLayers; i++)
    {
        allocateLayer(layers[i],
                      numNeuronsPerLayer[i-1],
                      numNeuronsPerLayer[i],
                      activationFunctionsPerLayer[i]);
    }

    // count connections
    numWeights     = 0;
    numBiases      = 0;
    numConnections = 0;
    for (int i = 1; i < numLayers; i++)
    {
        numBiases  += layers[i].numNeurons;
        numWeights += layers[i].numNeurons * layers[i].numNeuronsPrevLayer;
    }
    numConnections = numWeights + numBiases;

    // calculate weight and bias offsets for each layer
    weightOffset = new int[numLayers-1];
    weightOffset[0] = 0;
    for (int i = 1; i < numLayers-1; i++)
    {
        weightOffset[i] = weightOffset[i-1] +
                          (layers[i-1].numNeurons + 1) * layers[i].numNeurons;
    }
    biasOffset = new int[numLayers-1];
    for (int i = 0; i < numLayers-1; i++)
    {
        biasOffset[i] = weightOffset[i] +
                        layers[i+1].numNeurons * layers[i].numNeurons;
    }
    biasOnlyOffset = new int[numLayers-1];
    biasOnlyOffset[0] = 0;
    for (int i = 1; i < numLayers-1; i++)
    {
        biasOnlyOffset[i] = biasOnlyOffset[i-1] + layers[i].numNeurons;
    }
}

NeuralNetwork::~NeuralNetwork()
{
    for (int i = 0; i < numLayers; i++)
    {
        for (int j = 0; j < layers[i].numNeurons; j++)
        {
            delete[] layers[i].neurons[j].weights;
        }
        delete[] layers[i].neurons;
    }
    delete[] layers;
    delete[] weightOffset;
    delete[] biasOffset;
    delete[] biasOnlyOffset;
}

void NeuralNetwork::setNormalizeNeurons(bool normalizeNeurons)
{
    this->normalizeNeurons = normalizeNeurons;

    return;
}

int NeuralNetwork::getNumNeurons() const
{
    int count = 0;

    for (int i = 0; i < numLayers; i++)
    {
        count += layers[i].numNeurons;
    }

    return count;
}

int NeuralNetwork::getNumConnections() const
{
    return numConnections;
}

int NeuralNetwork::getNumWeights() const
{
    return numWeights;
}

int NeuralNetwork::getNumBiases() const
{
    return numBiases;
}

void NeuralNetwork::setConnections(double const* const& connections)
{
    int count = 0;

    for (int i = 1; i < numLayers; i++)
    {
        for (int j = 0; j < layers[i].numNeuronsPrevLayer; j++)
        {
            for (int k = 0; k < layers[i].numNeurons; k++)
            {
                layers[i].neurons[k].weights[j] = connections[count];
                count++;
            }
        }
        for (int j = 0; j < layers[i].numNeurons; j++)
        {
            layers[i].neurons[j].bias = connections[count];
            count++;
        }
    }

    return;
}

void NeuralNetwork::getConnections(double* connections) const
{
    int count = 0;

    for (int i = 1; i < numLayers; i++)
    {
        for (int j = 0; j < layers[i].numNeuronsPrevLayer; j++)
        {
            for (int k = 0; k < layers[i].numNeurons; k++)
            {
                connections[count] = layers[i].neurons[k].weights[j] ;
                count++;
            }
        }
        for (int j = 0; j < layers[i].numNeurons; j++)
        {
            connections[count] = layers[i].neurons[j].bias;
            count++;
        }
    }

    return;
}

void NeuralNetwork::initializeConnectionsRandomUniform(unsigned int seed)
{
    double* connections = new double[numConnections];

    srand(seed);
    for (int i = 0; i < numConnections; i++)
    {
        connections[i] = -1.0 + 2.0 * (double)rand() / RAND_MAX;
    }

    setConnections(connections);

    delete[] connections;

    return;
}

void NeuralNetwork::modifyConnections(ModificationScheme modificationScheme)
{
    if (modificationScheme == MS_ZEROBIAS)
    {
        for (int i = 0; i < numLayers; i++)
        {
            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                layers[i].neurons[j].bias = 0.0;
            }
        }
    }
    else if (modificationScheme == MS_ZEROOUTPUTWEIGHTS)
    {
        for (int i = 0; i < outputLayer->numNeurons; i++)
        {
            for (int j = 0; j < outputLayer->numNeuronsPrevLayer; j++)
            {
                outputLayer->neurons[i].weights[j] = 0.0;
            }
        }
    }
    else if (modificationScheme == MS_FANIN)
    {
        for (int i = 1; i < numLayers; i++)
        {
            if(layers[i].activationFunction == AF_TANH)
            {
                for (int j = 0; j < layers[i].numNeurons; j++)
                {
                    for (int k = 0; k < layers[i].numNeuronsPrevLayer; k++)
                    {
                        layers[i].neurons[j].weights[k] /=
                                           sqrt(layers[i].numNeuronsPrevLayer);
                    }
                }
            }
        }
    }
    else if (modificationScheme == MS_GLOROTBENGIO)
    {
        for (int i = 1; i < numLayers; i++)
        {
            if(layers[i].activationFunction == AF_TANH)
            {
                for (int j = 0; j < layers[i].numNeurons; j++)
                {
                    for (int k = 0; k < layers[i].numNeuronsPrevLayer; k++)
                    {
                        layers[i].neurons[j].weights[k] *= sqrt(6.0 / (
                                             layers[i].numNeuronsPrevLayer
                                           + layers[i].numNeurons));
                    }
                }
            }
        }
    }
    else if (modificationScheme == MS_NGUYENWIDROW)
    {
        double beta   = 0.0;
        double sum    = 0.0;
        double weight = 0.0;

        for (int i = 1; i < numLayers-1; i++)
        {
            beta = 0.7 * pow(layers[i].numNeurons,
                             1.0 / double(layers[i].numNeuronsPrevLayer));
            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                sum = 0.0;
                for (int k = 0; k < layers[i].numNeuronsPrevLayer; k++)
                {
                    weight = layers[i].neurons[j].weights[k];
                    sum += weight * weight;
                }
                sum = sqrt(sum);
                for (int k = 0; k < layers[i].numNeuronsPrevLayer; k++)
                {
                    layers[i].neurons[j].weights[k] *= beta / sum;
                    if (layers[i].activationFunction == AF_TANH)
                    {
                        layers[i].neurons[j].weights[k] *= 2.0;
                    }
                }
                layers[i].neurons[j].bias *= beta;
                if (layers[i].activationFunction == AF_TANH)
                {
                    layers[i].neurons[j].bias *= 2.0;
                }
            }
        }
        for (int i = 0; i < outputLayer->numNeurons; i++)
        {
            outputLayer->neurons[0].weights[i] *= 0.5;
        }
    }
    else
    {
        fprintf(stderr, "ERROR: Incorrect modifyConnections call.\n");
        exit(EXIT_FAILURE);
    }

    return;
}

void NeuralNetwork::modifyConnections(ModificationScheme modificationScheme,
                                      double parameter1,
                                      double parameter2)
{
    if (modificationScheme == MS_PRECONDITIONOUTPUT)
    {
        double mean  = parameter1;
        double sigma = parameter2;

        for (int i = 0; i < outputLayer->numNeurons; i++)
        {
            for (int j = 0; j < outputLayer->numNeuronsPrevLayer; j++)
            {
                outputLayer->neurons[i].weights[j] *= sigma;
            }
            outputLayer->neurons[i].bias += mean;
        }
    }
    else
    {
        fprintf(stderr, "ERROR: Incorrect modifyConnections call.\n");
        exit(EXIT_FAILURE);
    }

    return;
}

void NeuralNetwork::setInput(size_t const index, double const value) const
{
    Neuron& n = inputLayer->neurons[index];
    n.count++;
    n.value = value;
    n.min  = min(value, n.min);
    n.max  = max(value, n.max);
    n.sum  += value;
    n.sum2 += value * value;

    return;
}

void NeuralNetwork::setInput(double const* const& input) const
{
    for (int i = 0; i < inputLayer->numNeurons; i++)
    {
        double const& value = input[i];
        Neuron& n = inputLayer->neurons[i];
        n.count++;
        n.value = value;
        n.min  = min(value, n.min);
        n.max  = max(value, n.max);
        n.sum  += value;
        n.sum2 += value * value;
    }

    return;
}

void NeuralNetwork::getOutput(double* output) const
{
    for (int i = 0; i < outputLayer->numNeurons; i++)
    {
        output[i] = outputLayer->neurons[i].value;
    }

    return;
}

void NeuralNetwork::propagate()
{
    for (int i = 1; i < numLayers; i++)
    {
        propagateLayer(layers[i], layers[i-1]);
    }

    return;
}

void NeuralNetwork::calculateDEdG(double *dEdG) const
{
    double** inner = new double*[numHiddenLayers];
    double** outer = new double*[numHiddenLayers];

    for (int i = 0; i < numHiddenLayers; i++)
    {
        inner[i] = new double[layers[i+1].numNeurons];
        outer[i] = new double[layers[i+2].numNeurons];
    }

    for (int k = 0; k < layers[0].numNeurons; k++)
    {
        for (int i = 0; i < layers[1].numNeurons; i++)
        {
            inner[0][i] = layers[1].neurons[i].weights[k]
                        * layers[1].neurons[i].dfdx;
            if (normalizeNeurons) inner[0][i] /= layers[0].numNeurons;
        }
        for (int l = 1; l < numHiddenLayers+1; l++)
        {
            for (int i2 = 0; i2 < layers[l+1].numNeurons; i2++)
            {
                outer[l-1][i2] = 0.0;
                for (int i1 = 0; i1 < layers[l].numNeurons; i1++)
                {
                    outer[l-1][i2] += layers[l+1].neurons[i2].weights[i1]
                                    * inner[l-1][i1];
                }
                outer[l-1][i2] *= layers[l+1].neurons[i2].dfdx;
                if (normalizeNeurons) outer[l-1][i2] /= layers[l].numNeurons;
                if (l < numHiddenLayers) inner[l][i2] = outer[l-1][i2];
            }
        }
        dEdG[k] = outer[numHiddenLayers-1][0];
    }

    for (int i = 0; i < numHiddenLayers; i++)
    {
        delete[] inner[i];
        delete[] outer[i];
    }
    delete[] inner;
    delete[] outer;

    return;
}

void NeuralNetwork::calculateDEdc(double* dEdc) const
{
    int count = 0;

    for (int i = 0; i < numConnections; i++)
    {
        dEdc[i] = 0.0;
    }

    for (int i = 0; i < outputLayer->numNeurons; i++)
    {
        dEdc[biasOffset[numLayers-2]+i] = outputLayer->neurons[i].dfdx;
        if (normalizeNeurons)
        {
            dEdc[biasOffset[numLayers-2]+i] /=
                outputLayer->numNeuronsPrevLayer;
        }
    }

    for (int i = numLayers-2; i >= 0; i--)
    {
        count = 0;
        for (int j = 0; j < layers[i].numNeurons; j++)
        {
            for (int k = 0; k < layers[i+1].numNeurons; k++)
            {
                dEdc[weightOffset[i]+count] = dEdc[biasOffset[i]+k]
                                            * layers[i].neurons[j].value;
                count++;
                if (i >= 1)
                {
                    dEdc[biasOffset[i-1]+j] += dEdc[biasOffset[i]+k]
                        * layers[i+1].neurons[k].weights[j]
                        * layers[i].neurons[j].dfdx;
                }
            }
            if (normalizeNeurons && i >= 1)
            {
                dEdc[biasOffset[i-1]+j] /= layers[i].numNeuronsPrevLayer;
            }
        }
    }

    return;
}

void NeuralNetwork::calculateDFdc(double*              dFdc,
                                  double const* const& dGdxyz) const
{
    double* dEdb    = new double[numBiases];
    double* d2EdGdc = new double[numConnections];

    for (int i = 0; i < numBiases; i++)
    {
        dEdb[i] = 0.0;
    }
    for (int i = 0; i < numConnections; i++)
    {
        dFdc[i]    = 0.0;
        d2EdGdc[i] = 0.0;
    }

    calculateDEdb(dEdb);
    for (int i = 0; i < layers[0].numNeurons; i++)
    {
        for (int j = 0; j < numConnections; j++)
        {
            d2EdGdc[j] = 0.0;
        }
        calculateDxdG(i);
        calculateD2EdGdc(i, dEdb, d2EdGdc);
        for (int j = 0; j < numConnections; j++)
        {
            // Note: F = - dE / dx !!
            //           ^
            dFdc[j] -= d2EdGdc[j] * dGdxyz[i];
        }
    }

    delete[] dEdb;
    delete[] d2EdGdc;

    return;
}

void NeuralNetwork::writeConnections(std::ofstream& file) const
{
    // File header.
    vector<string> title;
    vector<string> colName;
    vector<string> colInfo;
    vector<size_t> colSize;
    title.push_back("Neural network connection values (weights and biases).");
    colSize.push_back(24);
    colName.push_back("connection");
    colInfo.push_back("Neural network connection value.");
    colSize.push_back(1);
    colName.push_back("t");
    colInfo.push_back("Connection type (a = weight, b = bias).");
    colSize.push_back(9);
    colName.push_back("index");
    colInfo.push_back("Index enumerating weights.");
    colSize.push_back(5);
    colName.push_back("l_s");
    colInfo.push_back("Starting point layer (end point layer for biases).");
    colSize.push_back(5);
    colName.push_back("n_s");
    colInfo.push_back("Starting point neuron in starting layer (end point "
                      "neuron for biases).");
    colSize.push_back(5);
    colName.push_back("l_e");
    colInfo.push_back("End point layer.");
    colSize.push_back(5);
    colName.push_back("n_e");
    colInfo.push_back("End point neuron in end layer.");
    appendLinesToFile(file,
                      createFileHeader(title, colSize, colName, colInfo));

    int count = 0;
    for (int i = 1; i < numLayers; i++)
    {
        for (int j = 0; j < layers[i].numNeuronsPrevLayer; j++)
        {
            for (int k = 0; k < layers[i].numNeurons; k++)
            {
                count++;
                file << strpr("%24.16E a %9d %5d %5d %5d %5d\n",
                              layers[i].neurons[k].weights[j],
                              count,
                              i - 1,
                              j + 1,
                              i,
                              k + 1);
            }
        }
        for (int j = 0; j < layers[i].numNeurons; j++)
        {
            count++;
            file << strpr("%24.16E b %9d %5d %5d\n",
                          layers[i].neurons[j].bias,
                          count,
                          i,
                          j + 1);
        }
    }

    return;
}

void NeuralNetwork::calculateDEdb(double* dEdb) const
{
    for (int i = 0; i < outputLayer->numNeurons; i++)
    {
        dEdb[biasOnlyOffset[numLayers-2]+i] = outputLayer->neurons[i].dfdx;
        if (normalizeNeurons)
        {
            dEdb[biasOnlyOffset[numLayers-2]+i] /=
                outputLayer->numNeuronsPrevLayer;
        }
    }

    for (int i = numLayers-2; i >= 0; i--)
    {
        for (int j = 0; j < layers[i].numNeurons; j++)
        {
            for (int k = 0; k < layers[i+1].numNeurons; k++)
            {
                if (i >= 1)
                {
                    dEdb[biasOnlyOffset[i-1]+j] += dEdb[biasOnlyOffset[i]+k]
                        * layers[i+1].neurons[k].weights[j]
                        * layers[i].neurons[j].dfdx;
                }
            }
            if (normalizeNeurons && i >= 1)
            {
                dEdb[biasOnlyOffset[i-1]+j] /= layers[i].numNeuronsPrevLayer;
            }
        }
    }

    return;
}

void NeuralNetwork::calculateDxdG(int index) const
{
    for (int i = 0; i < layers[1].numNeurons; i++)
    {
        layers[1].neurons[i].dxdG = layers[1].neurons[i].weights[index];
        if (normalizeNeurons)
        {
            layers[1].neurons[i].dxdG /= layers[1].numNeuronsPrevLayer;
        }
    }
    for (int i = 2; i < numLayers; i++)
    {
        for (int j = 0; j < layers[i].numNeurons; j++)
        {
            layers[i].neurons[j].dxdG = 0.0;
            for (int k = 0; k < layers[i-1].numNeurons; k++)
            {
                layers[i].neurons[j].dxdG += layers[i].neurons[j].weights[k]
                                           * layers[i-1].neurons[k].dfdx
                                           * layers[i-1].neurons[k].dxdG;
            }
            if (normalizeNeurons)
            {
                layers[i].neurons[j].dxdG /= layers[i].numNeuronsPrevLayer;
            }
        }
    }

    return;
}

void NeuralNetwork::calculateD2EdGdc(int                  index,
                                     double const* const& dEdb,
                                     double*              d2EdGdc) const
{
    int count = 0;

    for (int i = 0; i < outputLayer->numNeurons; i++)
    {
        d2EdGdc[biasOffset[numLayers-2]+i] = outputLayer->neurons[i].d2fdx2
                                           * outputLayer->neurons[i].dxdG;
        if (normalizeNeurons)
        {
            d2EdGdc[biasOffset[numLayers-2]+i] /=
                outputLayer->numNeuronsPrevLayer;
        }
    }

    for (int i = numLayers-2; i >= 0; i--)
    {
        count = 0;
        for (int j = 0; j < layers[i].numNeurons; j++)
        {
            for (int k = 0; k < layers[i+1].numNeurons; k++)
            {
                if (i == 0)
                {
                    d2EdGdc[weightOffset[i]+count] =
                        d2EdGdc[biasOffset[i]+k] * layers[i].neurons[j].value;
                    if (j == index)
                    {
                        d2EdGdc[weightOffset[i]+count] +=
                            dEdb[biasOnlyOffset[i]+k];
                    }
                }
                else
                {
                    d2EdGdc[weightOffset[i]+count] =
                        d2EdGdc[biasOffset[i]+k] * layers[i].neurons[j].value
                      + dEdb[biasOnlyOffset[i]+k] * layers[i].neurons[j].dfdx
                      * layers[i].neurons[j].dxdG;
                }
                count++;
                if (i >= 1)
                {
                    d2EdGdc[biasOffset[i-1]+j] +=
                        layers[i+1].neurons[k].weights[j]
                      * (d2EdGdc[biasOffset[i]+k] * layers[i].neurons[j].dfdx
                      + dEdb[biasOnlyOffset[i]+k]
                      * layers[i].neurons[j].d2fdx2
                      * layers[i].neurons[j].dxdG);
                }
            }
            if (normalizeNeurons && i >= 1)
            {
                d2EdGdc[biasOffset[i-1]+j] /= layers[i].numNeuronsPrevLayer;
            }
        }
    }

    return;
}

void NeuralNetwork::allocateLayer(Layer&             layer,
                                  int                numNeuronsPrevLayer,
                                  int                numNeurons,
                                  ActivationFunction activationFunction)
{
    layer.numNeurons          = numNeurons;
    layer.numNeuronsPrevLayer = numNeuronsPrevLayer;
    layer.activationFunction  = activationFunction;

    layer.neurons = new Neuron[layer.numNeurons];
    for (int i = 0; i < layer.numNeurons; i++)
    {
        layer.neurons[i].x      = 0.0;
        layer.neurons[i].value  = 0.0;
        layer.neurons[i].dfdx   = 0.0;
        layer.neurons[i].d2fdx2 = 0.0;
        layer.neurons[i].bias   = 0.0;
        layer.neurons[i].dxdG   = 0.0;
        layer.neurons[i].count  = 0;
        layer.neurons[i].min    =  numeric_limits<double>::max();
        layer.neurons[i].max    = -numeric_limits<double>::max();
        layer.neurons[i].sum    = 0.0;
        layer.neurons[i].sum2   = 0.0;
        if (layer.numNeuronsPrevLayer > 0)
        {
            layer.neurons[i].weights = new double[layer.numNeuronsPrevLayer];
            for (int j = 0; j < layer.numNeuronsPrevLayer; j++)
            {
                layer.neurons[i].weights[j] = 0.0;
            }
        }
        else
        {
            layer.neurons[i].weights = 0;
        }
    }

    return;
}

void NeuralNetwork::propagateLayer(Layer& layer, Layer& layerPrev)
{
    double dtmp = 0.0;

    for (int i = 0; i < layer.numNeurons; i++)
    {
        dtmp = 0.0;
        for (int j = 0; j < layer.numNeuronsPrevLayer; j++)
        {
            dtmp += layer.neurons[i].weights[j] * layerPrev.neurons[j].value;
        }
        dtmp += layer.neurons[i].bias;
        if (normalizeNeurons)
        {
            dtmp /= layer.numNeuronsPrevLayer;
        }

        layer.neurons[i].x = dtmp;
        if (layer.activationFunction == AF_IDENTITY)
        {
            layer.neurons[i].value  = dtmp;
            layer.neurons[i].dfdx   = 1.0;
            layer.neurons[i].d2fdx2 = 0.0;
        }
        else if (layer.activationFunction == AF_TANH)
        {
            dtmp = tanh(dtmp);
            layer.neurons[i].value  = dtmp;
            layer.neurons[i].dfdx   = 1.0 - dtmp * dtmp;
            layer.neurons[i].d2fdx2 = -2.0 * dtmp * (1.0 - dtmp * dtmp);
        }
        else if (layer.activationFunction == AF_LOGISTIC)
        {
            if (dtmp > EXP_LIMIT)
            {
                layer.neurons[i].value  = 1.0;
                layer.neurons[i].dfdx   = 0.0;
                layer.neurons[i].d2fdx2 = 0.0;
            }
            else if (dtmp < -EXP_LIMIT)
            {
                layer.neurons[i].value  = 0.0;
                layer.neurons[i].dfdx   = 0.0;
                layer.neurons[i].d2fdx2 = 0.0;
            }
            else
            {
                dtmp = 1.0 / (1.0 + exp(-dtmp));
                layer.neurons[i].value  = dtmp;
                layer.neurons[i].dfdx   = dtmp * (1.0 - dtmp);
                layer.neurons[i].d2fdx2 = dtmp * (1.0 - dtmp)
                                        * (1.0 - 2.0 * dtmp);
            }
        }
        else if (layer.activationFunction == AF_SOFTPLUS)
        {
            if (dtmp > EXP_LIMIT)
            {
                layer.neurons[i].value  = dtmp;
                layer.neurons[i].dfdx   = 1.0;
                layer.neurons[i].d2fdx2 = 0.0;
            }
            else if (dtmp < -EXP_LIMIT)
            {
                layer.neurons[i].value  = 0.0;
                layer.neurons[i].dfdx   = 0.0;
                layer.neurons[i].d2fdx2 = 0.0;
            }
            else
            {
                dtmp = exp(dtmp);
                layer.neurons[i].value  = log(1.0 + dtmp);
                dtmp = 1.0 / (1.0 + 1.0 / dtmp);
                layer.neurons[i].dfdx   = dtmp;
                layer.neurons[i].d2fdx2 = dtmp * (1.0 - dtmp);
            }
        }
        else if (layer.activationFunction == AF_RELU)
        {
            if (dtmp > 0.0)
            {
                layer.neurons[i].value  = dtmp;
                layer.neurons[i].dfdx   = 1.0;
                layer.neurons[i].d2fdx2 = 0.0;
            }
            else
            {
                layer.neurons[i].value  = 0.0;
                layer.neurons[i].dfdx   = 0.0;
                layer.neurons[i].d2fdx2 = 0.0;
            }
        }
        else if (layer.activationFunction == AF_GAUSSIAN)
        {
            double const tmpexp = exp(-0.5 * dtmp * dtmp);
            layer.neurons[i].value  = tmpexp;
            layer.neurons[i].dfdx   = -dtmp * tmpexp;
            layer.neurons[i].d2fdx2 = (dtmp * dtmp - 1.0) * tmpexp;
        }
        else if (layer.activationFunction == AF_COS)
        {
            double const tmpcos = cos(dtmp);
            layer.neurons[i].value  = tmpcos;
            layer.neurons[i].dfdx   = -sin(dtmp);
            layer.neurons[i].d2fdx2 = -tmpcos;
        }
        else if (layer.activationFunction == AF_REVLOGISTIC)
        {
            dtmp = 1.0 / (1.0 + exp(-dtmp));
            layer.neurons[i].value  = 1.0 - dtmp;
            layer.neurons[i].dfdx   = dtmp * (dtmp - 1.0);
            layer.neurons[i].d2fdx2 = dtmp * (dtmp - 1.0) * (1.0 - 2.0 * dtmp);
        }
        else if (layer.activationFunction == AF_EXP)
        {
            dtmp = exp(-dtmp);
            layer.neurons[i].value  = dtmp;
            layer.neurons[i].dfdx   = -dtmp;
            layer.neurons[i].d2fdx2 = dtmp;
        }
        else if (layer.activationFunction == AF_HARMONIC)
        {
            layer.neurons[i].value  = dtmp * dtmp;
            layer.neurons[i].dfdx   = 2.0 * dtmp;
            layer.neurons[i].d2fdx2 = 2.0;
        }
        layer.neurons[i].count++;
        dtmp = layer.neurons[i].x;
        layer.neurons[i].min  = min(dtmp, layer.neurons[i].min);
        layer.neurons[i].max  = max(dtmp, layer.neurons[i].max);
        layer.neurons[i].sum  += dtmp;
        layer.neurons[i].sum2 += dtmp * dtmp;
    }

    return;
}

void NeuralNetwork::resetNeuronStatistics()
{
    for (int i = 0; i < numLayers; i++)
    {
        for (int j = 0; j < layers[i].numNeurons; j++)
        {
            layers[i].neurons[j].count = 0;
            layers[i].neurons[j].min   =  numeric_limits<double>::max();
            layers[i].neurons[j].max   = -numeric_limits<double>::max();
            layers[i].neurons[j].sum   = 0.0;
            layers[i].neurons[j].sum2  = 0.0;
        }
    }

    return;
}

void NeuralNetwork::getNeuronStatistics(long*   count,
                                        double* min,
                                        double* max,
                                        double* sum,
                                        double* sum2) const
{
    int iNeuron = 0;

    for (int i = 0; i < numLayers; i++)
    {
        for (int j = 0; j < layers[i].numNeurons; j++)
        {
            count[iNeuron] = layers[i].neurons[j].count;
            min  [iNeuron] = layers[i].neurons[j].min;
            max  [iNeuron] = layers[i].neurons[j].max;
            sum  [iNeuron] = layers[i].neurons[j].sum;
            sum2 [iNeuron] = layers[i].neurons[j].sum2;
            iNeuron++;
        }
    }

    return;
}

/*
void NeuralNetwork::writeStatus(int element, int epoch)
{
    char  fName[LSTR] = "";
    FILE* fpn         = NULL;
    FILE* fpw         = NULL;

    for (int i = 0; i < numLayers; i++)
    {
        sprintf(fName, "nn.neurons.%03d.%1d.%06d", element, i, epoch);
        fpn = fopen(fName, "a");
        if (fpn == NULL)
        {
            fprintf(stderr, "ERROR: Could not open file: %s.\n", fName);
            exit(EXIT_FAILURE);
        }
        sprintf(fName, "nn.weights.%03d.%1d.%06d", element, i, epoch);
        fpw = fopen(fName, "a");
        if (fpw == NULL)
        {
            fprintf(stderr, "ERROR: Could not open file: %s.\n", fName);
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < layers[i].numNeurons; j++)
        {
            fprintf(fpn, "%4d %.8f %.8f %.8f %.8f %.8f %.8f\n", j, layers[i].neurons[j].x,
                    layers[i].neurons[j].value, layers[i].neurons[j].dfdx, layers[i].neurons[j].d2fdx2,
                    layers[i].neurons[j].bias, layers[i].neurons[j].dxdG);
            for (int k = 0; k < layers[i].numNeuronsPrevLayer; k++)
            {
                fprintf(fpw, "%4d %4d %.8f\n", j, k, layers[i].neurons[j].weights[k]);
            }
        }
        fclose(fpn);
        fclose(fpw);
    }

    return;

}
*/

long NeuralNetwork::getMemoryUsage()
{
    long mem        = sizeof(*this);
    int  numNeurons = getNumNeurons();

    mem += (numLayers - 1) * sizeof(int); // weightOffset
    mem += (numLayers - 1) * sizeof(int); // biasOffset
    mem += (numLayers - 1) * sizeof(int); // biasOnlyOffset
    mem += numLayers  * sizeof(Layer);    // layers
    mem += numNeurons * sizeof(Neuron);   // neurons
    mem += numWeights * sizeof(double);   // weights

    return mem;
}

vector<string> NeuralNetwork::info() const
{
    vector<string> v;
    int maxNeurons = 0;

    v.push_back(strpr("Number of weights    : %6zu\n", numWeights));
    v.push_back(strpr("Number of biases     : %6zu\n", numBiases));
    v.push_back(strpr("Number of connections: %6zu\n", numConnections));
    v.push_back(strpr("Architecture    "));
    for (int i = 0; i < numLayers; ++i)
    {
        maxNeurons = max(layers[i].numNeurons, maxNeurons);
        v.push_back(strpr(" %4d", layers[i].numNeurons));
    }
    v.push_back("\n");
    v.push_back("-----------------------------------------"
                "--------------------------------------\n");

    for (int i = 0; i < maxNeurons; ++i)
    {
        v.push_back(strpr("%4d", i + 1));
        string s = "";
        for (int j = 0; j < numLayers; ++j)
        {
            if (i < layers[j].numNeurons)
            {
                if (j == 0)
                {
                    s += strpr(" %3s", "G");
                }
                else if (layers[j].activationFunction == AF_IDENTITY)
                {
                    s += strpr(" %3s", "l");
                }
                else if (layers[j].activationFunction == AF_TANH)
                {
                    s += strpr(" %3s", "t");
                }
                else if (layers[j].activationFunction == AF_LOGISTIC)
                {
                    s += strpr(" %3s", "s");
                }
                else if (layers[j].activationFunction == AF_SOFTPLUS)
                {
                    s += strpr(" %3s", "p");
                }
                else if (layers[j].activationFunction == AF_RELU)
                {
                    s += strpr(" %3s", "r");
                }
                else if (layers[j].activationFunction == AF_GAUSSIAN)
                {
                    s += strpr(" %3s", "g");
                }
                else if (layers[j].activationFunction == AF_COS)
                {
                    s += strpr(" %3s", "c");
                }
                else if (layers[j].activationFunction == AF_REVLOGISTIC)
                {
                    s += strpr(" %3s", "S");
                }
                else if (layers[j].activationFunction == AF_EXP)
                {
                    s += strpr(" %3s", "e");
                }
                else if (layers[j].activationFunction == AF_HARMONIC)
                {
                    s += strpr(" %3s", "h");
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
