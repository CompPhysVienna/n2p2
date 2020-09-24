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
using namespace Eigen;
using namespace nnp;

NeuralNetworx::NeuralNetworx(vector<size_t>     numNeuronsPerLayer,
                             vector<Activation> activationPerLayer) :
    numLayers      (0),
    numConnections (0),
    numWeights     (0),
    numBiases      (0)
{
    initialize(numNeuronsPerLayer, activationPerLayer);
}

NeuralNetworx::NeuralNetworx(vector<size_t> numNeuronsPerLayer,
                             vector<string> activationStringPerLayer) :
    numLayers      (0),
    numConnections (0),
    numWeights     (0),
    numBiases      (0)
{
    // Set irrelevant input layer activation to linear.
    activationStringPerLayer.at(0) = "l";
    vector<Activation> activationPerLayer;
    for (auto a : activationStringPerLayer) 
    {
        activationPerLayer.push_back(activationFromString(a));
    }

    initialize(numNeuronsPerLayer, activationPerLayer);
}

NeuralNetworx::Layer::Layer(size_t numNeurons,
                            size_t numNeuronsPrevLayer,
                            Activation activation) :
    numNeurons         (numNeurons),
    numNeuronsPrevLayer(numNeuronsPrevLayer),
    fa                 (activation)
{
    // Input layer has no previous layer.
    if (numNeuronsPrevLayer > 0)
    {
        w.resize(numNeurons, numNeuronsPrevLayer);
        b.resize(numNeurons);
        x.resize(numNeurons);
        dydx.resize(numNeurons);
        d2ydx2.resize(numNeurons);
    }
    y.resize(numNeurons);
}

void NeuralNetworx::Layer::propagate(VectorXd const& yp)
{
    x = w * yp + b;

    if (fa == Activation::IDENTITY)
    {
        y = x;
        dydx.setOnes();
        d2ydx2.setZero();
    }
    else if (fa == Activation::TANH)
    {
        y = x.array().tanh();
        dydx = 1.0 - y.array() * y.array();
        d2ydx2 = -2.0 * y.array() * dydx.array();
    }
    else if (fa == Activation::LOGISTIC)
    {
        y = x.array().unaryExpr(
            [](double xi)
            {
                if      (xi >  EXP_LIMIT) return 1.0;
                else if (xi < -EXP_LIMIT) return 0.0;
                else return 1.0 / (1.0 + exp(-xi));
            });
        //dydx = (y.array() <= 0.0 || y.array() >= 1.0).select(
        //    0.0,
        //    y.array() * (1.0 - y.array()));
        dydx = y.array().unaryExpr(
            [](double yi)
            {
                if (yi <= 0.0 || yi >= 1.0) return 0.0;
                else return yi * (1.0 - yi);
            });
        //d2ydx2 = (y.array() <= 0.0 || y.array() >= 1.0).select(
        //    0.0,
        //    y.array() * (1.0 - y.array()) * (1.0 - 2.0 * y.array()));
        d2ydx2 = y.array().unaryExpr(
            [](double yi)
            {
                if (yi <= 0.0 || yi >= 1.0) return 0.0;
                else return yi * (1.0 - yi) * (1.0 - 2.0 * yi);
                ;
            });
    }
    else if (fa == Activation::SOFTPLUS)
    {
        y = x.array().unaryExpr(
            [](double xi)
            {
                if      (xi >  EXP_LIMIT) return xi;
                else if (xi < -EXP_LIMIT) return 0.0;
                else return log(1.0 + exp(xi));
            });
        dydx = x.array().unaryExpr(
            [](double xi)
            {
                if      (xi >  EXP_LIMIT) return 1.0;
                else if (xi < -EXP_LIMIT) return 0.0;
                else return 1.0 / (1.0 + exp(-xi));
            });
        d2ydx2 = x.array().unaryExpr(
            [](double xi)
            {
                if      (xi >  EXP_LIMIT) return 0.0;
                else if (xi < -EXP_LIMIT) return 0.0;
                else
                {
                    double const tmp = 1.0 / (1.0 + exp(-xi));
                    return tmp * (1.0 - tmp);
                }
                ;
            });
    }
    else if (fa == Activation::RELU)
    {
        //y = (x.array() > 0.0).select(x, 0.0);
        y = x.array().unaryExpr(
            [](double xi)
            {
                if (xi > 0.0) return xi;
                else return 0.0;
            });
        //dydx = (x.array() > 0.0).select(VectorXd::Ones(dydx.size()) 0.0);
        dydx = x.array().unaryExpr(
            [](double xi)
            {
                if (xi > 0.0) return 1.0;
                else return 0.0;
            });
        d2ydx2.setZero();
    }
    else if (fa == Activation::GAUSSIAN)
    {
        y = (-0.5 * x.array() * x.array()).exp();
        dydx = -x.array() * y.array();
        d2ydx2 = (x.array() * x.array() - 1.0) * y.array();
    }
    else if (fa == Activation::COS)
    {
        y = x.array().cos();
        dydx = x.array().sin();
        d2ydx2 = -y.array();
    }
    else if (fa == Activation::REVLOGISTIC)
    {
        VectorXd const tmp = 1.0 / (1.0 + (-x.array()).exp());
        y = 1.0 - tmp.array();
        dydx = tmp.array() * (tmp.array() - 1.0);
        d2ydx2 = tmp.array() * (tmp.array() - 1.0) * (1.0 - 2.0 * tmp.array());
    }
    else if (fa == Activation::EXP)
    {
        y = (-x.array()).exp();
        dydx = -y.array();
        d2ydx2 = y.array();
    }
    else if (fa == Activation::HARMONIC)
    {
        y = x.array() * x.array();
        dydx = 2.0 * x.array();
        d2ydx2.setConstant(2.0);
    }

    return;
}

void NeuralNetworx::Layer::initializeDerivInput()
{
    // Multiply each row of weights (i.e. all weights connected to a neuron
    // of this layer) with derivative of the activation function.
    dydyi = dydx.asDiagonal() * w;

    return;
}

void NeuralNetworx::Layer::propagateDerivInput(MatrixXd const& dypdyi)
{
    dydyi = dydx.asDiagonal() * (w * dypdyi);

    return;
}

void NeuralNetworx::setInput(vector<double> const& input)
{
    if (input.size() != layers.at(0).numNeurons)
    {
        throw runtime_error("ERROR: Input vector does not match number of "
                            "input layer neurons.");
    }
    Map<VectorXd const> y0(input.data(), input.size());
    layers.at(0).y = y0;

    return;
}

void NeuralNetworx::propagate(bool deriv)
{
    for (size_t i = 1; i < layers.size(); ++i)
    {
        layers.at(i).propagate(layers.at(i - 1).y);
        if (deriv)
        {
            if (i == 1) layers.at(i).initializeDerivInput();
            else layers.at(i).propagateDerivInput(layers.at(i - 1).dydyi);
        }
    }

    return;
}

void NeuralNetworx::propagate(vector<double> const& input, bool deriv)
{
    setInput(input);
    propagate(deriv);

    return;
}

void NeuralNetworx::getOutput(vector<double>& output) const
{
    output.clear();
    output.insert(output.begin(),
                  layers.back().y.data(),
                  layers.back().y.data() + layers.back().y.size());

    return;
}

void NeuralNetworx::getDerivInput(vector<vector<double>>& derivInput) const
{
    derivInput.clear();
    derivInput.resize(layers.back().dydyi.rows());
    for (size_t i = 0; i < derivInput.size(); ++i)
    {
        for (size_t j = 0; j < layers.at(0).numNeurons; ++j)
        {
            derivInput.at(i).push_back(layers.back().dydyi(i, j));
        }
    }

    return;
}

void NeuralNetworx::setConnections(vector<double> const& connections)
{
    if (connections.size() != numConnections)
    {
        throw runtime_error("ERROR: Provided vector has incorrect length.\n");
    }

    Map<VectorXd const> c(connections.data(), connections.size());

    size_t count = 0;
    for (vector<Layer>::iterator l = layers.begin() + 1;
         l != layers.end(); ++l)
    {
        size_t const n = l->w.cols();
        for (Index i = 0; i < l->y.size(); ++i)
        {
            l->w.row(i) = c.segment(count, n);
            count += n;
            l->b(i) = c(count);
            count++;
        }
    }

    return;
}

void NeuralNetworx::setConnectionsAO(vector<double> const& connections)
{
    if (connections.size() != numConnections)
    {
        throw runtime_error("ERROR: Provided vector has incorrect length.\n");
    }

    Map<VectorXd const> c(connections.data(), connections.size());

    size_t count = 0;
    for (vector<Layer>::iterator l = layers.begin() + 1;
         l != layers.end(); ++l)
    {
        size_t const n = l->w.rows();
        for (Index i = 0; i < l->w.cols(); ++i)
        {
            l->w.col(i) = c.segment(count, n);
            count += n;
        }
        l->b = c.segment(count, l->b.size());
        count += l->b.size();
    }

    return;
}

void NeuralNetworx::getConnections(vector<double>& connections) const
{
    if (connections.size() != numConnections)
    {
        throw runtime_error("ERROR: Provided vector has incorrect length.\n");
    }

    Map<VectorXd> c(connections.data(), connections.size());

    size_t count = 0;
    for (vector<Layer>::const_iterator l = layers.begin() + 1;
         l != layers.end(); ++l)
    {
        size_t const n = l->w.cols();
        for (Index i = 0; i < l->y.size(); ++i)
        {
            c.segment(count, n) = l->w.row(i);
            count += n;
            c(count) = l->b(i);
            count++;
        }
    }

    return;
}

void NeuralNetworx::getConnectionsAO(vector<double>& connections) const
{
    if (connections.size() != numConnections)
    {
        throw runtime_error("ERROR: Provided vector has incorrect length.\n");
    }

    Map<VectorXd> c(connections.data(), connections.size());

    size_t count = 0;
    for (vector<Layer>::const_iterator l = layers.begin() + 1;
         l != layers.end(); ++l)
    {
        size_t const n = l->w.rows();
        for (Index i = 0; i < l->w.cols(); ++i)
        {
            c.segment(count, n) = l->w.col(i);
            count += n;
        }
        c.segment(count, l->b.size()) = l->b;
        count += l->b.size();
    }

    return;
}

void NeuralNetworx::writeConnections(std::ofstream& file) const
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

    size_t count = 0;
    for (size_t i = 1; i < layers.size(); ++i)
    {
        Layer const& l = layers.at(i);
        for (size_t j = 0; j < l.numNeurons; j++)
        {
            for (size_t k = 0; k < l.numNeuronsPrevLayer; k++)
            {
                count++;
                file << strpr("%24.16E a %9d %5d %5d %5d %5d\n",
                              l.w(j, k),
                              count,
                              i - 1,
                              k + 1,
                              i,
                              j + 1);
            }
            count++;
            file << strpr("%24.16E b %9d %5d %5d\n",
                          l.b(j),
                          count,
                          i,
                          j + 1);
        }
    }

    return;
}

void NeuralNetworx::writeConnectionsAO(std::ofstream& file) const
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

    size_t count = 0;
    for (size_t i = 1; i < layers.size(); ++i)
    {
        Layer const& l = layers.at(i);
        for (size_t j = 0; j < l.numNeuronsPrevLayer; j++)
        {
            for (size_t k = 0; k < l.numNeurons; k++)
            {
                count++;
                file << strpr("%24.16E a %9d %5d %5d %5d %5d\n",
                              l.w(k, j),
                              count,
                              i - 1,
                              j + 1,
                              i,
                              k + 1);
            }
        }
        for (size_t j = 0; j < l.numNeurons; j++)
        {
            count++;
            file << strpr("%24.16E b %9d %5d %5d\n",
                          l.b(j),
                          count,
                          i,
                          j + 1);
        }
    }

    return;
}

map<string, double> NeuralNetworx::getNeuronProperties(size_t layer,
                                                       size_t neuron) const
{
    map<string, double> properties;

    properties["y"] = layers.at(layer).y(neuron);
    if (layers.at(layer).numNeuronsPrevLayer > 0)
    {
        properties["x"]      = layers.at(layer).x(neuron);
        properties["dydx"]   = layers.at(layer).dydx(neuron);
        properties["d2ydx2"] = layers.at(layer).d2ydx2(neuron);
    }

    return properties;
}

vector<string> NeuralNetworx::info() const
{
    vector<string> v;
    Eigen::Index maxNeurons = 0;

    v.push_back(strpr("Number of layers     : %6zu\n", layers.size()));
    v.push_back(strpr("Number of connections: %6zu\n", numConnections));
    v.push_back(strpr("Number of weights    : %6zu\n", numWeights));
    v.push_back(strpr("Number of biases     : %6zu\n", numBiases));
    v.push_back(strpr("Architecture    "));
    for (size_t i = 0; i < layers.size(); ++i)
    {
        maxNeurons = max(layers.at(i).y.size(),
                         maxNeurons);
    }
    v.push_back("\n");
    v.push_back("-----------------------------------------"
                "--------------------------------------\n");

    for (size_t i = 0; i < static_cast<size_t>(maxNeurons); ++i)
    {
        v.push_back(strpr("%4d", i + 1));
        string s = "";
        for (size_t j = 0; j < layers.size(); ++j)
        {
            if (i < static_cast<size_t>(layers.at(j).y.size()))
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

void NeuralNetworx::initialize(vector<size_t>     numNeuronsPerLayer,
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
    numLayers = 0;
    layers.clear();
    for (size_t i = 0; i < activationPerLayer.size(); ++i)
    {
        if (numNeuronsPerLayer.at(i + 1) < 1)
        {
            throw runtime_error("ERROR: Layer needs at least one neuron.\n");
        }
        layers.push_back(Layer(numNeuronsPerLayer.at(i + 1),
                               numNeuronsPerLayer.at(i),
                               activationPerLayer.at(i)));
        numLayers++;
    }

    // Compute number of connections, weights and biases.
    numConnections = 0;
    numWeights     = 0;
    numBiases      = 0;
    for (auto const& l : layers)
    {
        numConnections += l.w.size() + l.b.size();
        numWeights     += l.w.size();
        numBiases      += l.b.size();
    }

    return;
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
