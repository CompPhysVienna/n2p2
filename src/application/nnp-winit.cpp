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

#include "Log.h"
#include "NeuralNetwork.h"
#include "utility.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <gsl/gsl_histogram.h>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    string activationString = "l";
    int numNeuronsPerLayer = 10;
    size_t numBins = 1000;
    size_t numLayers = 7;
    vector<double> meanActivation(numLayers, 0.0);
    vector<double> sigmaActivation(numLayers, 0.0);
    vector<size_t> countActivation(numLayers, 0);
    vector<double> meanGradient(numLayers - 1, 0.0);
    vector<double> sigmaGradient(numLayers - 1, 0.0);
    vector<size_t> countGradient(numLayers - 1, 0);

    mt19937_64 rng;
    rng.seed(0);
    uniform_real_distribution<> distUniform(-1.0, 1.0);
    normal_distribution<> distNormal(0.0, 0.3);
    auto generatorUniform = [&distUniform, &rng]() {return distUniform(rng);};
    auto generatorNormal = [&distNormal, &rng]() {return distNormal(rng);};

    if (argc != 2)
    {
        cout << "USAGE: " << argv[0] << " <activation>\n"
             << "       <activation> ... Activation function type.\n";
        return 1;
    }

    activationString = argv[1];

    ofstream logFile;
    logFile.open("nnp-winit.log");
    Log log;
    log.registerStreamPointer(&logFile);
    log << "\n";
    log << "*** WEIGHT INITIALIZATION TESTING *******"
           "**************************************\n";
    log << "\n";

    NeuralNetworkTopology t;
    t.numLayers = numLayers;
    t.activationFunctionsPerLayer.resize(t.numLayers);
    fill(t.activationFunctionsPerLayer.begin(),
         t.activationFunctionsPerLayer.end() - 1,
         activationFromString(activationString));
    t.activationFunctionsPerLayer.back() = activationFromString("l");
    t.numNeuronsPerLayer.resize(t.numLayers);
    fill(t.numNeuronsPerLayer.begin(),
         t.numNeuronsPerLayer.end() - 1,
         numNeuronsPerLayer);
    t.numNeuronsPerLayer.back() = 1;

    NeuralNetwork nn(t.numLayers,
                     t.numNeuronsPerLayer.data(),
                     t.activationFunctionsPerLayer.data());
    
    log << nn.info();

    vector<gsl_histogram*> histActivation;
    vector<gsl_histogram*> histGradient;
    for (size_t i = 0; i < numLayers; ++i)
    {
        histActivation.push_back(gsl_histogram_alloc(numBins));
        gsl_histogram_set_ranges_uniform(histActivation.back(), -1.0, 1.0);
        if (i < numLayers - 1)
        {
            histGradient.push_back(gsl_histogram_alloc(numBins));
            gsl_histogram_set_ranges_uniform(histGradient.back(), -1.0, 1.0);
        }
    }

    for (size_t i = 0; i < 100000; ++i)
    {
        vector<double> input(numNeuronsPerLayer);
        generate(input.begin(), input.end(), generatorNormal);
        nn.setInput(input.data());

        vector<double> connections(nn.getNumConnections());
        generate(connections.begin(), connections.end(), generatorNormal);
        nn.setConnections(connections.data());
        nn.modifyConnections(NeuralNetwork::MS_GLOROTBENGIO_NORMAL);
        //nn.modifyConnections(NeuralNetwork::MS_FANIN_NORMAL);
        nn.modifyConnections(NeuralNetwork::MS_ZEROBIAS);

        nn.propagate();

        auto activations = nn.getNeuronsPerLayer();

        auto gradient = nn.getGradientsPerLayer();

        for (size_t il = 0; il < numLayers; ++il)
        {
            auto const& l = activations.at(il);
            for (auto const& a : l)
            {
                gsl_histogram_increment(histActivation.at(il), a);
                // Welford's algorithm (use sigma as M2 storage).
                countActivation.at(il)++;
                double const delta = a - meanActivation.at(il);
                meanActivation.at(il) += delta / countActivation.at(il);
                double const delta2 = a - meanActivation.at(il);
                sigmaActivation.at(il) += delta * delta2;
            }
            if (il < numLayers - 1)
            {
                auto const& l = gradient.at(il);
                for (auto const& g : l)
                {
                    gsl_histogram_increment(histGradient.at(il), g);
                    countGradient.at(il)++;
                    double const delta = g - meanGradient.at(il);
                    meanGradient.at(il) += delta / countGradient.at(il);
                    double const delta2 = g - meanGradient.at(il);
                    sigmaGradient.at(il) += delta * delta2;
                }
            }
        }
    }

    //ofstream f;
    //f.open("weights.dat");
    //nn.writeConnections(f);
    //f.close();

    for (size_t i = 0; i < numLayers; ++i)
    {
        sigmaActivation.at(i) = sqrt(sigmaActivation.at(i)
                                     / (countActivation.at(i) - 1));
        if (i < numLayers - 1)
        {
            sigmaGradient.at(i) = sqrt(sigmaGradient.at(i)
                                       / (countGradient.at(i) - 1));
        }
    }

    log << "-----------------------------------------"
           "--------------------------------------\n";
    log << "Layer      |";
    for (size_t i = 0; i < numLayers; ++i)
    {
        log << strpr(" %12zu", i + 1);
    }
    log << "\n";
    log << "-----------------------------------------"
           "--------------------------------------\n";
    log << "Activation |";
    log << strpr(" %12s", "G");
    for (size_t i = 1; i < numLayers; ++i)
    {
        log << strpr(" %12s", stringFromActivation(
                                  t.activationFunctionsPerLayer[i]).c_str());
    }
    log << "\n";
    log << "Count      |";
    for (size_t i = 0; i < numLayers; ++i)
    {
        log << strpr(" %12.3E", static_cast<double>(countActivation.at(i)));
    }
    log << "\n";
    log << "Mean       |";
    for (size_t i = 0; i < numLayers; ++i)
    {
        log << strpr(" %12.5f", meanActivation.at(i));
    }
    log << "\n";
    log << "Sigma      |";
    for (size_t i = 0; i < numLayers; ++i)
    {
        log << strpr(" %12.5f", sigmaActivation.at(i));
    }
    log << "\n";
    log << "-----------------------------------------"
           "--------------------------------------\n";
    log << "Gradient\n";
    log << "Count      |";
    log << strpr(" %12s", "");
    for (size_t i = 0; i < numLayers - 1; ++i)
    {
        log << strpr(" %12.3E", static_cast<double>(countGradient.at(i)));
    }
    log << "\n";
    log << "Mean       |";
    log << strpr(" %12s", "");
    for (size_t i = 0; i < numLayers - 1; ++i)
    {
        log << strpr(" %12.5f", meanGradient.at(i));
    }
    log << "\n";
    log << "Sigma      |";
    log << strpr(" %12s", "");
    for (size_t i = 0; i < numLayers - 1; ++i)
    {
        log << strpr(" %12.5f", sigmaGradient.at(i));
    }
    log << "\n";
    log << "-----------------------------------------"
           "--------------------------------------\n";

    log << strpr("Writing activation histograms for %zu layers.\n", numLayers);
    for (size_t i = 0; i < numLayers; ++i)
    {

        string fileName = strpr("activation.%03zu.out", i + 1);
        FILE* fp = 0;
        fp = fopen(fileName.c_str(), "w");

        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back(strpr("Activations histogram for layer %zu with "
                              "activation function %s.",
                              i + 1,
                              stringFromActivation(
                                  t.activationFunctionsPerLayer[i]).c_str()));
        colSize.push_back(16);
        colName.push_back("activation_l");
        colInfo.push_back("Activation value, left bin limit.");
        colSize.push_back(16);
        colName.push_back("activation_r");
        colInfo.push_back("Activation value, right bin limit.");
        colSize.push_back(16);
        colName.push_back("hist");
        colInfo.push_back("Histogram count.");
        appendLinesToFile(fp,
                          createFileHeader(title,
                                           colSize,
                                           colName,
                                           colInfo));
        gsl_histogram_fprintf(fp, histActivation.at(i), "%16.8E", "%16.8E");
        fflush(fp);
        fclose(fp);
        fp = nullptr;
    }

    log << strpr("Writing gradient histograms for %zu layers.\n",
                 numLayers - 1);
    for (size_t i = 0; i < numLayers - 1; ++i)
    {

        string fileName = strpr("gradient.%03zu.out", i + 2);
        FILE* fp = 0;
        fp = fopen(fileName.c_str(), "w");

        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back(strpr("Gradient histogram for layer %zu.", i + 2));
        colSize.push_back(16);
        colName.push_back("gradient_l");
        colInfo.push_back("Gradient value, left bin limit.");
        colSize.push_back(16);
        colName.push_back("gradient_r");
        colInfo.push_back("Gradient value, right bin limit.");
        colSize.push_back(16);
        colName.push_back("hist");
        colInfo.push_back("Histogram count.");
        appendLinesToFile(fp,
                          createFileHeader(title,
                                           colSize,
                                           colName,
                                           colInfo));
        gsl_histogram_fprintf(fp, histGradient.at(i), "%16.8E", "%16.8E");
        fflush(fp);
        fclose(fp);
        fp = nullptr;
    }

    log << "Finished.\n";
    log << "*****************************************"
           "**************************************\n";
    logFile.close();

    return 0;
}
