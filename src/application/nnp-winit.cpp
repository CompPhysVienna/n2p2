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

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    string activationString = "l";
    int numNeuronsPerLayer = 25;

    mt19937_64 rng;
    rng.seed(0);
    uniform_real_distribution<> dist(-1.0, 1.0);
    auto generator = [&dist, &rng]() {return dist(rng);}; 

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
    t.numLayers = 3;
    t.activationFunctionsPerLayer.resize(t.numLayers);
    fill(t.activationFunctionsPerLayer.begin(),
         t.activationFunctionsPerLayer.end(),
         activationFromString(activationString));
    t.numNeuronsPerLayer.resize(t.numLayers);
    fill(t.numNeuronsPerLayer.begin(),
         t.numNeuronsPerLayer.end(),
         numNeuronsPerLayer);

    NeuralNetwork nn(t.numLayers,
                     t.numNeuronsPerLayer.data(),
                     t.activationFunctionsPerLayer.data());
    
    log << nn.info();

    vector<double> input(numNeuronsPerLayer);
    generate(input.begin(), input.end(), generator);
    nn.setInput(input.data());

    vector<double> connections(nn.getNumConnections());
    generate(connections.begin(), connections.end(), generator);
    nn.setConnections(connections.data());

    nn.propagate();

    vector<double> output(numNeuronsPerLayer);
    nn.getOutput(output.data());

    for (auto o : output) log << strpr("%f\n", o);


    log << "Finished.\n";
    log << "*****************************************"
           "**************************************\n";
    logFile.close();

    return 0;
}
