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
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    vector<size_t> numNeuronsPerLayer = {10, 15, 15, 3};
    vector<string> activationPerLayer = {"G", "t", "s", "l"};
    NeuralNetworx nn(numNeuronsPerLayer, activationPerLayer);

    for (auto l : nn.info()) cout << l;

    return 0;
}
