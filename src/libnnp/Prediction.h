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

#ifndef PREDICTION_H
#define PREDICTION_H

#include "Mode.h"
#include "Structure.h"
#include <string> // std::string

namespace nnp
{

class Prediction : public Mode
{
public:
    Prediction();
    void readStructureFromFile(std::string const& fileName = "input.data");
    void setup();
    void predict();

    std::string fileNameSettings;
    std::string fileNameScaling;
    std::string formatWeightsFilesShort;
    std::string formatWeightsFilesCharge;
    Structure   structure;
};

}

#endif
