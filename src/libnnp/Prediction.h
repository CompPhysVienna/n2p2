// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
    std::string formatWeightsFiles;
    Structure   structure;
};

}

#endif
