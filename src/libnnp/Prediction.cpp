// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Prediction.h"
#include <fstream>   // std::ifstream
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

Prediction::Prediction() : Mode(),
                           fileNameSettings  ("input.nn"          ),
                           fileNameScaling   ("scaling.data"      ),
                           formatWeightsFiles("weights.%03zu.data")
{
}

void Prediction::setup()
{
    initialize();
    loadSettingsFile(fileNameSettings);
    setupGeneric();
    setupSymmetryFunctionScaling(fileNameScaling);
    setupNeuralNetworkWeights(formatWeightsFiles);
    setupSymmetryFunctionStatistics(false, false, true, false);
}

void Prediction::readStructureFromFile(string const& fileName)
{
    ifstream file;
    file.open(fileName.c_str());
    if (!file.is_open())
    {
        throw runtime_error("ERROR: Could not open file: \"" + fileName
                            + "\".\n");
    }
    structure.reset();
    structure.setElementMap(elementMap);
    structure.readFromFile(file);
    if (normalize)
    {
        structure.toNormalizedUnits(meanEnergy, convEnergy, convLength);
    }
    file.close();

    return;
}

void Prediction::predict()
{
    structure.calculateNeighborList(maxCutoffRadius);
#ifdef NOSFGROUPS
    calculateSymmetryFunctions(structure, true);
#else
    calculateSymmetryFunctionGroups(structure, true);
#endif
    calculateAtomicNeuralNetworks(structure, true);
    calculateEnergy(structure);
    calculateForces(structure);

    return;
}
