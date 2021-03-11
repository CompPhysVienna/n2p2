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

#include "Prediction.h"
#include <fstream>   // std::ifstream
#include <stdexcept> // std::runtime_error
#include "utility.h"

using namespace std;
using namespace nnp;

Prediction::Prediction() : Mode(),
                           fileNameSettings        ("input.nn"          ),
                           fileNameScaling         ("scaling.data"      ),
                           formatWeightsFilesShort ("weights.%03zu.data" ),
                           formatWeightsFilesCharge("weightse.%03zu.data")
{
}

void Prediction::setup()
{
    initialize();
    loadSettingsFile(fileNameSettings);
    setupGeneric();
    setupSymmetryFunctionScaling(fileNameScaling);
    setupNeuralNetworkWeights(formatWeightsFilesShort,
                              formatWeightsFilesCharge);
    setupSymmetryFunctionStatistics(false, false, true, false);
}

void Prediction::readStructureFromFile(string const& fileName)
{
    ifstream file;
    file.open(fileName.c_str());
    structure.reset();
    structure.setElementMap(elementMap);
    structure.readFromFile(fileName);
    removeEnergyOffset(structure);
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
#ifdef N2P2_NO_SF_GROUPS
    calculateSymmetryFunctions(structure, true);
#else
    calculateSymmetryFunctionGroups(structure, true);
#endif
    calculateAtomicNeuralNetworks(structure, true);
    calculateEnergy(structure);
    if (nnpType == NNPType::SHORT_CHARGE_NN) calculateCharge(structure);
    calculateForces(structure);
    if (normalize)
    {
        structure.toPhysicalUnits(meanEnergy, convEnergy, convLength);
    }
    addEnergyOffset(structure, false);
    addEnergyOffset(structure, true);

    return;
}

void Prediction::print()
{
    if (committeeMode == CommitteeMode::DISABLED)
    {
        log << strpr("NNP energy: %16.8E\n",
                                structure.energy);
        log << "\n";
        log << "NNP forces:\n";
        for (vector<Atom>::const_iterator it = structure.atoms.begin();
             it != structure.atoms.end(); ++it)
        {
            log << strpr("%10zu %2s %16.8E %16.8E %16.8E\n",
                            it->index + 1,
                            elementMap[it->element].c_str(),
                            it->element,
                            it->f[0],
                            it->f[1],
                            it->f[2]);
        }
    }
    else
    {
        log << strpr("Committee has %i members.\n",committeeSize);
        log << "-----------------------------------------"
                  "--------------------------------------\n";
        if (committeeMode == CommitteeMode::VALIDATION) 
            log << strpr("NNP energy | committee energy disagreement:\n");
        else
            log << strpr("NNP committee energy | committee energy disagreement:\n");
        log << strpr("    %16.8E | %16.8E\n",structure.energy, structure.committeeDisagreement);                                
        log << "\n";
        if (committeeMode == CommitteeMode::VALIDATION) 
            log << "NNP forces | committee force disagreements:\n";
        else
            log << "NNP committee forces | committee forces disagreement:\n";
        for (vector<Atom>::const_iterator it = structure.atoms.begin();
             it != structure.atoms.end(); ++it)
        {
            log << strpr("%10zu %2s %16.8E %16.8E %16.8E | %16.8E %16.8E %16.8E\n",
                            it->index + 1,
                            elementMap[it->element].c_str(),
                            it->element,
                            it->f[0],
                            it->f[1],
                            it->f[2],
                            it->committeeDisagreement[0],
                            it->committeeDisagreement[1],
                            it->committeeDisagreement[2]);
        }
        ofstream committee;
        committee.open("committee.log");
        committee << "-----------------------------------------"
                      "--------------------------------------\n";
        committee << "Committee energy:\n";
        committee << "-----------------------------------------"
                      "--------------------------------------\n";        
        for (auto& ecom : structure.energyCom)
            committee << strpr("\t%16.8E\n",ecom);
        committee << "\n-----------------------------------------"
                      "--------------------------------------\n";
        committee << "Committee forces (x,y,z):\n"; 
        committee << "-----------------------------------------"
                      "--------------------------------------\n";
        for (vector<Atom>::const_iterator it = structure.atoms.begin();
            it != structure.atoms.end(); ++it)
        {
            for (size_t c = 0; c < it->fCom.size(); ++c)
            {
                committee << strpr("%10zu %2s %16.8E %16.8E %16.8E\n",
                                    it->index + 1,
                                    elementMap[it->element].c_str(),
                                    it->element,
                                    it->fCom.at(c)[0],
                                    it->fCom.at(c)[1],
                                    it->fCom.at(c)[2]);
            }
        }
        committee.close();
    }
}
