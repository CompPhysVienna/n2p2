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
#include "Stopwatch.h"
#include <fstream>   // std::ifstream
#include <iostream>
#include <map>       // std::map
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
    map<string, string> formatWeights {
        {"short", formatWeightsFilesShort},
        {"elec", formatWeightsFilesCharge}
    };
    setupNeuralNetworkWeights(formatWeights);
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
        structure.toNormalizedUnits(meanEnergy, convEnergy, convLength, convCharge);
    }
    file.close();

    return;
}

void Prediction::predict()
{
    evaluateNNP(structure);
    if (normalize)
    {
        structure.toPhysicalUnits(meanEnergy, convEnergy, convLength, convCharge);
    }
    addEnergyOffset(structure, false);
    addEnergyOffset(structure, true);

    return;
}

void Prediction::print()
{
    if (committeeMode == CommitteeMode::DISABLED)
    {
        log << strpr("NNP         total energy: %16.8E\n",
                     structure.energy);
        if (nnpType == NNPType::HDNNP_4G)
        {
            log << strpr("NNP electrostatic energy: %16.8E\n",
                         structure.energyElec);
            log << "\n";
            log << "NNP charges:\n";
            for (auto const& a : structure.atoms)
            {
                log << strpr("%10zu %2s %16.8E\n",
                             a.index + 1,
                             elementMap[a.element].c_str(),
                             a.charge);
            }
            log << strpr("NNP total charge: %16.8E (ref: %16.8E)\n",
                         structure.charge, structure.chargeRef);
        }
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
        ofstream comMembers;
        comMembers.open("committee-members.log");
        comMembers << "-----------------------------------------"
                      "--------------------------------------\n";
        comMembers << "Committee energy: \n";
        comMembers << "-----------------------------------------"
                      "--------------------------------------\n";        
        for (size_t c = 0; c < structure.energyCom.size(); ++c)
            comMembers << strpr("%10zu %16.8E\n",c,structure.energyCom[c]);
        comMembers << "\n-----------------------------------------"
                      "--------------------------------------\n";
        comMembers << "AtomIndex CommitteeMember Element Force(x) Force(y) Force(z)\n"; 
        comMembers << "-----------------------------------------"
                      "--------------------------------------\n";
        for (vector<Atom>::const_iterator it = structure.atoms.begin();
            it != structure.atoms.end(); ++it)
        {
            for (size_t c = 0; c < it->fCom.size(); ++c)
            {
                comMembers << strpr("%10zu %3i %2s %16.8E %16.8E %16.8E\n",
                                    it->index + 1,
                                    c,
                                    elementMap[it->element].c_str(),
                                    it->element,
                                    it->fCom.at(c)[0],
                                    it->fCom.at(c)[1],
                                    it->fCom.at(c)[2]);
            }
        }
        comMembers.close();
    }
    log << "-----------------------------------------"
          "--------------------------------------\n";
    log << "Writing output files...\n";
    log << " - energy.out\n";
    ofstream file;
    file.open("energy.out");

    // File header.
    vector<string> title;
    vector<string> colName;
    vector<string> colInfo;
    vector<size_t> colSize;
    title.push_back("Energy comparison.");
    colSize.push_back(10);
    colName.push_back("conf");
    colInfo.push_back("Configuration index (starting with 1).");
    colSize.push_back(10);
    colName.push_back("natoms");
    colInfo.push_back("Number of atoms in configuration.");
    colSize.push_back(24);
    colName.push_back("Eref");
    colInfo.push_back("Reference potential energy.");
    colSize.push_back(24);
    colName.push_back("Ennp");
    colInfo.push_back("Potential energy predicted by NNP.");
    colSize.push_back(24);
    colName.push_back("Ediff");
    colInfo.push_back("Difference in energy per atom between reference and "
                      "NNP prediction.");
    if (committeeMode != CommitteeMode::DISABLED)
    {
        colSize.push_back(24);
        colName.push_back("Ecomm");
        colInfo.push_back("Committee disagreement.");
    }
    colSize.push_back(24);
    colName.push_back("E_offset");
    colInfo.push_back("Sum of atomic offset energies "
                      "(included in column Ennp).");
    appendLinesToFile(file,
                      createFileHeader(title, colSize, colName, colInfo));

    if (committeeMode == CommitteeMode::DISABLED)
    {
        file << strpr("%10zu %10zu %24.16E %24.16E %24.16E %24.16E\n",
                      structure.index + 1,
                      structure.numAtoms,
                      structure.energyRef,
                      structure.energy,
                      (structure.energyRef - structure.energy)
                      / structure.numAtoms,
                      getEnergyOffset(structure));
    }
    else
    {
        file << strpr("%10zu %10zu %24.16E %24.16E %24.16E %24.16E "
                      "%24.16E\n",
                      structure.index + 1,
                      structure.numAtoms,
                      structure.energyRef,
                      structure.energy,
                      (structure.energyRef - structure.energy)
                      / structure.numAtoms,
                      structure.committeeDisagreement,
                      getEnergyOffset(structure));
    }
    file.close();

    log << " - nnatoms.out\n";
    file.open("nnatoms.out");

    // File header.
    title.clear();
    colName.clear();
    colInfo.clear();
    colSize.clear();
    title.push_back("Energy contributions calculated from NNP.");
    colSize.push_back(10);
    colName.push_back("conf");
    colInfo.push_back("Configuration index (starting with 1).");
    colSize.push_back(10);
    colName.push_back("index");
    colInfo.push_back("Atom index (starting with 1).");
    colSize.push_back(3);
    colName.push_back("Z");
    colInfo.push_back("Nuclear charge of atom.");
    colSize.push_back(24);
    colName.push_back("Qref");
    colInfo.push_back("Reference atomic charge.");
    colSize.push_back(24);
    colName.push_back("Qnnp");
    colInfo.push_back("NNP atomic charge.");
    colSize.push_back(24);
    colName.push_back("Eref_atom");
    colInfo.push_back("Reference atomic energy contribution.");
    colSize.push_back(24);
    colName.push_back("Ennp_atom");
    colInfo.push_back("Atomic energy contribution (physical units, no mean "
                      "or offset energy added).");
    appendLinesToFile(file,
                      createFileHeader(title, colSize, colName, colInfo));

    for (vector<Atom>::const_iterator it = structure.atoms.begin();
         it != structure.atoms.end(); ++it)
    {
        file << strpr("%10zu %10zu %3zu %24.16E %24.16E %24.16E %24.16E\n",
                      structure.index + 1,
                      it->index + 1,
                      elementMap.atomicNumber(it->element),
                      it->chargeRef,
                      it->charge,
                      0.0,
                      it->energy);
    }
    file.close();

    log << " - nnforces.out\n";
    file.open("nnforces.out");

    // File header.
    title.clear();
    colName.clear();
    colInfo.clear();
    colSize.clear();
    title.push_back("Atomic force comparison (ordered by atom index).");
    colSize.push_back(10);
    colName.push_back("conf");
    colInfo.push_back("Configuration index (starting with 1).");
    colSize.push_back(10);
    colName.push_back("index");
    colInfo.push_back("Atom index (starting with 1).");
    colSize.push_back(24);
    colName.push_back("fxRef");
    colInfo.push_back("Reference force in x direction.");
    colSize.push_back(24);
    colName.push_back("fyRef");
    colInfo.push_back("Reference force in y direction.");
    colSize.push_back(24);
    colName.push_back("fzRef");
    colInfo.push_back("Reference force in z direction.");
    colSize.push_back(24);
    colName.push_back("fx");
    colInfo.push_back("Force in x direction.");
    colSize.push_back(24);
    colName.push_back("fy");
    colInfo.push_back("Force in y direction.");
    colSize.push_back(24);
    colName.push_back("fz");
    colInfo.push_back("Force in z direction.");
    if (committeeMode != CommitteeMode::DISABLED)
    {
        colSize.push_back(24);
        colName.push_back("fxComm");
        colInfo.push_back("Committee disagreement in x dir.");
        colSize.push_back(24);
        colName.push_back("fyComm");
        colInfo.push_back("Committee disagreement in y dir.");
        colSize.push_back(24);
        colName.push_back("fzComm");
        colInfo.push_back("Committee disagreement in z dir.");
    }
    appendLinesToFile(file,
                      createFileHeader(title, colSize, colName, colInfo));

    for (vector<Atom>::const_iterator it = structure.atoms.begin();
         it != structure.atoms.end(); ++it)
    {
        if (committeeMode == CommitteeMode::DISABLED)
        {
            file << strpr("%10zu %10zu %24.16E %24.16E %24.16E %24.16E "
                          "%24.16E %24.16E\n",
                          structure.index + 1,
                          it->index + 1,
                          it->fRef[0],
                          it->fRef[1],
                          it->fRef[2],
                          it->f[0],
                          it->f[1],
                          it->f[2]);
        }
        else
        {
            file << strpr("%10zu %10zu %24.16E %24.16E %24.16E %24.16E "
                          "%24.16E %24.16E %24.16E %24.16E %24.16E\n",
                          structure.index + 1,
                          it->index + 1,
                          it->fRef[0],
                          it->fRef[1],
                          it->fRef[2],
                          it->f[0],
                          it->f[1],
                          it->f[2],
                          it->committeeDisagreement[0],
                          it->committeeDisagreement[1],
                          it->committeeDisagreement[2]);
        }
    }
    file.close();
}
