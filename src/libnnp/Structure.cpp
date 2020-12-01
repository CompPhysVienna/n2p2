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

#include "Structure.h"
#include "utility.h"
#include <algorithm> // std::max
#include <cmath>     // fabs
#include <cstdlib>   // atof
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <string>    // std::getline
#include <iostream>

using namespace std;
using namespace nnp;

Structure::Structure() :
    isPeriodic                    (false     ),
    isTriclinic                   (false     ),
    hasNeighborList               (false     ),
    hasSymmetryFunctions          (false     ),
    hasSymmetryFunctionDerivatives(false     ),
    index                         (0         ),
    numAtoms                      (0         ),
    numElements                   (0         ),
    numElementsPresent            (0         ),
    energy                        (0.0       ),
    energyRef                     (0.0       ),
    charge                        (0.0       ),
    chargeRef                     (0.0       ),
    volume                        (0.0       ),
    sampleType                    (ST_UNKNOWN),
    comment                       (""        )
{
    for (size_t i = 0; i < 3; i++)
    {
        pbc[i] = 0;
        box[i][0] = 0.0;
        box[i][1] = 0.0;
        box[i][2] = 0.0;
        invbox[i][0] = 0.0;
        invbox[i][1] = 0.0;
        invbox[i][2] = 0.0;
    }
}

void Structure::setElementMap(ElementMap const& elementMap)
{
    this->elementMap = elementMap;
    numElements = elementMap.size();
    numAtomsPerElement.resize(numElements, 0);

    return;
}

void Structure::addAtom(Atom const& atom, string const& element)
{
    atoms.push_back(Atom());
    atoms.back() = atom;
    // The number of elements may have changed.
    atoms.back().numNeighborsPerElement.resize(elementMap.size(), 0);
    atoms.back().clearNeighborList();
    atoms.back().index                = numAtoms;
    atoms.back().indexStructure       = index;
    atoms.back().element              = elementMap[element];
    atoms.back().numSymmetryFunctions = 0;
    numAtoms++;
    numAtomsPerElement[elementMap[element]]++;

    return;
}

void Structure::readFromFile(string const fileName)
{
    ifstream file;

    file.open(fileName.c_str());
    if (!file.is_open())
    {
        throw runtime_error("ERROR: Could not open file: \"" + fileName
                            + "\".\n");
    }
    readFromFile(file);
    file.close();

    return;
}

void Structure::readFromFile(ifstream& file)
{
    string         line;
    vector<string> lines;
    vector<string> splitLine;

    // read first line, should be keyword "begin".
    getline(file, line);
    lines.push_back(line);
    splitLine = split(reduce(line));
    if (splitLine.at(0) != "begin")
    {
        throw runtime_error("ERROR: Unexpected file content, expected"
                            " \"begin\" keyword.\n");
    }

    while (getline(file, line))
    {
        lines.push_back(line);
        splitLine = split(reduce(line));
        if (splitLine.at(0) == "end") break;
    }

    readFromLines(lines);

    return;
}


void Structure::readFromLines(vector<string> const& lines)
{
    size_t         iBoxVector = 0;
    vector<string> splitLine;

    // read first line, should be keyword "begin".
    splitLine = split(reduce(lines.at(0)));
    if (splitLine.at(0) != "begin")
    {
        throw runtime_error("ERROR: Unexpected line content, expected"
                            " \"begin\" keyword.\n");
    }

    for (vector<string>::const_iterator line = lines.begin();
         line != lines.end(); ++line)
    {
        splitLine = split(reduce(*line));
        if (splitLine.at(0) == "begin")
        {
        }
        else if (splitLine.at(0) == "comment")
        {
            size_t position = line->find("comment");
            string tmpLine = *line;
            comment = tmpLine.erase(position, splitLine.at(0).length() + 1);
        }
        else if (splitLine.at(0) == "lattice")
        {
            if (iBoxVector > 2)
            {
                throw runtime_error("ERROR: Too many box vectors.\n");
            }
            box[iBoxVector][0] = atof(splitLine.at(1).c_str());
            box[iBoxVector][1] = atof(splitLine.at(2).c_str());
            box[iBoxVector][2] = atof(splitLine.at(3).c_str());
            iBoxVector++;
            if (iBoxVector == 3)
            {
                isPeriodic = true;
                if (box[0][1] > numeric_limits<double>::min() ||
                    box[0][2] > numeric_limits<double>::min() ||
                    box[1][0] > numeric_limits<double>::min() ||
                    box[1][2] > numeric_limits<double>::min() ||
                    box[2][0] > numeric_limits<double>::min() ||
                    box[2][1] > numeric_limits<double>::min())
                {
                    isTriclinic = true;
                }
                calculateInverseBox();
                calculateVolume();
            }
        }
        else if (splitLine.at(0) == "atom")
        {
            atoms.push_back(Atom());
            atoms.back().index          = numAtoms;
            atoms.back().indexStructure = index;
            atoms.back().tag            = numAtoms;
            atoms.back().r[0]           = atof(splitLine.at(1).c_str());
            atoms.back().r[1]           = atof(splitLine.at(2).c_str());
            atoms.back().r[2]           = atof(splitLine.at(3).c_str());
            atoms.back().element        = elementMap[splitLine.at(4)];
            atoms.back().chargeRef      = atof(splitLine.at(5).c_str());
            atoms.back().fRef[0]        = atof(splitLine.at(7).c_str());
            atoms.back().fRef[1]        = atof(splitLine.at(8).c_str());
            atoms.back().fRef[2]        = atof(splitLine.at(9).c_str());
            atoms.back().numNeighborsPerElement.resize(numElements, 0);
            numAtoms++;
            numAtomsPerElement[elementMap[splitLine.at(4)]]++;
        }
        else if (splitLine.at(0) == "energy")
        {
            energyRef = atof(splitLine[1].c_str());
        }
        else if (splitLine.at(0) == "charge")
        {
            chargeRef = atof(splitLine[1].c_str());
        }
        else if (splitLine.at(0) == "end")
        {
            if (!(iBoxVector == 0 || iBoxVector == 3))
            {
                throw runtime_error("ERROR: Strange number of box vectors.\n");
            }
            break;
        }
        else
        {
            throw runtime_error("ERROR: Unexpected file content, "
                                "unknown keyword \"" + splitLine.at(0) +
                                "\".\n");
        }
    }

    for (size_t i = 0; i < numElements; i++)
    {
        if (numAtomsPerElement[i] > 0)
        {
            numElementsPresent++;
        }
    }

    if (isPeriodic)
    {
        for (vector<Atom>::iterator it = atoms.begin();
             it != atoms.end(); ++it)
        {
            remap((*it));
        }
    }

    return;
}

void Structure::calculateNeighborList(double cutoffRadius)
{
    if (isPeriodic)
    {
        calculatePbcCopies(cutoffRadius);

        // Use square of cutoffRadius (faster).
        cutoffRadius *= cutoffRadius;

        size_t i = 0;
#ifdef _OPENMP
        #pragma omp parallel for private(i)
#endif
        for (i = 0; i < numAtoms; i++)
        {
            // Count atom i as unique neighbor.
            atoms[i].neighborsUnique.push_back(i);
            atoms[i].numNeighborsUnique++;
            for (size_t j = 0; j < numAtoms; j++)
            {
                for (int bc0 = -pbc[0]; bc0 <= pbc[0]; bc0++)
                {
                    for (int bc1 = -pbc[1]; bc1 <= pbc[1]; bc1++)
                    {
                        for (int bc2 = -pbc[2]; bc2 <= pbc[2]; bc2++)
                        {
                            if (!(i == j && bc0 == 0 && bc1 == 0 && bc2 == 0))
                            {
                                Vec3D dr = atoms[i].r - atoms[j].r
                                         + bc0 * box[0]
                                         + bc1 * box[1]
                                         + bc2 * box[2];
                                if (dr.norm2() <= cutoffRadius)
                                {
                                    atoms[i].neighbors.
                                        push_back(Atom::Neighbor());
                                    atoms[i].neighbors.
                                        back().index = j;
                                    atoms[i].neighbors.
                                        back().tag = j;
                                    atoms[i].neighbors.
                                        back().element = atoms[j].element;
                                    atoms[i].neighbors.
                                        back().d = dr.norm();
                                    atoms[i].neighbors.
                                        back().dr = dr;
                                    atoms[i].numNeighborsPerElement[
                                        atoms[j].element]++;
                                    atoms[i].numNeighbors++;
                                    // Count atom j only once as unique
                                    // neighbor.
                                    if (atoms[i].neighborsUnique.back() != j &&
                                        i != j)
                                    {
                                        atoms[i].neighborsUnique.push_back(j);
                                        atoms[i].numNeighborsUnique++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            atoms[i].hasNeighborList = true;
        }
    }
    else
    {
        // Use square of cutoffRadius (faster).
        cutoffRadius *= cutoffRadius;

        size_t i = 0;
#ifdef _OPENMP
        #pragma omp parallel for private(i)
#endif
        for (i = 0; i < numAtoms; i++)
        {
            // Count atom i as unique neighbor.
            atoms[i].neighborsUnique.push_back(i);
            atoms[i].numNeighborsUnique++;
            for (size_t j = 0; j < numAtoms; j++)
            {
                if (i != j)
                {
                    Vec3D dr = atoms[i].r - atoms[j].r;
                    if (dr.norm2() <= cutoffRadius)
                    {
                        atoms[i].neighbors.push_back(Atom::Neighbor());
                        atoms[i].neighbors.back().index   = j;
                        atoms[i].neighbors.back().tag     = j;
                        atoms[i].neighbors.back().element = atoms[j].element;
                        atoms[i].neighbors.back().d       = dr.norm();
                        atoms[i].neighbors.back().dr      = dr;
                        atoms[i].numNeighborsPerElement[atoms[j].element]++;
                        atoms[i].numNeighbors++;
                        atoms[i].neighborsUnique.push_back(j);
                        atoms[i].numNeighborsUnique++;
                    }
                }
            }
            atoms[i].hasNeighborList = true;
        }
    }

    hasNeighborList = true;

    return;
}

size_t Structure::getMaxNumNeighbors() const
{
    size_t maxNumNeighbors = 0;

    for(vector<Atom>::const_iterator it = atoms.begin();
        it != atoms.end(); ++it)
    {
        maxNumNeighbors = max(maxNumNeighbors, it->numNeighbors);
    }

    return maxNumNeighbors;
}

void Structure::calculatePbcCopies(double cutoffRadius)
{
    Vec3D axb;
    Vec3D axc;
    Vec3D bxc;

    axb = box[0].cross(box[1]).normalize();
    axc = box[0].cross(box[2]).normalize();
    bxc = box[1].cross(box[2]).normalize();

    double proja = fabs(box[0] * bxc);
    double projb = fabs(box[1] * axc);
    double projc = fabs(box[2] * axb);

    pbc[0] = 0;
    pbc[1] = 0;
    pbc[2] = 0;
    while (pbc[0] * proja <= cutoffRadius)
    {
        pbc[0]++;
    }
    while (pbc[1] * projb <= cutoffRadius)
    {
        pbc[1]++;
    }
    while (pbc[2] * projc <= cutoffRadius)
    {
        pbc[2]++;
    }

    return;
}

void Structure::calculateInverseBox()
{
    double invdet = box[0][0] * box[1][1] * box[2][2]
                  + box[1][0] * box[2][1] * box[0][2]
                  + box[2][0] * box[0][1] * box[1][2]
                  - box[2][0] * box[1][1] * box[0][2]
                  - box[0][0] * box[2][1] * box[1][2]
                  - box[1][0] * box[0][1] * box[2][2];
    invdet = 1.0 / invdet;

    invbox[0][0] = box[1][1] * box[2][2] - box[2][1] * box[1][2];
    invbox[0][1] = box[2][1] * box[0][2] - box[0][1] * box[2][2];
    invbox[0][2] = box[0][1] * box[1][2] - box[1][1] * box[0][2];
    invbox[0] *= invdet;

    invbox[1][0] = box[2][0] * box[1][2] - box[1][0] * box[2][2];
    invbox[1][1] = box[0][0] * box[2][2] - box[2][0] * box[0][2];
    invbox[1][2] = box[1][0] * box[0][2] - box[0][0] * box[1][2];
    invbox[1] *= invdet;

    invbox[2][0] = box[1][0] * box[2][1] - box[2][0] * box[1][1];
    invbox[2][1] = box[2][0] * box[0][1] - box[0][0] * box[2][1];
    invbox[2][2] = box[0][0] * box[1][1] - box[1][0] * box[0][1];
    invbox[2] *= invdet;

    return;
}

void Structure::calculateVolume()
{
    volume = fabs(box[0] * (box[1].cross(box[2])));

    return;
}

void Structure::remap(Atom& atom)
{
    Vec3D f = atom.r[0] * invbox[0]
            + atom.r[1] * invbox[1]
            + atom.r[2] * invbox[2];

    // Quick and dirty... there may be a more elegant way!
    if (f[0] > 1.0) f[0] -= (int)f[0];
    if (f[1] > 1.0) f[1] -= (int)f[1];
    if (f[2] > 1.0) f[2] -= (int)f[2];

    if (f[0] < 0.0) f[0] += 1.0 - (int)f[0];
    if (f[1] < 0.0) f[1] += 1.0 - (int)f[1];
    if (f[2] < 0.0) f[2] += 1.0 - (int)f[2];

    if (f[0] == 1.0) f[0] = 0.0;
    if (f[1] == 1.0) f[1] = 0.0;
    if (f[2] == 1.0) f[2] = 0.0;

    atom.r = f[0] * box[0]
           + f[1] * box[1]
           + f[2] * box[2];

    return;
}

void Structure::toNormalizedUnits(double meanEnergy,
                                  double convEnergy,
                                  double convLength)
{
    if (isPeriodic)
    {
        box[0] *= convLength;
        box[1] *= convLength;
        box[2] *= convLength;
        invbox[0] /= convLength;
        invbox[1] /= convLength;
        invbox[2] /= convLength;
    }

    energyRef = (energyRef - numAtoms * meanEnergy) * convEnergy;
    energy = (energy - numAtoms * meanEnergy) * convEnergy;
    volume *= convLength * convLength * convLength;

    for (vector<Atom>::iterator it = atoms.begin(); it != atoms.end(); ++it)
    {
        it->toNormalizedUnits(convEnergy, convLength);
    }

    return;
}

void Structure::toPhysicalUnits(double meanEnergy,
                                double convEnergy,
                                double convLength)
{
    if (isPeriodic)
    {
        box[0] /= convLength;
        box[1] /= convLength;
        box[2] /= convLength;
        invbox[0] *= convLength;
        invbox[1] *= convLength;
        invbox[2] *= convLength;
    }

    energyRef = energyRef / convEnergy + numAtoms * meanEnergy;
    energy = energy / convEnergy + numAtoms * meanEnergy;
    volume /= convLength * convLength * convLength;

    for (vector<Atom>::iterator it = atoms.begin(); it != atoms.end(); ++it)
    {
        it->toPhysicalUnits(convEnergy, convLength);
    }

    return;
}

void Structure::freeAtoms(bool all)
{
    for (vector<Atom>::iterator it = atoms.begin(); it != atoms.end(); ++it)
    {
        it->free(all);
    }
    if (all) hasSymmetryFunctions = false;
    hasSymmetryFunctionDerivatives = false;

    return;
}

void Structure::reset()
{
    isPeriodic                     = false     ;
    isTriclinic                    = false     ;
    hasNeighborList                = false     ;
    hasSymmetryFunctions           = false     ;
    hasSymmetryFunctionDerivatives = false     ;
    index                          = 0         ;
    numAtoms                       = 0         ;
    numElementsPresent             = 0         ;
    energy                         = 0.0       ;
    energyRef                      = 0.0       ;
    charge                         = 0.0       ;
    chargeRef                      = 0.0       ;
    volume                         = 0.0       ;
    sampleType                     = ST_UNKNOWN;
    comment                        = ""        ;

    for (size_t i = 0; i < 3; ++i)
    {
        pbc[i] = 0;
        for (size_t j = 0; j < 3; ++j)
        {
            box[i][j] = 0.0;
            invbox[i][j] = 0.0;
        }
    }

    numElements = elementMap.size();
    numAtomsPerElement.clear();
    numAtomsPerElement.resize(numElements, 0);
    atoms.clear();
    vector<Atom>(atoms).swap(atoms);

    return;
}

void Structure::clearNeighborList()
{
    for (size_t i = 0; i < numAtoms; i++)
    {
        Atom& a = atoms.at(i);
        // This may have changed if atoms are added via addAtoms().
        a.numNeighborsPerElement.resize(numElements, 0);
        a.clearNeighborList();
    }
    hasNeighborList = false;
    hasSymmetryFunctions = false;
    hasSymmetryFunctionDerivatives = false;

    return;
}

void Structure::updateError(string const&        property,
                            map<string, double>& error,
                            size_t&              count) const
{
    if (property == "energy")
    {
        count++;
        double diff = energyRef - energy;
        error.at("RMSEpa") += diff * diff / (numAtoms * numAtoms);
        error.at("RMSE") += diff * diff;
        diff = fabs(diff);
        error.at("MAEpa") += diff / numAtoms;
        error.at("MAE") += diff;
    }
    else if (property == "force" || property == "charge")
    {
        for (vector<Atom>::const_iterator it = atoms.begin();
             it != atoms.end(); ++it)
        {
            it->updateError(property, error, count);
        }
    }
    else
    {
        throw runtime_error("ERROR: Unknown property for error update.\n");
    }

    return;
}

string Structure::getEnergyLine() const
{
    return strpr("%10zu %16.8E %16.8E\n",
                 index,
                 energyRef / numAtoms,
                 energy / numAtoms);
}

vector<string> Structure::getForcesLines() const
{
    vector<string> v;
    for (vector<Atom>::const_iterator it = atoms.begin();
         it != atoms.end(); ++it)
    {
        vector<string> va = it->getForcesLines();
        v.insert(v.end(), va.begin(), va.end());
    }

    return v;
}

vector<string> Structure::getChargesLines() const
{
    vector<string> v;
    for (vector<Atom>::const_iterator it = atoms.begin();
         it != atoms.end(); ++it)
    {
        v.push_back(it->getChargeLine());
    }

    return v;
}

void Structure::writeToFile(string const fileName,
                            bool const   ref,
                            bool const   append) const
{
    ofstream file;

    if (append)
    {
        file.open(fileName.c_str(), ofstream::app);
    }
    else
    {
        file.open(fileName.c_str());
    }
    if (!file.is_open())
    {
        throw runtime_error("ERROR: Could not open file: \"" + fileName
                            + "\".\n");
    }
    writeToFile(&file, ref );
    file.close();

    return;
}

void Structure::writeToFile(ofstream* const& file, bool const ref) const
{
    if (!file->is_open())
    {
        runtime_error("ERROR: Cannot write to file, file is not open.\n");
    }

    (*file) << "begin\n";
    (*file) << strpr("comment %s\n", comment.c_str());
    if (isPeriodic)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            (*file) << strpr("lattice %24.16E %24.16E %24.16E\n",
                             box[i][0], box[i][1], box[i][2]);
        }
    }
    for (vector<Atom>::const_iterator it = atoms.begin();
         it != atoms.end(); ++it)
    {
        if (ref)
        {
            (*file) << strpr("atom %24.16E %24.16E %24.16E %2s %24.16E %24.16E"
                             " %24.16E %24.16E %24.16E\n",
                             it->r[0],
                             it->r[1],
                             it->r[2],
                             elementMap[it->element].c_str(),
                             it->chargeRef,
                             0.0,
                             it->fRef[0],
                             it->fRef[1],
                             it->fRef[2]);
        }
        else
        {
            (*file) << strpr("atom %24.16E %24.16E %24.16E %2s %24.16E %24.16E"
                             " %24.16E %24.16E %24.16E\n",
                             it->r[0],
                             it->r[1],
                             it->r[2],
                             elementMap[it->element].c_str(),
                             it->charge,
                             0.0,
                             it->f[0],
                             it->f[1],
                             it->f[2]);

        }
    }
    if (ref) (*file) << strpr("energy %24.16E\n", energyRef);
    else     (*file) << strpr("energy %24.16E\n", energy);
    if (ref) (*file) << strpr("charge %24.16E\n", chargeRef);
    else     (*file) << strpr("charge %24.16E\n", charge);
    (*file) << strpr("end\n");

    return;
}

void Structure::writeToFileXyz(ofstream* const& file) const
{
    if (!file->is_open())
    {
        runtime_error("ERROR: Could not write to file.\n");
    }

    (*file) << strpr("%d\n", numAtoms);
    if (isPeriodic)
    {
        (*file) << "Lattice=\"";
        (*file) << strpr("%24.16E %24.16E %24.16E "   ,
                         box[0][0], box[0][1], box[0][2]);
        (*file) << strpr("%24.16E %24.16E %24.16E "   ,
                         box[1][0], box[1][1], box[1][2]);
        (*file) << strpr("%24.16E %24.16E %24.16E\"\n",
                         box[2][0], box[2][1], box[2][2]);
    }
    else
    {
        (*file) << "\n";
    }
    for (vector<Atom>::const_iterator it = atoms.begin();
         it != atoms.end(); ++it)
    {
        (*file) << strpr("%-2s %24.16E %24.16E %24.16E\n",
                         elementMap[it->element].c_str(),
                         it->r[0],
                         it->r[1],
                         it->r[2]);
    }

    return;
}

void Structure::writeToFilePoscar(ofstream* const& file) const
{
    writeToFilePoscar(file, elementMap.getElementsString());

    return;
}

void Structure::writeToFilePoscar(ofstream* const& file,
                                  string const elements) const
{
    if (!file->is_open())
    {
        runtime_error("ERROR: Could not write to file.\n");
    }

    vector<string> ve = split(elements);
    vector<size_t> elementOrder;
    for (size_t i = 0; i < ve.size(); ++i)
    {
        elementOrder.push_back(elementMap[ve.at(i)]);
    }
    if (elementOrder.size() != elementMap.size())
    {
        runtime_error("ERROR: Inconsistent element declaration.\n");
    }

    (*file) << strpr("%s, ", comment.c_str());
    (*file) << strpr("ATOM=%s", elementMap[elementOrder.at(0)].c_str());
    for (size_t i = 1; i < elementOrder.size(); ++i)
    {
        (*file) << strpr(" %s", elementMap[elementOrder.at(i)].c_str());
    }
    (*file) << "\n";
    (*file) << "1.0\n";
    if (isPeriodic)
    {
        (*file) << strpr("%24.16E %24.16E %24.16E\n",
                         box[0][0], box[0][1], box[0][2]);
        (*file) << strpr("%24.16E %24.16E %24.16E\n",
                         box[1][0], box[1][1], box[1][2]);
        (*file) << strpr("%24.16E %24.16E %24.16E\n",
                         box[2][0], box[2][1], box[2][2]);
    }
    else
    {
        runtime_error("ERROR: Writing non-periodic structure to POSCAR file "
                      "is not implemented.\n");
    }
    (*file) << strpr("%d", numAtomsPerElement.at(elementOrder.at(0)));
    for (size_t i = 1; i < numAtomsPerElement.size(); ++i)
    {
        (*file) << strpr(" %d", numAtomsPerElement.at(elementOrder.at(i)));
    }
    (*file) << "\n";
    (*file) << "Cartesian\n";
    for (size_t i = 0; i < elementOrder.size(); ++i)
    {
        for (vector<Atom>::const_iterator it = atoms.begin();
             it != atoms.end(); ++it)
        {
            if (it->element == elementOrder.at(i))
            {
                (*file) << strpr("%24.16E %24.16E %24.16E\n",
                                 it->r[0],
                                 it->r[1],
                                 it->r[2]);
            }
        }
    }

    return;
}

vector<string> Structure::info() const
{
    vector<string> v;

    v.push_back(strpr("********************************\n"));
    v.push_back(strpr("STRUCTURE                       \n"));
    v.push_back(strpr("********************************\n"));
    vector<string> vm = elementMap.info();
    v.insert(v.end(), vm.begin(), vm.end());
    v.push_back(strpr("index                          : %d\n", index));
    v.push_back(strpr("isPeriodic                     : %d\n", isPeriodic        ));
    v.push_back(strpr("isTriclinic                    : %d\n", isTriclinic       ));
    v.push_back(strpr("hasNeighborList                : %d\n", hasNeighborList   ));
    v.push_back(strpr("hasSymmetryFunctions           : %d\n", hasSymmetryFunctions));
    v.push_back(strpr("hasSymmetryFunctionDerivatives : %d\n", hasSymmetryFunctionDerivatives));
    v.push_back(strpr("numAtoms                       : %d\n", numAtoms          ));
    v.push_back(strpr("numElements                    : %d\n", numElements       ));
    v.push_back(strpr("numElementsPresent             : %d\n", numElementsPresent));
    v.push_back(strpr("pbc                            : %d %d %d\n", pbc[0], pbc[1], pbc[2]));
    v.push_back(strpr("energy                         : %16.8E\n", energy        ));
    v.push_back(strpr("energyRef                      : %16.8E\n", energyRef     ));
    v.push_back(strpr("charge                         : %16.8E\n", charge        ));
    v.push_back(strpr("chargeRef                      : %16.8E\n", chargeRef     ));
    v.push_back(strpr("volume                         : %16.8E\n", volume        ));
    v.push_back(strpr("sampleType                     : %d\n", (int)sampleType));
    v.push_back(strpr("comment                        : %s\n", comment.c_str()   ));
    v.push_back(strpr("box[0]                         : %16.8E %16.8E %16.8E\n", box[0][0], box[0][1], box[0][2]));
    v.push_back(strpr("box[1]                         : %16.8E %16.8E %16.8E\n", box[1][0], box[1][1], box[1][2]));
    v.push_back(strpr("box[2]                         : %16.8E %16.8E %16.8E\n", box[2][0], box[2][1], box[2][2]));
    v.push_back(strpr("invbox[0]                      : %16.8E %16.8E %16.8E\n", invbox[0][0], invbox[0][1], invbox[0][2]));
    v.push_back(strpr("invbox[1]                      : %16.8E %16.8E %16.8E\n", invbox[1][0], invbox[1][1], invbox[1][2]));
    v.push_back(strpr("invbox[2]                      : %16.8E %16.8E %16.8E\n", invbox[2][0], invbox[2][1], invbox[2][2]));
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("numAtomsPerElement         [*] : %d\n", numAtomsPerElement.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < numAtomsPerElement.size(); ++i)
    {
        v.push_back(strpr("%29d  : %d\n", i, numAtomsPerElement.at(i)));
    }
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("atoms                      [*] : %d\n", atoms.size()));
    v.push_back(strpr("--------------------------------\n"));
    for (size_t i = 0; i < atoms.size(); ++i)
    {
        v.push_back(strpr("%29d  :\n", i));
        vector<string> va = atoms[i].info();
        v.insert(v.end(), va.begin(), va.end());
    }
    v.push_back(strpr("--------------------------------\n"));
    v.push_back(strpr("********************************\n"));

    return v;
}
