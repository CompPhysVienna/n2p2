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

#ifndef NOMPI
#include <mpi.h>
#include "mpi-extra.h"
#endif
#include "InterfaceLammps.h"
#include "Atom.h"
#include "Element.h"
#include "utility.h"
#include <cmath>
#include <string>
#include <iostream>

#define TOLCUTOFF 1.0E-2

using namespace std;
using namespace nnp;

InterfaceLammps::InterfaceLammps() : myRank      (0    ),
                                     initialized (false),
                                     showew      (true ),
                                     resetew     (false),
                                     showewsum   (0    ),
                                     maxew       (0    ),
                                     cflength    (1.0  ),
                                     cfenergy    (1.0  )
                                     
{
}

void InterfaceLammps::initialize(char* const& directory,
                                 bool         showew,
                                 bool         resetew,
                                 int          showewsum,
                                 int          maxew,
                                 double       cflength,
                                 double       cfenergy,
                                 double       lammpsCutoff,
                                 int          lammpsNtypes,
                                 int          myRank)
{
    this->showew = showew;
    this->resetew = resetew;
    this->showewsum = showewsum;
    this->maxew = maxew;
    this->cflength = cflength;
    this->cfenergy = cfenergy;
    this->myRank = myRank;
    log.writeToStdout = false;
    string dir(directory);
    Mode::initialize();
    loadSettingsFile(dir + "input.nn");
    setupGeneric();
    setupSymmetryFunctionScaling(dir + "scaling.data");
    bool collectStatistics = false;
    bool collectExtrapolationWarnings = false;
    bool writeExtrapolationWarnings = false;
    bool stopOnExtrapolationWarnings = false;
    if (showew == true || showewsum > 0 || maxew >= 0)
    {
        collectExtrapolationWarnings = true;
    }
    setupSymmetryFunctionStatistics(collectStatistics,
                                    collectExtrapolationWarnings,
                                    writeExtrapolationWarnings,
                                    stopOnExtrapolationWarnings);
    setupNeuralNetworkWeights(dir + "weights.%03d.data");

    log << "\n";
    log << "*** SETUP: LAMMPS INTERFACE *************"
           "**************************************\n";
    log << "\n";

    if (showew)
    {
        log << "Individual extrapolation warnings will be shown.\n";
    }
    else
    {
        log << "Individual extrapolation warnings will not be shown.\n";
    }

    if (showewsum != 0)
    {
        log << strpr("Extrapolation warning summary will be shown every %d"
                     " timesteps.\n", showewsum);
    }
    else
    {
        log << "Extrapolation warning summary will not be shown.\n";
    }

    if (maxew != 0)
    {
        log << strpr("The simulation will be stopped when %d extrapolation"
                     " warnings are exceeded.\n", maxew);
    }
    else
    {
        log << "No extrapolation warning limit set.\n";
    }

    if (resetew)
    {
        log << "Extrapolation warning counter is reset every time step.\n";
    }
    else
    {
        log << "Extrapolation warnings are accumulated over all time steps.\n";
    }

    log << "-----------------------------------------"
           "--------------------------------------\n";
    log << "CAUTION: If the LAMMPS unit system differs from the one used\n";
    log << "         during NN training, appropriate conversion factors\n";
    log << "         must be provided (see keywords cflength and cfenergy).\n";
    log << "\n";
    log << strpr("Length unit conversion factor: %24.16E\n", cflength);
    log << strpr("Energy unit conversion factor: %24.16E\n", cfenergy);
    double sfCutoff = getMaxCutoffRadius();
    log << "\n";
    log << "Checking consistency of cutoff radii (in LAMMPS units):\n";
    log << strpr("LAMMPS Cutoff (via pair_coeff)  : %11.3E\n", lammpsCutoff);
    log << strpr("Maximum symmetry function cutoff: %11.3E\n", sfCutoff);
    if (lammpsCutoff < sfCutoff)
    {
        throw runtime_error("ERROR: LAMMPS cutoff via pair_coeff keyword is"
                            " smaller than maximum symmetry function"
                            " cutoff.\n");
    }
    else if (fabs(sfCutoff - lammpsCutoff) / lammpsCutoff > TOLCUTOFF)
    {
        log << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
        log << "WARNING: Potential length units mismatch!\n";
        log << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    }
    else
    {
        log << "Cutoff radii are consistent.\n";
    }

    if (elementMap.size() != (size_t)lammpsNtypes)
    {
        throw runtime_error(strpr("ERROR: Number of LAMMPS atom types (%d) and"
                                  " NNP elements (%zu) does not match.\n",
                                  lammpsNtypes, elementMap.size()));
    }
    log << "-----------------------------------------"
           "--------------------------------------\n";
    log << "CAUTION: Please ensure that this mapping between LAMMPS\n";
    log << "         atom types and NNP elements is consistent:\n";
    log << "\n";
    log << "---------------------------\n";
    log << "LAMMPS type  |  NNP element\n";
    log << "---------------------------\n";
    for (int i = 0; i < lammpsNtypes; ++i)
    {
        log << strpr("%11d <-> %2s (%3zu)\n",
                     i + 1,
                     elementMap[(size_t)i].c_str(),
                     elementMap.atomicNumber((size_t)i));
    }
    log << "---------------------------\n";
    log << "\n";
    log << "NNP setup for LAMMPS completed.\n";

    log << "*****************************************"
           "**************************************\n";

    structure.setElementMap(elementMap);

    initialized = true;
}

void InterfaceLammps::setLocalAtoms(int              numAtomsLocal,
                                    int const* const atomTag,
                                    int const* const atomType)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        structure.numAtomsPerElement[i] = 0;
    }
    structure.index                          = myRank;
    structure.numAtoms                       = numAtomsLocal;
    structure.hasNeighborList                = false;
    structure.hasSymmetryFunctions           = false;
    structure.hasSymmetryFunctionDerivatives = false;
    structure.energy                         = 0.0;
    structure.atoms.resize(numAtomsLocal);
    for (size_t i = 0; i < structure.atoms.size(); i++)
    {
        Atom& a = structure.atoms[i];
        a.free(true);
        a.index                          = i;
        a.indexStructure                 = myRank;
        a.tag                            = atomTag[i];
        a.element                        = atomType[i] - 1;
        a.numNeighbors                   = 0;
        a.hasSymmetryFunctions           = false;
        a.hasSymmetryFunctionDerivatives = false;
        a.neighbors.clear();
        a.numNeighborsPerElement.clear();
        a.numNeighborsPerElement.resize(numElements, 0);
        structure.numAtomsPerElement[a.element]++;
    }

    return;
}

void InterfaceLammps::addNeighbor(int    i,
                                  int    j,
                                  int    tag,
                                  int    type,
                                  double dx,
                                  double dy,
                                  double dz,
                                  double d2)
{
    Atom& a = structure.atoms[i];
    a.numNeighbors++;
    a.neighbors.push_back(Atom::Neighbor());
    a.numNeighborsPerElement.at(type - 1)++;
    Atom::Neighbor& n = a.neighbors.back();
    n.index   = j;
    n.tag     = tag;
    n.element = type - 1;
    n.dr[0]   = dx * cflength;
    n.dr[1]   = dy * cflength;
    n.dr[2]   = dz * cflength;
    n.d       = sqrt(d2) * cflength;
    if (normalize)
    {
        n.dr[0] *= convLength;
        n.dr[1] *= convLength;
        n.dr[2] *= convLength;
        n.d     *= convLength;
    }

    return;
}

void InterfaceLammps::process()
{
#ifdef NOSFGROUPS
    calculateSymmetryFunctions(structure, true);
#else
    calculateSymmetryFunctionGroups(structure, true);
#endif
    calculateAtomicNeuralNetworks(structure, true);
    calculateEnergy(structure);
    if (normalize)
    {
        structure.energy = physicalEnergy(structure, false);
    }
    addEnergyOffset(structure, false);

    return;
}

double InterfaceLammps::getMaxCutoffRadius() const
{
    if (normalize) return maxCutoffRadius / convLength / cflength;
    else return maxCutoffRadius / cflength;
}

double InterfaceLammps::getEnergy() const
{
    return structure.energy / cfenergy;
}

double InterfaceLammps::getAtomicEnergy(int index) const
{
    Atom const& a = structure.atoms.at(index);

    if (normalize) return physicalEnergy(a.energy) / cfenergy;
    else return a.energy / cfenergy;
}

void InterfaceLammps::getForces(double* const* const& atomF) const
{
    double const cfforce = cflength / cfenergy;
    double convForce = 1.0;
    if (normalize)
    {
        convForce = convLength / convEnergy;
    }

    // Loop over all local atoms. Neural network and Symmetry function
    // derivatives are saved in the dEdG arrays of atoms and dGdr arrays of
    // atoms and their neighbors. These are now summed up to the force
    // contributions of local and ghost atoms.
    Atom const* a = NULL;

//#ifdef _OPENMP
//    #pragma omp parallel for private (a)
//#endif
    for (size_t i =  0; i < structure.atoms.size(); ++i)
    {
    //for (vector<Atom>::const_iterator a = structure.atoms.begin();
    //     a != structure.atoms.end(); ++a)
    //{
        // Set pointer to atom.
        a = &(structure.atoms.at(i));

        // Temporarily save the number of symmetry functions of this atom.
        size_t const numSymmetryFunctions = a->numSymmetryFunctions;
        // Loop over all neighbor atoms. Some are local, some are ghost atoms.
        for (vector<Atom::Neighbor>::const_iterator n = a->neighbors.begin();
             n != a->neighbors.end(); ++n)
        {
            // Temporarily save the neighbor index. Note: this is the index for
            // the LAMMPS force array.
            size_t const in = n->index;
            // Now loop over all symmetry functions and add force contributions
            // (local + ghost atoms).
            for (size_t s = 0; s < numSymmetryFunctions; ++s)
            {
                double const dEdG = a->dEdG[s] * cfforce * convForce;
                double const* const dGdr = n->dGdr[s].r;
//#ifdef _OPENMP
//                #pragma omp atomic update
//#endif
                atomF[in][0] -= dEdG * dGdr[0];
//#ifdef _OPENMP
//                #pragma omp atomic update
//#endif
                atomF[in][1] -= dEdG * dGdr[1];
//#ifdef _OPENMP
//                #pragma omp atomic update
//#endif
                atomF[in][2] -= dEdG * dGdr[2];
            }
        }
        // Temporarily save the atom index. Note: this is the index for
        // the LAMMPS force array.
        size_t const ia = a->index;
        // Loop over all symmetry functions and add force contributions (local
        // atoms).
        for (size_t s = 0; s < numSymmetryFunctions; ++s)
        {
            double const dEdG = a->dEdG[s] * cfforce * convForce;
            double const* const dGdr = a->dGdr[s].r;
//#ifdef _OPENMP
//            #pragma omp atomic update
//#endif
            atomF[ia][0] -= dEdG * dGdr[0];
//#ifdef _OPENMP
//            #pragma omp atomic update
//#endif
            atomF[ia][1] -= dEdG * dGdr[1];
//#ifdef _OPENMP
//            #pragma omp atomic update
//#endif
            atomF[ia][2] -= dEdG * dGdr[2];
        }
    }

    return;
}

long InterfaceLammps::getEWBufferSize() const
{
    long bs = 0;
#ifndef NOMPI
    int ss = 0; // size_t size.
    int ds = 0; // double size.
    int cs = 0; // char size.
    MPI_Pack_size(1, MPI_SIZE_T, MPI_COMM_WORLD, &ss);
    MPI_Pack_size(1, MPI_DOUBLE, MPI_COMM_WORLD, &ds);
    MPI_Pack_size(1, MPI_CHAR  , MPI_COMM_WORLD, &cs);

    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        map<size_t, SymmetryFunctionStatistics::Container> const& m =
                                                           it->statistics.data;
        bs += ss; // n.
        for (map<size_t, SymmetryFunctionStatistics::Container>::const_iterator
             it2 = m.begin(); it2 != m.end(); ++it2)
        {
            bs += ss; // index   (it2->first).
            bs += ss; // countEW (it2->second.countEW).
            bs += ds; // Gmin    (it2->second.Gmin).
            bs += ds; // Gmax    (it2->second.Gmax).
            size_t countEW = it2->second.countEW;
            bs += countEW * ss; // indexStructureEW.
            bs += countEW * ss; // indexAtomEW.
            bs += countEW * ds; // valueEW.
        }
    }
#endif
    return bs;
}

void InterfaceLammps::fillEWBuffer(char* const& buf, int bs) const
{
#ifndef NOMPI
    int p = 0;
    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        map<size_t, SymmetryFunctionStatistics::Container> const& m =
                                                           it->statistics.data;
        size_t n = m.size();
        MPI_Pack(&(n), 1, MPI_SIZE_T, buf, bs, &p, MPI_COMM_WORLD);
        for (map<size_t, SymmetryFunctionStatistics::Container>::const_iterator
             it2 = m.begin(); it2 != m.end(); ++it2)
        {
            MPI_Pack(&(it2->first                          ),       1, MPI_SIZE_T, buf, bs, &p, MPI_COMM_WORLD);
            size_t countEW = it2->second.countEW;
            MPI_Pack(&(countEW                             ),       1, MPI_SIZE_T, buf, bs, &p, MPI_COMM_WORLD);
            MPI_Pack(&(it2->second.Gmin                    ),       1, MPI_DOUBLE, buf, bs, &p, MPI_COMM_WORLD);
            MPI_Pack(&(it2->second.Gmax                    ),       1, MPI_DOUBLE, buf, bs, &p, MPI_COMM_WORLD);
            MPI_Pack(&(it2->second.indexStructureEW.front()), countEW, MPI_SIZE_T, buf, bs, &p, MPI_COMM_WORLD);
            MPI_Pack(&(it2->second.indexAtomEW.front()     ), countEW, MPI_SIZE_T, buf, bs, &p, MPI_COMM_WORLD);
            MPI_Pack(&(it2->second.valueEW.front()         ), countEW, MPI_DOUBLE, buf, bs, &p, MPI_COMM_WORLD);
        }
    }
#endif
    return;
}

void InterfaceLammps::extractEWBuffer(char const* const& buf, int bs)
{
#ifndef NOMPI
    int p = 0;
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        size_t n = 0;
        MPI_Unpack(buf, bs, &p, &(n), 1, MPI_SIZE_T, MPI_COMM_WORLD);
        for (size_t i = 0; i < n; ++i)
        {
            size_t index = 0;
            MPI_Unpack(buf, bs, &p, &(index), 1, MPI_SIZE_T, MPI_COMM_WORLD);
            SymmetryFunctionStatistics::
            Container& d = it->statistics.data[index];
            size_t countEW = 0;
            MPI_Unpack(buf, bs, &p, &(countEW                      ),       1, MPI_SIZE_T, MPI_COMM_WORLD);
            MPI_Unpack(buf, bs, &p, &(d.Gmin                       ),       1, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack(buf, bs, &p, &(d.Gmax                       ),       1, MPI_DOUBLE, MPI_COMM_WORLD);
            // indexStructureEW.
            d.indexStructureEW.resize(d.countEW + countEW);
            MPI_Unpack(buf, bs, &p, &(d.indexStructureEW[d.countEW]), countEW, MPI_SIZE_T, MPI_COMM_WORLD);
            // indexAtomEW.
            d.indexAtomEW.resize(d.countEW + countEW);
            MPI_Unpack(buf, bs, &p, &(d.indexAtomEW[d.countEW]     ), countEW, MPI_SIZE_T, MPI_COMM_WORLD);
            // valueEW.
            d.valueEW.resize(d.countEW + countEW);
            MPI_Unpack(buf, bs, &p, &(d.valueEW[d.countEW]         ), countEW, MPI_DOUBLE, MPI_COMM_WORLD);

            d.countEW += countEW;
        }
    }
#endif
    return;
}

void InterfaceLammps::writeExtrapolationWarnings()
{
    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        vector<string> vs = it->statistics.getExtrapolationWarningLines();
        for (vector<string>::const_iterator it2 = vs.begin();
             it2 != vs.end(); ++it2)
        {
            log << (*it2);
        }
    }

    return;
}

void InterfaceLammps::clearExtrapolationWarnings()
{
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->statistics.clear();
    }

    return;
}
