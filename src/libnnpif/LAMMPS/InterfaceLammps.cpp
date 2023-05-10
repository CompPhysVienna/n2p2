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

#ifndef N2P2_NO_MPI
#include <mpi.h>
#include "mpi-extra.h"
#endif
#include "InterfaceLammps.h"
#include "Atom.h"
#include "Element.h"
#include "utility.h"
#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <iostream>
#include <limits>
//#include <stdexcept>

#define TOLCUTOFF 1.0E-2

using namespace std;
using namespace nnp;
using namespace Eigen;

InterfaceLammps::InterfaceLammps() : myRank             (0    ),
                                     initialized        (false),
                                     hasGlobalStructure (false),
                                     showew             (true ),
                                     resetew            (false),
                                     showewsum          (0    ),
                                     maxew              (0    ),
                                     cflength           (1.0  ),
                                     cfenergy           (1.0  )

{
}

void InterfaceLammps::initialize(char const* const& directory,
                                 char const* const& emap,
                                 bool               showew,
                                 bool               resetew,
                                 int                showewsum,
                                 int                maxew,
                                 double             cflength,
                                 double             cfenergy,
                                 double             lammpsCutoff,
                                 int                lammpsNtypes,
                                 int                myRank)
{
    this->emap = emap;
    this->showew = showew;
    this->resetew = resetew;
    this->showewsum = showewsum;
    this->maxew = maxew;
    this->cflength = cflength;
    this->cfenergy = cfenergy;
    this->myRank = myRank;
    log.writeToStdout = false;
    string dir(directory);
    char const separator = '/';
    if (dir.back() != separator) dir += separator;
    Mode::initialize();
    loadSettingsFile(dir + "input.nn");
    setupGeneric(dir);
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
    setupNeuralNetworkWeights(dir);

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

    log << "-----------------------------------------"
           "--------------------------------------\n";
    log << "Element mapping string from LAMMPS to n2p2: \""
           + this->emap + "\"\n";
    // Create default element mapping.
    if (this->emap == "")
    {
        if (elementMap.size() != (size_t)lammpsNtypes)
        {
            throw runtime_error(strpr("ERROR: No element mapping given and "
                                      "number of LAMMPS atom types (%d) and "
                                      "NNP elements (%zu) does not match.\n",
                                      lammpsNtypes, elementMap.size()));
        }
        log << "Element mapping string empty, creating default mapping.\n";
        for (int i = 0; i < lammpsNtypes; ++i)
        {
            mapTypeToElement[i + 1] = i;
            mapElementToType[i] = i + 1;
        }
    }
    // Read element mapping from pair_style argument.
    else
    {
        vector<string> emapSplit = split(reduce(trim(this->emap), " \t", ""),
                                         ',');
        if (elementMap.size() < emapSplit.size())
        {
            throw runtime_error(strpr("ERROR: Element mapping is inconsistent,"
                                      " NNP elements: %zu,"
                                      " emap elements: %zu.\n",
                                      elementMap.size(),
                                      emapSplit.size()));
        }
        for (string s : emapSplit)
        {
            vector<string> typeString = split(s, ':');
            if (typeString.size() != 2)
            {
                throw runtime_error(strpr("ERROR: Invalid element mapping "
                                          "string: \"%s\".\n", s.c_str()));
            }
            int t = stoi(typeString.at(0));
            if (t > lammpsNtypes)
            {
                throw runtime_error(strpr("ERROR: LAMMPS type \"%d\" not "
                                          "present, there are only %d types "
                                          "defined.\n", t, lammpsNtypes));
            }
            size_t e = elementMap[typeString.at(1)];
            mapTypeToElement[t] = e;
            mapElementToType[e] = t;
        }
    }
    log << "\n";
    log << "CAUTION: Please ensure that this mapping between LAMMPS\n";
    log << "         atom types and NNP elements is consistent:\n";
    log << "\n";
    log << "---------------------------\n";
    log << "LAMMPS type  |  NNP element\n";
    log << "---------------------------\n";
    for (int i = 1; i <= lammpsNtypes; ++i)
    {
        if (mapTypeToElement.find(i) != mapTypeToElement.end())
        {
            size_t e = mapTypeToElement.at(i);
            log << strpr("%11d <-> %2s (%3zu)\n",
                         i,
                         elementMap[e].c_str(),
                         elementMap.atomicNumber(e));
            ignoreType[i] = false;
        }
        else
        {
            log << strpr("%11d <-> --\n", i);
            ignoreType[i] = true;

        }
    }
    log << "---------------------------\n";
    log << "\n";
    log << "NNP setup for LAMMPS completed.\n";

    log << "*****************************************"
           "**************************************\n";

    structure.setElementMap(elementMap);

    initialized = true;
}

void InterfaceLammps::setGlobalStructureStatus(bool const status)
{
    hasGlobalStructure = status;
}

bool InterfaceLammps::getGlobalStructureStatus()
{
    return hasGlobalStructure;
}

void InterfaceLammps::setLocalAtoms(int              numAtomsLocal,
                                    int const* const atomType)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        structure.numAtomsPerElement[i] = 0;
    }
    structure.index                          = myRank;
    structure.numAtoms                       = 0;
    structure.hasNeighborList                = false;
    structure.hasSymmetryFunctions           = false;
    structure.hasSymmetryFunctionDerivatives = false;
    structure.energy                         = 0.0;
    structure.atoms.clear();
    indexMap.clear();
    structure.atoms.reserve(numAtomsLocal);
    indexMap.resize(numAtomsLocal, numeric_limits<size_t>::max());
    for (int i = 0; i < numAtomsLocal; i++)
    {
        if (ignoreType[atomType[i]]) continue;
        indexMap.at(i) = structure.numAtoms;
        structure.numAtoms++;
        structure.atoms.push_back(Atom());
        Atom& a = structure.atoms.back();
        a.index                          = i;
        a.indexStructure                 = myRank;
        a.element                        = mapTypeToElement[atomType[i]];
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

void InterfaceLammps::setLocalAtomPositions(double const* const* const atomPos)
{
    for (size_t i = 0; i < structure.numAtoms; ++i)
    {
        Atom& a = structure.atoms.at(i);
        a.r[0] = atomPos[i][0] * cflength;
        a.r[1] = atomPos[i][1] * cflength;
        a.r[2] = atomPos[i][2] * cflength;
        if (normalize)
        {
            a.r[0] *= convLength;
            a.r[1] *= convLength;
            a.r[2] *= convLength;
        }
    }

    return;
}

void InterfaceLammps::setLocalTags(int const* const atomTag)
{
    for (size_t i = 0; i < structure.atoms.size(); i++)
    {
        // Implicit conversion from int to int64_t!
        structure.atoms.at(i).tag = atomTag[i];
    }

    return;
}

void InterfaceLammps::setLocalTags(int64_t const* const atomTag)
{
    for (size_t i = 0; i < structure.atoms.size(); i++)
    {
        structure.atoms.at(i).tag = atomTag[i];
    }

    return;
}

void InterfaceLammps::setBoxVectors(double const* boxlo,
                                    double const* boxhi,
                                    double const  xy,
                                    double const  xz,
                                    double const  yz)
{
    structure.isPeriodic = true;

    // Box vector a
    structure.box[0][0] = boxhi[0] - boxlo[0];
    structure.box[0][1] = 0;
    structure.box[0][2] = 0;

    // Box vector b
    structure.box[1][0] = xy;
    structure.box[1][1] = boxhi[1] - boxlo[1];
    structure.box[1][2] = 0;

    // Box vector c
    structure.box[2][0] = xz;
    structure.box[2][1] = yz;
    structure.box[2][2] = boxhi[2] - boxlo[2];

    // LAMMPS may set triclinic = 1 even if the following condition is not
    // satisfied.
    if (structure.box[0][1] > numeric_limits<double>::min() ||
        structure.box[0][2] > numeric_limits<double>::min() ||
        structure.box[1][0] > numeric_limits<double>::min() ||
        structure.box[1][2] > numeric_limits<double>::min() ||
        structure.box[2][0] > numeric_limits<double>::min() ||
        structure.box[2][1] > numeric_limits<double>::min())
    {
        structure.isTriclinic = true;
    }

    for(size_t i = 0; i < 3; ++i)
    {
        structure.box[i] *= cflength;
        if (normalize) structure.box[i] *= convLength;
    }

    structure.calculateInverseBox();
    structure.calculateVolume();
    //cout << "Box vectors: \n";
    //for(size_t i = 0; i < 3; ++i)
    //{
    //    for(size_t j = 0; j < 3; ++j)
    //    {
    //        cout << structure.box[i][j] / convLength << " ";
    //    }
    //    cout << endl;
    //}

}

void InterfaceLammps::allocateNeighborlists(int const* const numneigh)
{
    for(size_t i = 0; i < structure.numAtoms; ++i)
    {
        auto& a = structure.atoms.at(i);
        a.neighbors.reserve(numneigh[i]);
    }
}

void InterfaceLammps::addNeighbor(int     i,
                                  int     j,
                                  int64_t tag,
                                  int     type,
                                  double  dx,
                                  double  dy,
                                  double  dz,
                                  double  d2)
{
    if (ignoreType[type] ||
        indexMap.at(i) == numeric_limits<size_t>::max()) return;
    Atom& a = structure.atoms[indexMap.at(i)];
    a.numNeighbors++;
    a.neighbors.push_back(Atom::Neighbor());
    a.numNeighborsPerElement.at(mapTypeToElement[type])++;
    Atom::Neighbor& n = a.neighbors.back();

    n.index = j;
    n.tag = tag;
    n.element = mapTypeToElement[type];
    n.dr[0]   = dx * cflength;
    n.dr[1]   = dy * cflength;
    n.dr[2]   = dz * cflength;
    n.d       = sqrt(d2) * cflength;
    if (normalize)
    {
        n.dr *= convLength;
        n.d  *= convLength;
    }

    return;
}


void InterfaceLammps::finalizeNeighborList()
{
    if (nnpType == NNPType::HDNNP_4G)
    {
        for (auto& a : structure.atoms)
        {
            a.hasNeighborList = true;
        }
        // Ewald summation cut-off depends on box vectors.
        structure.calculateMaxCutoffRadiusOverall(
                                            ewaldSetup,
                                            screeningFunction.getOuter(),
                                            maxCutoffRadius);
        structure.sortNeighborList();
        structure.setupNeighborCutoffMap(cutoffs);
    }

}

void InterfaceLammps::process()
{
#ifdef N2P2_NO_SF_GROUPS
    calculateSymmetryFunctions(structure, true);
#else
    calculateSymmetryFunctionGroups(structure, true);
#endif
    calculateAtomicNeuralNetworks(structure, true, "");
    if (nnpType == NNPType::HDNNP_4G)
    {
        chargeEquilibration(structure, true);
        calculateAtomicNeuralNetworks(structure, true, "short");
        ewaldSetup.logEwaldCutoffs(log, convLength * cflength);
    }
    calculateEnergy(structure);
    if (nnpType == NNPType::HDNNP_4G ||
        nnpType == NNPType::HDNNP_Q) calculateCharge(structure);
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

double InterfaceLammps::getMaxCutoffRadiusOverall()
{
    double cutoff = 0;
    if(nnpType == NNPType::HDNNP_4G)
    {
        structure.calculateMaxCutoffRadiusOverall(
                                        ewaldSetup,
                                        screeningFunction.getOuter(),
                                        maxCutoffRadius);
        cutoff = structure.maxCutoffRadiusOverall / cflength;
        if (normalize) cutoff /= convLength;
    }
    else cutoff = getMaxCutoffRadius();
    return cutoff;
}

double InterfaceLammps::getEnergy() const
{
    return structure.energy / cfenergy;
}

double InterfaceLammps::getAtomicEnergy(int index) const
{
    Atom const& a = structure.atoms.at(index);
    Element const& e = elements.at(a.element);

    if (normalize)
    {
        return (physical("energy", a.energy)
                + meanEnergy
                + e.getAtomicEnergyOffset()) / cfenergy;
    }
    else
    {
        return (a.energy + e.getAtomicEnergyOffset()) / cfenergy;
    }
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
    for (auto const& a : structure.atoms)
    {
        size_t const ia = a.index;
        Vec3D selfForce = a.calculateSelfForceShort();
        selfForce *= cfforce * convForce;
        // TODO: ia is not the right index when some atom types are excluded / ignored
        //       (see use of indexmap)
        add3DVecToArray(atomF[ia], selfForce);

#ifndef N2P2_FULL_SFD_MEMORY
        vector<vector<size_t> > const& tableFull
            = elements.at(a.element).getSymmetryFunctionTable();
#endif
        // Loop over all neighbor atoms. Some are local, some are ghost atoms.

        //for (auto const& n : a.neighbors)
        size_t const numNeighbors = a.getStoredMinNumNeighbors(maxCutoffRadius);
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (size_t k = 0; k < numNeighbors; ++k)
        {
            Atom::Neighbor const& n = a.neighbors[k];
            // Temporarily save the neighbor index. Note: this is the index for
            // the LAMMPS force array.
            size_t const in = n.index;

#ifndef N2P2_FULL_SFD_MEMORY
            Vec3D pairForce = a.calculatePairForceShort(n, &tableFull);
#else
            Vec3D pairForce = a.calculatePairForceShort(n);
#endif
            pairForce *= cfforce * convForce;
            add3DVecToArray(atomF[in], pairForce);
        }
    }

    // Comment: Will not work with multiple MPI tasks but this routine will
    //          probably be obsolete when Emir's solution is finished.
    if (nnpType == NNPType::HDNNP_4G)
    {
        Structure const& s = structure;
        VectorXd lambdaTotal = s.calculateForceLambdaTotal();

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        // OpenMP 4.0 doesn't support range based loops
        for (size_t i = 0; i < s.numAtoms; ++i)
        {
            auto const& ai = s.atoms[i];
            add3DVecToArray(atomF[i], -ai.pEelecpr * cfforce * convForce);

            for (auto const& aj : s.atoms)
            {
                size_t const j = aj.index;

#ifndef N2P2_FULL_SFD_MEMORY
                vector<vector<size_t> > const& tableFull
                        = elements.at(aj.element).getSymmetryFunctionTable();
                Vec3D dChidr = aj.calculateDChidr(ai.index,
                                                  maxCutoffRadius,
                                                  &tableFull);
#else
                Vec3D dChidr = aj.calculateDChidr(ai.index,
                                                  maxCutoffRadius);
#endif

                Vec3D remainingForce = -lambdaTotal(j) * (ai.dAdrQ[j] + dChidr);
                add3DVecToArray(atomF[i], remainingForce * cfforce * convForce);

            }
        }
    }
    return;
}

void InterfaceLammps::getCharges(double* const& atomQ) const
{
    if (nnpType != NNPType::HDNNP_4G) return;
    if (!atomQ) return;

    Structure const& s = structure;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < s.numAtoms; ++i)
    {
        atomQ[i] = s.atoms[i].charge;
    }
}

long InterfaceLammps::getEWBufferSize() const
{
    long bs = 0;
#ifndef N2P2_NO_MPI
    int ss = 0; // size_t size.
    int ds = 0; // double size.
    int cs = 0; // char size.
    MPI_Pack_size(1, MPI_SIZE_T, MPI_COMM_WORLD, &ss);
    MPI_Pack_size(1, MPI_DOUBLE, MPI_COMM_WORLD, &ds);
    MPI_Pack_size(1, MPI_CHAR  , MPI_COMM_WORLD, &cs);

    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        map<size_t, SymFncStatistics::Container> const& m
            = it->statistics.data;
        bs += ss; // n.
        for (map<size_t, SymFncStatistics::Container>::const_iterator
             it2 = m.begin(); it2 != m.end(); ++it2)
        {
            bs += ss; // index   (it2->first).
            bs += ss; // countEW (it2->second.countEW).
            bs += ss; // type    (it2->second.type).
            bs += ds; // Gmin    (it2->second.Gmin).
            bs += ds; // Gmax    (it2->second.Gmax).
            bs += ss; // element.length() (it2->second.element.length()).
            bs += (it2->second.element.length() + 1) * cs; // element.
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
#ifndef N2P2_NO_MPI
    int p = 0;
    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        map<size_t, SymFncStatistics::Container> const& m =
            it->statistics.data;
        size_t n = m.size();
        MPI_Pack((void *) &(n), 1, MPI_SIZE_T, buf, bs, &p, MPI_COMM_WORLD);
        for (map<size_t, SymFncStatistics::Container>::const_iterator
             it2 = m.begin(); it2 != m.end(); ++it2)
        {
            MPI_Pack((void *) &(it2->first                          ),       1, MPI_SIZE_T, buf, bs, &p, MPI_COMM_WORLD);
            size_t countEW = it2->second.countEW;
            MPI_Pack((void *) &(countEW                             ),       1, MPI_SIZE_T, buf, bs, &p, MPI_COMM_WORLD);
            MPI_Pack((void *) &(it2->second.type                    ),       1, MPI_SIZE_T, buf, bs, &p, MPI_COMM_WORLD);
            MPI_Pack((void *) &(it2->second.Gmin                    ),       1, MPI_DOUBLE, buf, bs, &p, MPI_COMM_WORLD);
            MPI_Pack((void *) &(it2->second.Gmax                    ),       1, MPI_DOUBLE, buf, bs, &p, MPI_COMM_WORLD);
            // it2->element
            size_t ts = it2->second.element.length() + 1;
            MPI_Pack((void *) &ts                                    ,       1, MPI_SIZE_T, buf, bs, &p, MPI_COMM_WORLD);
            MPI_Pack((void *) it2->second.element.c_str()            ,      ts, MPI_CHAR  , buf, bs, &p, MPI_COMM_WORLD);
            MPI_Pack((void *) &(it2->second.indexStructureEW.front()), countEW, MPI_SIZE_T, buf, bs, &p, MPI_COMM_WORLD);
            MPI_Pack((void *) &(it2->second.indexAtomEW.front()     ), countEW, MPI_SIZE_T, buf, bs, &p, MPI_COMM_WORLD);
            MPI_Pack((void *) &(it2->second.valueEW.front()         ), countEW, MPI_DOUBLE, buf, bs, &p, MPI_COMM_WORLD);
        }
    }
#endif
    return;
}

void InterfaceLammps::extractEWBuffer(char const* const& buf, int bs)
{
#ifndef N2P2_NO_MPI
    int p = 0;
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        size_t n = 0;
        MPI_Unpack((void *) buf, bs, &p, &(n), 1, MPI_SIZE_T, MPI_COMM_WORLD);
        for (size_t i = 0; i < n; ++i)
        {
            size_t index = 0;
            MPI_Unpack((void *) buf, bs, &p, &(index), 1, MPI_SIZE_T, MPI_COMM_WORLD);
            SymFncStatistics::Container& d = it->statistics.data[index];
            size_t countEW = 0;
            MPI_Unpack((void *) buf, bs, &p, &(countEW                      ),       1, MPI_SIZE_T, MPI_COMM_WORLD);
            MPI_Unpack((void *) buf, bs, &p, &(d.type                       ),       1, MPI_SIZE_T, MPI_COMM_WORLD);
            MPI_Unpack((void *) buf, bs, &p, &(d.Gmin                       ),       1, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Unpack((void *) buf, bs, &p, &(d.Gmax                       ),       1, MPI_DOUBLE, MPI_COMM_WORLD);
            // d.element
            size_t ts = 0;
            MPI_Unpack((void *) buf, bs, &p, &ts                             ,       1, MPI_SIZE_T, MPI_COMM_WORLD);
            char* element = new char[ts];
            MPI_Unpack((void *) buf, bs, &p, element                         ,      ts, MPI_CHAR  , MPI_COMM_WORLD);
            d.element = element;
            delete[] element;
            // indexStructureEW.
            d.indexStructureEW.resize(d.countEW + countEW);
            MPI_Unpack((void *) buf, bs, &p, &(d.indexStructureEW[d.countEW]), countEW, MPI_SIZE_T, MPI_COMM_WORLD);
            // indexAtomEW.
            d.indexAtomEW.resize(d.countEW + countEW);
            MPI_Unpack((void *) buf, bs, &p, &(d.indexAtomEW[d.countEW]     ), countEW, MPI_SIZE_T, MPI_COMM_WORLD);
            // valueEW.
            d.valueEW.resize(d.countEW + countEW);
            MPI_Unpack((void *) buf, bs, &p, &(d.valueEW[d.countEW]         ), countEW, MPI_DOUBLE, MPI_COMM_WORLD);

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

void InterfaceLammps::writeToFile(string const fileName,
                                  bool const   append)
{
    structure.toPhysicalUnits(meanEnergy, convEnergy, convLength, convCharge);
    structure.writeToFile(fileName, false, append);
    structure.toNormalizedUnits(meanEnergy, convEnergy, convLength, convCharge);
}

void InterfaceLammps::add3DVecToArray(double *const & arr, Vec3D const& v) const
{
    arr[0] += v[0];
    arr[1] += v[1];
    arr[2] += v[2];
}
