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

#include "Kspace.h"
#include "Structure.h"
#include "Vec3D.h"
#include "utility.h"
#include <Eigen/Dense> // MatrixXd, VectorXd
#include <algorithm>   // std::max
#include <cmath>       // fabs, erf
#include <cstdlib>     // atof
#include <limits>      // std::numeric_limits
#include <stdexcept>   // std::runtime_error
#include <string>      // std::getline
#include <iostream>

using namespace std;
using namespace nnp;
using namespace Eigen;


Structure::Structure() :
    isPeriodic                    (false     ),
    isTriclinic                   (false     ),
    hasNeighborList               (false     ),
    NeighborListIsSorted          (false     ),
    hasSymmetryFunctions          (false     ),
    hasSymmetryFunctionDerivatives(false     ),
    index                         (0         ),
    numAtoms                      (0         ),
    numElements                   (0         ),
    numElementsPresent            (0         ),
    energy                        (0.0       ),
    energyRef                     (0.0       ),
    energyShort                   (0.0       ),
    energyElec                    (0.0       ),
    hasCharges                    (false     ),
    charge                        (0.0       ),
    chargeRef                     (0.0       ),
    volume                        (0.0       ),
    maxCutoffRadiusOverall        (0.0       ),
    sampleType                    (ST_UNKNOWN),
    comment                       (""        ),
    hasAMatrix                    (false     )
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
            if (splitLine.size() > 1)
            {
                for (vector<string>::const_iterator word =
                     splitLine.begin() + 1; word != splitLine.end(); ++word)
                {
                    if      (*word == "set=train") sampleType = ST_TRAINING;
                    else if (*word == "set=test") sampleType = ST_TEST;
                    else
                    {
                        throw runtime_error("ERROR: Unknown keyword in "
                                            "structure specification, check "
                                            "\"begin\" arguments.\n");
                    }
                }
            }
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
            atoms.back().tag            = numAtoms; // Implicit conversion!
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
            if (isPeriodic &&
                abs(chargeRef) > 10*std::numeric_limits<double>::epsilon())
            {
                throw runtime_error("ERROR: In structure with index "
                                    + to_string(index)
                                    + "; if PBCs are applied, the\n"
                                    "simulation cell has to be neutrally "
                                    "charged.\n");
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

    if (isPeriodic) remap();

    return;
}

void Structure::calculateMaxCutoffRadiusOverall(
                                            EwaldSetup& ewaldSetup,
                                            double rcutScreen,
                                            double maxCutoffRadius)
{
    maxCutoffRadiusOverall = max(rcutScreen, maxCutoffRadius);
    if (isPeriodic)
    {
        ewaldSetup.calculateParameters(volume, numAtoms);
        maxCutoffRadiusOverall = max(maxCutoffRadiusOverall,
                                     ewaldSetup.params.rCut);
    }
}

void Structure::calculateNeighborList(
                                        double      cutoffRadius,
                                        bool        sortByDistance)
{
    cutoffRadius = max( cutoffRadius, maxCutoffRadiusOverall );
    //cout << "CUTOFF: " << cutoffRadius << "\n";

    if (isPeriodic)
    {
        calculatePbcCopies(cutoffRadius, pbc);

        // Use square of cutoffRadius (faster).
        cutoffRadius *= cutoffRadius;

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < numAtoms; i++)
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
                                        back().tag = j; // Implicit conversion!
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

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < numAtoms; i++)
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
                        atoms[i].neighbors.back().tag     = j; // Impl. conv.!
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

    if (sortByDistance) sortNeighborList();

    return;
}

void Structure::calculateNeighborList(double                cutoffRadius,
                                      std::vector<
                                      std::vector<double>>& cutoffs)
{
    calculateNeighborList(cutoffRadius, true);
    setupNeighborCutoffMap(cutoffs);
}

void Structure::sortNeighborList()
{
#ifdef _OPENMP
        #pragma omp parallel for
#endif
    for (size_t i = 0; i < numAtoms; ++i)
    {
        sort(atoms[i].neighbors.begin(), atoms[i].neighbors.end());
        //TODO: maybe sort neighborsUnique too?
        atoms[i].NeighborListIsSorted = true;
    }
    NeighborListIsSorted = true;
}

void Structure::setupNeighborCutoffMap(vector<
                                       vector<double>> cutoffs)
{
    if ( !NeighborListIsSorted )
        throw runtime_error("NeighborCutoffs map needs a sorted neighbor list");

    for ( auto& elementCutoffs : cutoffs )
    {
        if ( maxCutoffRadiusOverall > 0 &&
             !vectorContains(elementCutoffs, maxCutoffRadiusOverall))
        {
            elementCutoffs.push_back(maxCutoffRadiusOverall);
        }

        if ( !is_sorted(elementCutoffs.begin(), elementCutoffs.end()) )
        {
            sort(elementCutoffs.begin(), elementCutoffs.end());
        }
    }

    // Loop over all atoms in this structure and create its neighborCutoffs map
    for( auto& a : atoms )
    {
        size_t const eIndex = a.element;
        vector<double> const& elementCutoffs = cutoffs.at(eIndex);
        size_t in = 0;
        size_t ic = 0;

        while( in < a.numNeighbors && ic < elementCutoffs.size() )
        {
            Atom::Neighbor const& n = a.neighbors[in];
            // If neighbor distance is higher than current "desired cutoff"
            // then add neighbor cutoff.
            // Attention: a step in the neighbor list could jump over multiple
            // params -> don't increment neighbor index
            if( n.d > elementCutoffs[ic] )
            {
                a.neighborCutoffs.emplace( elementCutoffs.at(ic), in );
                ++ic;
            }
            else if ( in == a.numNeighbors - 1 )
            {
                a.neighborCutoffs.emplace( elementCutoffs.at(ic), a.numNeighbors );
                ++ic;
            }
            else ++in;
        }
    }
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

void Structure::calculatePbcCopies(double cutoffRadius, int (&pbc)[3])
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

// TODO: Not needed anymore, should we keep it?
bool Structure::canMinimumImageConventionBeApplied(double cutoffRadius)
{
    Vec3D axb;
    Vec3D axc;
    Vec3D bxc;

    axb = box[0].cross(box[1]).normalize();
    axc = box[0].cross(box[2]).normalize();
    bxc = box[1].cross(box[2]).normalize();

    double proj[3];
    proj[0] = fabs(box[0] * bxc);
    proj[1] = fabs(box[1] * axc);
    proj[2] = fabs(box[2] * axb);

    double minProj = *min_element(proj, proj+3);
    return (cutoffRadius < minProj / 2.0);
}

Vec3D Structure::applyMinimumImageConvention(Vec3D const& dr)
{
    Vec3D ds = invbox * dr;
    Vec3D dsNINT;

    for (size_t i=0; i<3; ++i)
    {
        dsNINT[i] = round(ds[i]);
    }
    Vec3D drMin = box * (ds - dsNINT);

    return drMin;
}

void Structure::calculateVolume()
{
    volume = fabs(box[0] * (box[1].cross(box[2])));

    return;
}

double Structure::calculateElectrostaticEnergy(
                                            EwaldSetup&              ewaldSetup,
                                            VectorXd                 hardness,
                                            MatrixXd                 gammaSqrt2,
                                            VectorXd                 sigmaSqrtPi,
                                            ScreeningFunction const& fs,
                                            double const             fourPiEps,
                                            ErfcBuf&                 erfcBuf)
{
    A.resize(numAtoms + 1, numAtoms + 1);
    A.setZero();
    VectorXd b(numAtoms + 1);
    VectorXd hardnessJ(numAtoms);
    VectorXd Q;
    erfcBuf.reset(atoms, 2);

#ifdef _OPENMP
#pragma omp parallel
    {
#endif
    if (isPeriodic)
    {
#ifdef _OPENMP
#pragma omp single
        {
#endif
        if (!NeighborListIsSorted)
        {
            throw runtime_error("ERROR: Neighbor list needs to "
                                "be sorted for Ewald summation!\n");
        }

        // Should always happen already before neighborlist construction.
        //ewaldSetup.calculateParameters(volume, numAtoms);
#ifdef _OPENMP
        }
#endif
        KspaceGrid grid;
        grid.setup(box, ewaldSetup);
        double const rcutReal = ewaldSetup.params.rCut;
        double const sqrt2eta = sqrt(2.0) * ewaldSetup.params.eta;

#ifdef _OPENMP
        #pragma omp for schedule(dynamic)
#endif
        for (size_t i = 0; i < numAtoms; ++i)
        {
            Atom const &ai = atoms.at(i);
            size_t const ei = ai.element;

            // diagonal including self interaction
            A(i, i) += hardness(ei)
                       + (1.0 / sigmaSqrtPi(ei) - 2 / (sqrt2eta * sqrt(M_PI)))
                         / fourPiEps;

            hardnessJ(i) = hardness(ei);
            b(i) = -ai.chi;

            // real part
            size_t const numNeighbors = ai.getStoredMinNumNeighbors(rcutReal);
            for (size_t k = 0; k < numNeighbors; ++k)
            {
                auto const &n = ai.neighbors[k];
                size_t j = n.tag;
                if (j < i) continue;

                double const rij = n.d;
                size_t const ej = n.element;

                double erfcSqrt2Eta = erfcBuf.getf(i, k, 0, rij / sqrt2eta);
                double erfcGammaSqrt2 =
                            erfcBuf.getf(i, k, 1, rij / gammaSqrt2(ei, ej));

                A(i, j) += (erfcSqrt2Eta - erfcGammaSqrt2) / (rij * fourPiEps);
            }

            // reciprocal part
            //for (size_t j = i + 1; j < numAtoms; ++j)
            for (size_t j = i; j < numAtoms; ++j)
            {
                Atom const &aj = atoms.at(j);
                for (auto const &gv: grid.kvectors)
                {
                    // Multiply by 2 because our grid is only a half-sphere
                    // Vec3D const dr = applyMinimumImageConvention(ai.r - aj.r);
                    Vec3D const dr = ai.r - aj.r;
                    A(i, j) += 2 * gv.coeff * cos(gv.k * dr) / fourPiEps;
                    //A(i, j) += 2 * gv.coeff * cos(gv.k * (ai.r - aj.r));
                }
                A(j, i) = A(i, j);
            }
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (size_t i = 0; i < numAtoms; ++i)
        {
            Atom const &ai = atoms.at(i);
            size_t const ei = ai.element;

            A(i, i) = hardness(ei)
                      + 1.0 / (sigmaSqrtPi(ei) * fourPiEps);
            hardnessJ(i) = hardness(ei);
            b(i) = -ai.chi;
            for (size_t j = i + 1; j < numAtoms; ++j)
            {
                Atom const &aj = atoms.at(j);
                size_t const ej = aj.element;
                double const rij = (ai.r - aj.r).norm();

                A(i, j) = erf(rij / gammaSqrt2(ei, ej)) / (rij * fourPiEps);
                A(j, i) = A(i, j);

            }
        }
    }
#ifdef _OPENMP
#pragma omp single
    {
#endif
    A.col(numAtoms).setOnes();
    A.row(numAtoms).setOnes();
    A(numAtoms, numAtoms) = 0.0;
    hasAMatrix = true;
    b(numAtoms) = chargeRef;
#ifdef _OPENMP
    }
#endif
    //TODO: sometimes only recalculation of A matrix is needed, because
    //      Qs are stored.
    Q = A.colPivHouseholderQr().solve(b);
#ifdef _OPENMP
    #pragma omp for nowait
#endif
    for (size_t i = 0; i < numAtoms; ++i)
    {
        atoms.at(i).charge = Q(i);
    }
#ifdef _OPENMP
    } // end of parallel region
#endif

    lambda = Q(numAtoms);
    hasCharges = true;
    double error = (A * Q - b).norm() / b.norm();

    // We need matrix E not A, which only differ by the hardness terms along the diagonal
    energyElec = 0.5 * Q.head(numAtoms).transpose()
               * (A.topLeftCorner(numAtoms, numAtoms) -
                MatrixXd(hardnessJ.asDiagonal())) * Q.head(numAtoms);
    energyElec += calculateScreeningEnergy(gammaSqrt2, sigmaSqrtPi, fs, fourPiEps);

    return error;
}

double Structure::calculateScreeningEnergy(
                                        Eigen::MatrixXd          gammaSqrt2,
                                        VectorXd                 sigmaSqrtPi,
                                        ScreeningFunction const& fs,
                                        double const             fourPiEps)

{
    double energyScreen = 0;
    double const rcutScreen = fs.getOuter();

#ifdef _OPENMP
    #pragma omp parallel
    {
        double localEnergyScreen = 0;
#endif
        if (isPeriodic)
        {
#ifdef _OPENMP
            #pragma omp for schedule(dynamic)
#endif
            for (size_t i = 0; i < numAtoms; ++i)
            {
                Atom const& ai = atoms.at(i);
                size_t const ei = ai.element;
                double const Qi = ai.charge;
#ifdef _OPENMP
                localEnergyScreen +=  -Qi * Qi
                                   / (2 * sigmaSqrtPi(ei) * fourPiEps);
#else
                energyScreen +=  -Qi * Qi
                              / (2 * sigmaSqrtPi(ei) * fourPiEps);
#endif

                size_t const numNeighbors = ai.getStoredMinNumNeighbors(rcutScreen);
                for (size_t k = 0; k < numNeighbors; ++k)
                {
                    auto const& n = ai.neighbors[k];
                    size_t const j = n.tag;
                    if (j < i) continue;
                    double const rij = n.d;
                    size_t const ej = n.element;
                    //TODO: Maybe add charge to neighbor class?
                    double const Qj = atoms.at(j).charge;
#ifdef _OPENMP
                    localEnergyScreen += Qi * Qj * erf(rij / gammaSqrt2(ei, ej))
                                    * (fs.f(rij) - 1) / (rij * fourPiEps);
#else
                    energyScreen += Qi * Qj * erf(rij / gammaSqrt2(ei, ej))
                                    * (fs.f(rij) - 1) / (rij * fourPiEps);
#endif

               }
            }
        }
        else
        {
#ifdef _OPENMP
            #pragma omp for schedule(dynamic)
#endif
            for (size_t i = 0; i < numAtoms; ++i)
            {
                Atom const& ai = atoms.at(i);
                size_t const ei = ai.element;
                double const Qi = ai.charge;
#ifdef _OPENMP
                localEnergyScreen +=  -Qi * Qi
                                   / (2 * sigmaSqrtPi(ei) * fourPiEps);
#else
                energyScreen +=  -Qi * Qi
                              / (2 * sigmaSqrtPi(ei) * fourPiEps);
#endif

                for (size_t j = i + 1; j < numAtoms; ++j)
                {
                    Atom const& aj = atoms.at(j);
                    double const Qj = aj.charge;
                    double const rij = (ai.r - aj.r).norm();
                    if ( rij < rcutScreen )
                    {
#ifdef _OPENMP
                        localEnergyScreen += Qi * Qj * A(i, j) * (fs.f(rij) - 1);
#else
                        energyScreen += Qi * Qj * A(i, j) * (fs.f(rij) - 1);
#endif
                    }
                }
            }
        }
#ifdef _OPENMP
        #pragma omp atomic
        energyScreen += localEnergyScreen;
    }
#endif
    //cout << "energyScreen = " << energyScreen << endl;
    return energyScreen;
}


void Structure::calculateDAdrQ(
                            EwaldSetup&  ewaldSetup,
                            MatrixXd     gammaSqrt2,
                            double const fourPiEps,
                            ErfcBuf&     erfcBuf)
{

#ifdef _OPENMP
    vector<Vec3D> dAdrQDiag(numAtoms);
    #pragma omp parallel
    {

    #pragma omp declare reduction(vec_Vec3D_plus : std::vector<Vec3D> : \
                  std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<Vec3D>())) \
                  initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
    #pragma omp for
#endif
    // TODO: This initialization loop could probably be avoid, maybe use
    // default constructor?
    for (size_t i = 0; i < numAtoms; ++i)
    {
        Atom &ai = atoms.at(i);
        // dAdrQ(numAtoms+1,:) entries are zero
        ai.dAdrQ.resize(numAtoms + 1);
    }

    if (isPeriodic)
    {
        // TODO: We need same Kspace grid as in calculateScreeningEnergy, should
        // we cache it for reuse? Note that we can't calculate dAdrQ already in
        // the loops of calculateElectrostaticEnergy because at this point we don't
        // have the charges.

        KspaceGrid grid;
        grid.setup(box, ewaldSetup);
        double rcutReal = ewaldSetup.params.rCut;
        double const sqrt2eta = sqrt(2.0) * ewaldSetup.params.eta;

#ifdef _OPENMP
        // This way of parallelization slightly changes the result because of
        // numerical errors.
        #pragma omp for reduction(vec_Vec3D_plus : dAdrQDiag) schedule(dynamic)
#endif
        for (size_t i = 0; i < numAtoms; ++i)
        {
            Atom &ai = atoms.at(i);
            size_t const ei = ai.element;
            double const Qi = ai.charge;

            // real part
            size_t const numNeighbors = ai.getStoredMinNumNeighbors(rcutReal);
            for (size_t k = 0; k < numNeighbors; ++k)
            {
                auto const &n = ai.neighbors[k];
                size_t j = n.tag;
                //if (j < i) continue;
                if (j <= i) continue;

                double const rij = n.d;
                Atom &aj = atoms.at(j);
                size_t const ej = aj.element;
                double const Qj = aj.charge;

                double erfcSqrt2Eta = erfcBuf.getf(i, k, 0, rij / sqrt2eta);
                double erfcGammaSqrt2 =
                        erfcBuf.getf(i, k, 1, rij / gammaSqrt2(ei, ej));

                Vec3D dAijdri;
                dAijdri = n.dr / pow(rij, 2)
                          * (2 / sqrt(M_PI) * (-exp(-pow(rij / sqrt2eta, 2))
                          / sqrt2eta + exp(-pow(rij / gammaSqrt2(ei, ej), 2))
                          / gammaSqrt2(ei, ej)) - 1 / rij
                          * (erfcSqrt2Eta - erfcGammaSqrt2));
                dAijdri /= fourPiEps;
                // Make use of symmetry: dA_{ij}/dr_i = dA_{ji}/dr_i
                // = -dA_{ji}/dr_j = -dA_{ij}/dr_j
                ai.dAdrQ[i] += dAijdri * Qj;
                ai.dAdrQ[j] += dAijdri * Qi;
                aj.dAdrQ[i] -= dAijdri * Qj;
#ifdef _OPENMP
                dAdrQDiag[j] -= dAijdri * Qi;
#else
                aj.dAdrQ[j] -= dAijdri * Qi;
#endif
            }

            // reciprocal part
            for (size_t j = i + 1; j < numAtoms; ++j)
            {
                Atom &aj = atoms.at(j);
                double const Qj = aj.charge;
                Vec3D dAijdri;
                for (auto const &gv: grid.kvectors)
                {
                    // Multiply by 2 because our grid is only a half-sphere
                    //Vec3D const dr = applyMinimumImageConvention(ai.r - aj.r);
                    Vec3D const dr = ai.r - aj.r;
                    dAijdri -= 2 * gv.coeff * sin(gv.k * dr) * gv.k;
                }
                dAijdri /= fourPiEps;

                ai.dAdrQ[i] += dAijdri * Qj;
                ai.dAdrQ[j] += dAijdri * Qi;
                aj.dAdrQ[i] -= dAijdri * Qj;
#ifdef _OPENMP
                dAdrQDiag[j] -= dAijdri * Qi;
#else
                aj.dAdrQ[j] -= dAijdri * Qi;
#endif
            }
        }

    } else
    {
#ifdef _OPENMP
        #pragma omp for reduction(vec_Vec3D_plus : dAdrQDiag) schedule(dynamic)
#endif
        for (size_t i = 0; i < numAtoms; ++i)
        {
            Atom &ai = atoms.at(i);
            size_t const ei = ai.element;
            double const Qi = ai.charge;

            for (size_t j = i + 1; j < numAtoms; ++j)
            {
                Atom &aj = atoms.at(j);
                size_t const ej = aj.element;
                double const Qj = aj.charge;

                double rij = (ai.r - aj.r).norm();
                Vec3D dAijdri;
                dAijdri = (ai.r - aj.r) / pow(rij, 2)
                          * (2 / (sqrt(M_PI) * gammaSqrt2(ei, ej))
                             * exp(-pow(rij / gammaSqrt2(ei, ej), 2))
                             - erf(rij / gammaSqrt2(ei, ej)) / rij);
                dAijdri /= fourPiEps;
                // Make use of symmetry: dA_{ij}/dr_i = dA_{ji}/dr_i
                // = -dA_{ji}/dr_j = -dA_{ij}/dr_j
                ai.dAdrQ[i] += dAijdri * Qj;
                ai.dAdrQ[j] = dAijdri * Qi;
                aj.dAdrQ[i] = -dAijdri * Qj;
#ifdef _OPENMP
                dAdrQDiag[j] -= dAijdri * Qi;
#else
                aj.dAdrQ[j] -= dAijdri * Qi;
#endif
            }
        }
    }
#ifdef _OPENMP
    #pragma omp for
    for (size_t i = 0; i < numAtoms; ++i)
    {
        Atom& ai = atoms[i];
        ai.dAdrQ[i] += dAdrQDiag[i];
    }
    }
#endif
    return;
}

void Structure::calculateDQdChi(vector<Eigen::VectorXd> &dQdChi)
{
    dQdChi.clear();
    dQdChi.reserve(numAtoms);
    for (size_t i = 0; i < numAtoms; ++i)
    {
        // Including Lagrange multiplier equation.
        VectorXd b(numAtoms+1);
        b.setZero();
        b(i) = -1.;
        dQdChi.push_back(A.colPivHouseholderQr().solve(b).head(numAtoms));
    }
    return;
}

void Structure::calculateDQdJ(vector<Eigen::VectorXd> &dQdJ)
{
    dQdJ.clear();
    dQdJ.reserve(numElements);
    for (size_t i = 0; i < numElements; ++i)
    {
        // Including Lagrange multiplier equation.
        VectorXd b(numAtoms+1);
        b.setZero();
        for (size_t j = 0; j < numAtoms; ++j)
        {
            Atom const &aj = atoms.at(j);
            if (i == aj.element) b(j) = -aj.charge;
        }
        dQdJ.push_back(A.colPivHouseholderQr().solve(b).head(numAtoms));
    }
    return;
}

void Structure::calculateDQdr(  vector<size_t> const&   atomIndices,
                                vector<size_t> const&   compIndices,
                                double const            maxCutoffRadius,
                                vector<Element> const&  elements)
{
    if (atomIndices.size() != compIndices.size())
        throw runtime_error("ERROR: In calculation of dQ/dr both atom index and"
                            " component index must be specified.");
    for (size_t i = 0; i < atomIndices.size(); ++i)
    {
        Atom& a = atoms.at(atomIndices[i]);
        if ( a.dAdrQ.size() == 0 )
            throw runtime_error("ERROR: dAdrQ needs to be calculated before "
                                "calculating dQdr");
        a.dQdr.resize(numAtoms);

        // b stores (-dChidr - dAdrQ), last element for charge conservation.
        VectorXd b(numAtoms + 1);
        b.setZero();
        for (size_t j = 0; j < numAtoms; ++j)
        {
            Atom const& aj = atoms.at(j);

#ifndef N2P2_FULL_SFD_MEMORY
            vector<vector<size_t> > const *const tableFull
                    = &(elements.at(aj.element).getSymmetryFunctionTable());
#else
            vector<vector<size_t> > const *const tableFull = nullptr;
#endif
            b(j) -= aj.calculateDChidr(atomIndices[i],
                                       maxCutoffRadius,
                                       tableFull)[compIndices[i]];
            b(j) -= a.dAdrQ.at(j)[compIndices[i]];
        }
        VectorXd dQdr = A.colPivHouseholderQr().solve(b).head(numAtoms);
        for (size_t j = 0; j < numAtoms; ++j)
        {
            a.dQdr.at(j)[compIndices[i]] = dQdr(j);
        }
    }
    return;
}


void Structure::calculateElectrostaticEnergyDerivatives(
                                        Eigen::VectorXd          hardness,
                                        Eigen::MatrixXd          gammaSqrt2,
                                        VectorXd                 sigmaSqrtPi,
                                        ScreeningFunction const& fs,
                                        double const             fourPiEps)
{
    // Reset in case structure is used again (e.g. during training)
    for (Atom &ai : atoms)
    {
        ai.pEelecpr = Vec3D{};
        ai.dEelecdQ = 0.0;
    }

    double rcutScreen = fs.getOuter();
    for (size_t i = 0; i < numAtoms; ++i)
    {
        Atom& ai = atoms.at(i);
        size_t const ei = ai.element;
        double const Qi = ai.charge;

        // TODO: This loop could be reduced by making use of symmetry, j>=i or
        // so
        for (size_t j = 0; j < numAtoms; ++j)
        {
            Atom& aj = atoms.at(j);
            double const Qj = aj.charge;

            ai.pEelecpr += 0.5 * Qj * ai.dAdrQ[j];

            // Diagonal terms contain self-interaction --> screened
            if (i != j) ai.dEelecdQ += Qj * A(i,j);
            else if (isPeriodic)
            {
                ai.dEelecdQ += Qi * (A(i,i) - hardness(ei)
                                - 1 / (sigmaSqrtPi(ei) * fourPiEps));
            }
        }

        if (isPeriodic)
        {
            for (auto const& ajN : ai.neighbors)
            {
                size_t j = ajN.tag;
                Atom& aj = atoms.at(j);
                if (j < i) continue;
                double const rij = ajN.d;
                if (rij >= rcutScreen) break;

                size_t const ej = aj.element;
                double const Qj = atoms.at(j).charge;

                double erfRij = erf(rij / gammaSqrt2(ei,ej));
                double fsRij = fs.f(rij);

                // corrections due to screening
                Vec3D Tij = Qi * Qj * ajN.dr / pow(rij,2)
                                * (2 / (sqrt(M_PI) * gammaSqrt2(ei,ej))
                                * exp(- pow(rij / gammaSqrt2(ei,ej),2))
                                * (fsRij - 1) + erfRij * fs.df(rij) - erfRij
                                * (fsRij - 1) / rij);
                Tij /= fourPiEps;
                ai.pEelecpr += Tij;
                aj.pEelecpr -= Tij;

                double Sij = erfRij * (fsRij - 1) / rij;
                Sij /= fourPiEps;
                ai.dEelecdQ += Qj * Sij;
                aj.dEelecdQ += Qi * Sij;
            }
        }
        else
        {
            for (size_t j = i + 1; j < numAtoms; ++j)
            {
                Atom& aj = atoms.at(j);
                double const rij = (ai.r - aj.r).norm();
                if (rij >= rcutScreen) continue;

                size_t const ej = aj.element;
                double const Qj = atoms.at(j).charge;

                double erfRij = erf(rij / gammaSqrt2(ei,ej));
                double fsRij = fs.f(rij);

                // corrections due to screening
                Vec3D Tij = Qi * Qj * (ai.r - aj.r) / pow(rij,2)
                                * (2 / (sqrt(M_PI) * gammaSqrt2(ei,ej))
                                * exp(- pow(rij / gammaSqrt2(ei,ej),2))
                                * (fsRij - 1) + erfRij * fs.df(rij) - erfRij
                                * (fsRij - 1) / rij);
                Tij /= fourPiEps;
                ai.pEelecpr += Tij;
                aj.pEelecpr -= Tij;

                double Sij = erfRij * (fsRij - 1) / rij;
                Sij /= fourPiEps;
                ai.dEelecdQ += Qj * Sij;
                aj.dEelecdQ += Qi * Sij;
            }
        }
    }
   return;
}

VectorXd const Structure::calculateForceLambdaTotal() const
{
    VectorXd dEdQ(numAtoms+1);
    for (size_t i = 0; i < numAtoms; ++i)
    {
        Atom const& ai = atoms.at(i);
        dEdQ(i) = ai.dEelecdQ + ai.dEdG.back();
    }
    dEdQ(numAtoms) = 0;
    VectorXd const lambdaTotal = A.colPivHouseholderQr().solve(-dEdQ);
    return lambdaTotal;
}

VectorXd const Structure::calculateForceLambdaElec() const
{
    VectorXd dEelecdQ(numAtoms+1);
    for (size_t i = 0; i < numAtoms; ++i)
    {
        Atom const& ai = atoms.at(i);
        dEelecdQ(i) = ai.dEelecdQ;
    }
    dEelecdQ(numAtoms) = 0;
    VectorXd const lambdaElec = A.colPivHouseholderQr().solve(-dEelecdQ);
    return lambdaElec;
}

void Structure::remap()
{
    for (vector<Atom>::iterator it = atoms.begin(); it != atoms.end(); ++it)
    {
        remap((*it));
    }
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
                                  double convLength,
                                  double convCharge)
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
    chargeRef *= convCharge;
    charge *= convCharge;
    volume *= convLength * convLength * convLength;

    for (vector<Atom>::iterator it = atoms.begin(); it != atoms.end(); ++it)
    {
        it->toNormalizedUnits(convEnergy, convLength, convCharge);
    }

    return;
}

void Structure::toPhysicalUnits(double meanEnergy,
                                double convEnergy,
                                double convLength,
                                double convCharge)
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
    chargeRef /= convCharge;
    charge /= convCharge;
    volume /= convLength * convLength * convLength;

    for (vector<Atom>::iterator it = atoms.begin(); it != atoms.end(); ++it)
    {
        it->toPhysicalUnits(convEnergy, convLength, convCharge);
    }

    return;
}

void Structure::freeAtoms(bool all, double const maxCutoffRadius)
{
    for (vector<Atom>::iterator it = atoms.begin(); it != atoms.end(); ++it)
    {
        it->free(all, maxCutoffRadius);
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

void Structure::clearElectrostatics(bool clearDQdr)
{
    A.resize(0,0);
    hasAMatrix = false;
    for (auto& a : atoms)
    {
        vector<Vec3D>().swap(a.dAdrQ);
        if (clearDQdr)
        {
            vector<Vec3D>().swap(a.dQdr);
        }
    }
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
