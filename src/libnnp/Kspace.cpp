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
//
#include "Kspace.h"

using namespace std;
using namespace nnp;

Kpoint::Kpoint() : k    (Vec3D()),
                   knorm(0.0    ),
                   coeff(0.0    )
{
}

KspaceGrid::KspaceGrid() : eta   (0.0),
                           rcut  (0.0),
                           volume(0.0),
                           pre   (0.0)
{
}

double KspaceGrid::setup(Vec3D box[3], double precision, size_t numAtoms)
{
    volume = fabs(box[0] * (box[1].cross(box[2])));
    pre = 2.0 * M_PI / volume;

    kbox[0] = pre * box[1].cross(box[2]);
    kbox[1] = pre * box[2].cross(box[0]);
    kbox[2] = pre * box[0].cross(box[1]);

    eta = 1.0 / sqrt(2.0 * M_PI);
    // Regular Ewald eta.
    if (numAtoms != 0) eta *= pow(volume * volume / numAtoms, 1.0 / 6.0);
    // Matrix version eta.
    else eta *= pow(volume, 1.0 / 3.0);

    // Reciprocal cutoff radius.
    rcut = sqrt(-2.0 * log(precision)) / eta;

    // Compute box copies required in each direction.
    calculatePbcCopies(rcut);

    // TODO: Create k-points here.

    // Return real space cutoff radius.
    return sqrt(-2.0 * log(precision)) * eta;
}

void KspaceGrid::calculatePbcCopies(double cutoffRadius)
{
    Vec3D axb;
    Vec3D axc;
    Vec3D bxc;

    axb = kbox[0].cross(kbox[1]).normalize();
    axc = kbox[0].cross(kbox[2]).normalize();
    bxc = kbox[1].cross(kbox[2]).normalize();

    double proja = fabs(kbox[0] * bxc);
    double projb = fabs(kbox[1] * axc);
    double projc = fabs(kbox[2] * axb);

    n[0] = 0;
    n[1] = 0;
    n[2] = 0;
    while (n[0] * proja <= cutoffRadius) n[0]++;
    while (n[1] * projb <= cutoffRadius) n[1]++;
    while (n[2] * projc <= cutoffRadius) n[2]++;

    return;
}

