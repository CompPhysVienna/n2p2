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
#include <iostream>
#include <fstream>

using namespace std;
using namespace nnp;


Kvector::Kvector() : k     (Vec3D()),
                     knorm2(0.0    ),
                     coeff (0.0    )
{
}

Kvector::Kvector(Vec3D v) : k     (v       ),
                            knorm2(v.norm()),
                            coeff (0.0     )
{
}

KspaceGrid::KspaceGrid() : eta   (0.0),
                           kCut  (0.0),
                           volume(0.0),
                           pre   (0.0)
{
}

void KspaceGrid::setup(Vec3D box[3], EwaldSetup& ewaldSetup)
{
    volume = fabs(box[0] * (box[1].cross(box[2])));
    pre = 2.0 * M_PI / volume;

    kbox[0] = pre * box[1].cross(box[2]);
    kbox[1] = pre * box[2].cross(box[0]);
    kbox[2] = pre * box[0].cross(box[1]);

    eta = ewaldSetup.params.eta;
    kCut = ewaldSetup.params.kCut;

    // Compute box copies required in each direction.
    calculatePbcCopies(kCut);

    // Compute k-grid (only half sphere because of symmetry)
    for (int i = 0; i <= n[0]; ++i)
    {
        int sj = -n[1];
        if (i == 0) sj = 0;
        for (int j = sj; j <= n[1]; ++j)
        {
            int sk = -n[2];
            if (i == 0 && j == 0) sk = 0;
            for (int k = sk; k <= n[2]; ++k)
            {
                if (i == 0 && j == 0 && k == 0) continue;
                Vec3D kv = i * kbox[0] + j * kbox[1] + k * kbox[2];
                double knorm2 = kv.norm2();
                if (kv.norm2() < kCut * kCut)
                {
                    kvectors.push_back(kv);
                    kvectors.back().knorm2 = knorm2; // TODO: Really necessary?
                    kvectors.back().coeff = exp(-0.5 * eta * eta * knorm2)
                                          * 2.0 * pre / knorm2;
                }
            }
        }
    }
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
