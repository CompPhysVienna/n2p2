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

#ifndef KSPACE_H
#define KSPACE_H

#include "Vec3D.h"
#include "Ewald.h"
#include <vector> // std::vector

namespace nnp
{
enum class KSPACESolver {
    /// Solver 0: Ewald summation.
    EWALD_SUM,
    /// Solver 1: PPPM.
    PPPM,
    /// Solver 2: Ewald summation in LAMMPS.
    EWALD_SUM_LAMMPS
};
class Kvector
{
public:
    /// A single k-vector (as Vec3D).
    Vec3D  k;
    /// Square of norm of k-vector.
    double knorm2;
    /// Precomputed coefficient for Ewald summation.
    double coeff;

    /// Constructor.
    Kvector();
    /// Constructor with Vec3D.
    Kvector(Vec3D v);
};

class KspaceGrid
{
public:
    /// Method for calculating the reciprocal part.
    KSPACESolver kspaceSolver;
    /// Ewald summation eta parameter.
    double               eta;
    /// Cutoff in reciprocal space.
    double               kCut;
    /// Cutoff in real space.
    double               rCut;
    /// Volume of real box.
    double               volume;
    /// Ewald sum prefactor @f$\frac{2\pi}{V}@f$.
    double               pre;
    /// Required box copies in each box vector direction.
    int                  n[3];
    /// Reciprocal box vectors.
    Vec3D                kbox[3];
    /// Vector containing all k-vectors.
    std::vector<Kvector> kvectors;
    
    /// Constructor.
    KspaceGrid();
    /** Set up reciprocal box vectors and eta.
     *
     * @param[in] box Real box vectors.
     * @param[in] precision Desired presicion for Ewald summation.
     * @param[in] halfSphere Generate only k-vectors for half sphere.
     * @param[in] numAtoms Number of atoms in system. Optional, if provided
     *                     will use "regular" Ewald optimal eta, otherwise use
     *                     "matrix" version of eta.
     *
     * @return Real space cutoff radius.
     */
    void setup(Vec3D       box[3], EwaldSetup& ewaldSetup);

private:
    /** Compute box copies in each direction.
     *
     * @param[in] cutoffRadius Cutoff radius.
     *
     * TODO: This is code copy from Structure class!
     */
    void calculatePbcCopies(double cutoffRadius);
};
    /** Compute Cut-off in real space for Ewald summation.
     *
     * @param[in] box Real box vectors.
     * @param[in] precision Desired presicion for Ewald summation.
     * @param[in] numAtoms Number of atoms in system. Optional, if provided
     *                     will use "regular" Ewald optimal eta, otherwise use
     *                     "matrix" version of eta.
     *
     * @return Real space cutoff radius.
     */
    //double getRcutReal(Vec3D box[3], double precision, size_t numAtoms = 0);
}

#endif

