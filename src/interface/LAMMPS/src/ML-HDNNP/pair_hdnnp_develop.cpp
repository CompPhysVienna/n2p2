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

#include <mpi.h>
#include <string.h>
#include "pair_hdnnp_develop.h"
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "update.h"
#include "utils.h"

#include <stdexcept>
//TODO:  remove later
#include <iostream>

#include "InterfaceLammps.h"    // n2p2 interface header

using namespace LAMMPS_NS;
using namespace std;


/* ---------------------------------------------------------------------- */

PairHDNNPDevelop::PairHDNNPDevelop(LAMMPS *lmp) : PairHDNNP(lmp) {}

/* ---------------------------------------------------------------------- */

void PairHDNNPDevelop::compute(int eflag, int vflag)
{
  if(eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;

  // Set number of local atoms and add index and element.
  interface->setLocalAtoms(atom->nlocal, atom->type);
  // Transfer tags separately. Interface::setLocalTags is overloaded internally
  // to work with both -DLAMMPS_SMALLBIG (tagint = int) and -DLAMMPS_BIGBIG
  // (tagint = int64_t)
  interface->setLocalTags(atom->tag);


  // Also set absolute atom positions.
  interface->setLocalAtomPositions(atom->x);

  // Set Box vectors if system is periodic in all 3 dims.
  if(domain->nonperiodic == 0)
  {
      interface->setBoxVectors(domain->boxlo,
                               domain->boxhi,
                               domain->xy,
                               domain->xz,
                               domain->yz);
  }

  updateNeighborlistCutoff();

  // Transfer local neighbor list to NNP interface. Has to be called after
  // setBoxVectors if structure is periodic!
  transferNeighborList(maxCutoffRadiusNeighborList);

  // Compute symmetry functions, atomic neural networks and add up energy.
  interface->process();

  // Do all stuff related to extrapolation warnings.
  if(showew == true || showewsum > 0 || maxew >= 0) {
    handleExtrapolationWarnings();
  }

  // Calculate forces of local and ghost atoms.
  interface->getForces(atom->f);

  // Transfer charges LAMMPS. Does nothing if nnpType != 4G.
  interface->getCharges(atom->q);

  // Add energy contribution to total energy.
  if (eflag_global)
     ev_tally(0,0,atom->nlocal,1,interface->getEnergy(),0.0,0.0,0.0,0.0,0.0);

  // Add atomic energy if requested (CAUTION: no physical meaning!).
  if (eflag_atom)
    for (int i = 0; i < atom->nlocal; ++i)
      eatom[i] = interface->getAtomicEnergy(i);

  // If virial needed calculate via F dot r.
  if (vflag_fdotr) virial_fdotr_compute();

    //interface.writeToFile("md_run.data", true);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairHDNNPDevelop::init_style()
{
  if (comm->nprocs > 1) {
    throw runtime_error("ERROR: Pair style \"nnp/develop\" can only be used "
                        "with a single MPI task.\n");
  }

  if(domain->dimension != 3)
    throw runtime_error("ERROR: Only 3d systems can be used!");

  if (!(domain->xperiodic == domain->yperiodic
     && domain->yperiodic == domain->zperiodic))
    throw runtime_error("ERROR: System must be either aperiodic or periodic "
                        "in all 3 dimmensions!");

  // Required for charge equilibration scheme.
  if (atom->map_style == Atom::MAP_NONE)
      throw runtime_error("ERROR: pair style requires atom map yes");

  PairHDNNP::init_style();


  maxCutoffRadiusNeighborList = maxCutoffRadius;

  interface->setGlobalStructureStatus(true);
}


double PairHDNNPDevelop::init_one(int i, int j)
{
    //cutsq[i][j] = cutsq[j][i] = pow(maxCutoffRadiusNeighborList,2);
    return maxCutoffRadiusNeighborList;
}


void PairHDNNPDevelop::transferNeighborList(double const cutoffRadius)
{
  // Transfer neighbor list to NNP.
  double rc2 = cutoffRadius * cutoffRadius;
  interface->allocateNeighborlists(list->numneigh);
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int ii = 0; ii < list->inum; ++ii) {
    int i = list->ilist[ii];
    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      int j = list->firstneigh[i][jj];
      j &= NEIGHMASK;
      double dx = atom->x[i][0] - atom->x[j][0];
      double dy = atom->x[i][1] - atom->x[j][1];
      double dz = atom->x[i][2] - atom->x[j][2];
      double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 <= rc2) {
        if (!interface->getGlobalStructureStatus())
          // atom->tag[j] will be implicitly converted to int64_t internally.
          interface->addNeighbor(i,j,atom->tag[j],atom->type[j],dx,dy,dz,d2);
        else
          interface->addNeighbor(i,j,atom->map(atom->tag[j]),atom->type[j],dx,dy,dz,d2);
      }
    }
  }
  interface->finalizeNeighborList();
}


void PairHDNNPDevelop::updateNeighborlistCutoff()
{
    double maxCutoffRadiusOverall = interface->getMaxCutoffRadiusOverall();
    if(maxCutoffRadiusOverall > maxCutoffRadiusNeighborList)
    {
        // TODO: Increase slightly to compensate for rounding errors?
        utils::logmesg(lmp, fmt::format("WARNING: Cutoff too small, need at "
                                        "least: {}\n", maxCutoffRadiusOverall));

        //maxCutoffRadiusNeighborList = maxCutoffRadiusOverall;
        //reinit();

        //mixed_flag = 1;
        //for (int i = 1; i <= atom->ntypes; i++)
        //{
        //    for (int j = i; j <= atom->ntypes; j++)
        //    {
        //        if ((i != j) && setflag[i][j]) mixed_flag = 0;
        //        cutsq[i][j] = cutsq[j][i] = pow(maxCutoffRadiusNeighborList, 2);
        //    }
        //}
        //cutforce = MAX(cutforce, maxCutoffRadiusNeighborList);

        //neighbor->init();
        //neighbor->build(0);
    }
}


