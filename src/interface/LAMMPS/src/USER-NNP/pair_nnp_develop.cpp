// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <mpi.h>
#include <string.h>
#include "pair_nnp_develop.h"
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

using namespace LAMMPS_NS;
using namespace std;


/* ---------------------------------------------------------------------- */

PairNNPDevelop::PairNNPDevelop(LAMMPS *lmp) : PairNNP(lmp) {}

/* ---------------------------------------------------------------------- */

void PairNNPDevelop::compute(int eflag, int vflag)
{
  if(eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;

  // Set number of local atoms and add index and element.
  interface.setLocalAtoms(atom->nlocal, atom->type);
  // Transfer tags separately. Interface::setLocalTags is overloaded internally
  // to work with both -DLAMMPS_SMALLBIG (tagint = int) and -DLAMMPS_BIGBIG
  // (tagint = int64_t)
  interface.setLocalTags(atom->tag);


  // Also set absolute atom positions.
  interface.setLocalAtomPositions(atom->x);

  // Set Box vectors if system is periodic in all 3 dims.
  if(domain->nonperiodic == 0)
  {
      interface.setBoxVectors(domain->boxlo,
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
  interface.process();

  // Do all stuff related to extrapolation warnings.
  if(showew == true || showewsum > 0 || maxew >= 0) {
    handleExtrapolationWarnings();
  }

  // Calculate forces of local and ghost atoms.
  interface.getForces(atom->f);

  // Transfer charges LAMMPS. Does nothing if nnpType != 4G.
  interface.getCharges(atom->q);

  // Add energy contribution to total energy.
  if (eflag_global)
     ev_tally(0,0,atom->nlocal,1,interface.getEnergy(),0.0,0.0,0.0,0.0,0.0);

  // Add atomic energy if requested (CAUTION: no physical meaning!).
  if (eflag_atom)
    for (int i = 0; i < atom->nlocal; ++i)
      eatom[i] = interface.getAtomicEnergy(i);

  // If virial needed calculate via F dot r.
  if (vflag_fdotr) virial_fdotr_compute();

    //interface.writeToFile("md_run.data", true);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairNNPDevelop::init_style()
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

  PairNNP::init_style();


  maxCutoffRadiusNeighborList = maxCutoffRadius;

  interface.setGlobalStructureStatus(true);
}


double PairNNPDevelop::init_one(int i, int j)
{
    //cutsq[i][j] = cutsq[j][i] = pow(maxCutoffRadiusNeighborList,2);
    return maxCutoffRadiusNeighborList;
}


void PairNNPDevelop::updateNeighborlistCutoff()
{
    double maxCutoffRadiusOverall = interface.getMaxCutoffRadiusOverall();
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


