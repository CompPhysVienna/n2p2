// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <string.h>
#include <stdlib.h>  //exit(0);
#include <vector>
#include "pair_nnp.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "kspace.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "domain.h" //periodicity
#include "fix_nnp.h"
#include "kspace_nnp.h"
#include <chrono> //time
#include <thread>

using namespace LAMMPS_NS;
using namespace std::chrono;
using namespace nnp;
using namespace std;

#define SQR(x) ((x)*(x))

/* ---------------------------------------------------------------------- */

PairNNP::PairNNP(LAMMPS *lmp) : Pair(lmp),
    kspacennp                      (nullptr),
    periodic                       (false  ),
    showew                         (false  ),
    resetew                        (false  ),
    showewsum                      (0      ),
    maxew                          (0      ),
    numExtrapolationWarningsTotal  (0      ),
    numExtrapolationWarningsSummary(0      ),
    cflength                       (0.0    ),
    cfenergy                       (0.0    ),
    maxCutoffRadius                (0.0    ),
    directory                      (nullptr),
    emap                           (nullptr),
    list                           (nullptr),
    chi                            (nullptr),
    hardness                       (nullptr),
    sigmaSqrtPi                    (nullptr),
    gammaSqrt2                     (nullptr),
    dEdQ                           (nullptr),
    forceLambda                    (nullptr),
    grad_tol                       (0.0    ),
    min_tol                        (0.0    ),
    step                           (0.0    ),
    maxit                          (0      ),
    minim_init_style               (0      ),
    T                              (nullptr),
    s                              (nullptr),
    E_elec                         (0.0    ),
    kcoeff_sum                     (0.0    ),
    type_all                       (nullptr),
    type_loc                       (nullptr),
    dEdLambda_loc                  (nullptr),
    dEdLambda_all                  (nullptr),
    qall_loc                       (nullptr),
    qall                           (nullptr),
    xx                             (nullptr),
    xy                             (nullptr),
    xz                             (nullptr),
    xx_loc                         (nullptr),
    xy_loc                         (nullptr),
    xz_loc                         (nullptr),
    forceLambda_loc                (nullptr),
    forceLambda_all                (nullptr),
    erfc_val                       (nullptr),
    kcos                           (nullptr),
    ksinx                          (nullptr),
    ksiny                          (nullptr),
    ksinz                          (nullptr),
    screening_info                 (nullptr)
{

    MPI_Comm_rank(world,&me);
    MPI_Comm_size(world,&nprocs);
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairNNP::~PairNNP()
{
}

void PairNNP::compute(int eflag, int vflag)
{
  if(eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;

  if (interface.getNnpType() == InterfaceLammps::NNPType::HDNNP_2G)
  {
      // Set number of local atoms and add index and element.
      interface.setLocalAtoms(atom->nlocal,atom->tag,atom->type);

      // Transfer local neighbor list to NNP interface.
      transferNeighborList();

      // Compute symmetry functions, atomic neural networks and add up energy.
      interface.process();

      // Do all stuff related to extrapolation warnings.
      if(showew == true || showewsum > 0 || maxew >= 0) {
          handleExtrapolationWarnings();
      }

      // get short-range forces of local and ghost atoms.
      interface.getForces(atom->f);

  }
  else if (interface.getNnpType() == InterfaceLammps::NNPType::HDNNP_4G)
  {
      // Transfer charges into n2p2 before running second set of NNs
      transferCharges();

      // Add electrostatic energy contribution to the total energy before conversion TODO:check
      interface.addElectrostaticEnergy(E_elec);

      // Run second set of NNs for the short range contributions
      interface.process();

      // Get short-range forces of local and ghost atoms.
      interface.getForces(atom->f);

      // Initialize global arrays
      for (int i =0; i < atom->natoms; i++){
          qall[i] = 0.0;
          qall_loc[i] = 0.0;
          dEdLambda_all[i] = 0.0;
          dEdLambda_loc[i] = 0.0;
          forceLambda[i] = 0.0;
          dEdQ[i] = 0.0;
          forceLambda_loc[i] = 0.0;
          forceLambda_all[i] = 0.0;
          xx_loc[i] = 0.0;
          xy_loc[i] = 0.0;
          xz_loc[i] = 0.0;
          xx[i] = 0.0;
          xy[i] = 0.0;
          xz[i] = 0.0;
          type_loc[i] = 0;
          type_all[i] = 0;
      }

      // Create global sparse arrays here
      for (int i = 0; i < atom->nlocal; i++){
          qall_loc[atom->tag[i]-1] = atom->q[i]; // global charge vector on each proc
          xx_loc[atom->tag[i]-1] = atom->x[i][0];
          xy_loc[atom->tag[i]-1] = atom->x[i][1];
          xz_loc[atom->tag[i]-1] = atom->x[i][2];
          type_loc[atom->tag[i]-1] = atom->type[i];
      }

      // Communicate atomic charges and positions
      MPI_Allreduce(qall_loc,qall,atom->natoms,MPI_DOUBLE,MPI_SUM,world);
      MPI_Allreduce(type_loc,type_all,atom->natoms,MPI_INT,MPI_SUM,world);
      MPI_Allreduce(xx_loc,xx,atom->natoms,MPI_DOUBLE,MPI_SUM,world);
      MPI_Allreduce(xy_loc,xy,atom->natoms,MPI_DOUBLE,MPI_SUM,world);
      MPI_Allreduce(xz_loc,xz,atom->natoms,MPI_DOUBLE,MPI_SUM,world);

      //TODO: it did not work when they were in the constructor as they are in FixNNP, check
      if (periodic){
          kspacennp = nullptr;
          kspacennp = (KSpaceNNP *) force->kspace_match("^nnp",0);
      }

      // Calculates and stores k-space terms for speedup
      calculate_kspace_terms();

      // Calculate dEelecdQ and add pEelecpr contribution to the total force vector
      calculateElecDerivatives(dEdQ,atom->f); // TODO: calculate fElec separately ?

      // Read dEdG array from n2p2
      interface.getdEdQ(dEdQ);

      // Calculate lambda vector that is necessary for optimized force calculation
      calculateForceLambda(); // TODO: lambdaElec & f_elec ?

      // Communicate forceLambda
      for (int i = 0; i < atom->nlocal; i++){
          forceLambda_loc[atom->tag[i]-1] = forceLambda[i];
      }
      for (int i = 0; i < atom->natoms; i++){
          forceLambda_all[i] = 0.0;
      }
      MPI_Allreduce(forceLambda_loc,forceLambda_all,atom->natoms,MPI_DOUBLE,MPI_SUM,world);

      // Add electrostatic contributions and calculate final force vector
      calculateElecForce(atom->f);

      // TODO check
      memory->destroy(erfc_val);
      
      // Do all stuff related to extrapolation warnings.
      if(showew == true || showewsum > 0 || maxew >= 0) {
          handleExtrapolationWarnings();
      }
  }
  // Add energy contribution to total energy.
  if (eflag_global)
     ev_tally(0,0,atom->nlocal,1,interface.getEnergy(),0.0,0.0,0.0,0.0,0.0);

  // Add atomic energy if requested (CAUTION: no physical meaning!).
  if (eflag_atom)
    for (int i = 0; i < atom->nlocal; ++i)
      eatom[i] = interface.getAtomicEnergy(i);

  // If virial needed calculate via F dot r.
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairNNP::settings(int narg, char **arg)
{
  int iarg = 0;

  if (narg == 0) error->all(FLERR,"Illegal pair_style command");

  // default settings
  int len = strlen("nnp/") + 1;
  directory = new char[len];
  strcpy(directory,"nnp/");
  showew = true;
  showewsum = 0;
  maxew = 0;
  resetew = false;
  cflength = 1.0;
  cfenergy = 1.0;
  len = strlen("") + 1;
  emap = new char[len];
  strcpy(emap,"");
  numExtrapolationWarningsTotal = 0;
  numExtrapolationWarningsSummary = 0;

  while(iarg < narg) {
    // set NNP directory
    if (strcmp(arg[iarg],"dir") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal pair_style command");
      delete[] directory;
      len = strlen(arg[iarg+1]) + 2;
      directory = new char[len];
      sprintf(directory, "%s/", arg[iarg+1]);
      iarg += 2;
    // element mapping
    } else if (strcmp(arg[iarg],"emap") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal pair_style command");
      delete[] emap;
      len = strlen(arg[iarg+1]) + 1;
      emap = new char[len];
      sprintf(emap, "%s", arg[iarg+1]);
      iarg += 2;
    // show extrapolation warnings
    } else if (strcmp(arg[iarg],"showew") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal pair_style command");
      if (strcmp(arg[iarg+1],"yes") == 0)
        showew = true;
      else if (strcmp(arg[iarg+1],"no") == 0)
        showew = false;
      else
        error->all(FLERR,"Illegal pair_style command");
      iarg += 2;
    // show extrapolation warning summary
    } else if (strcmp(arg[iarg],"showewsum") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal pair_style command");
      showewsum = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    // maximum allowed extrapolation warnings
    } else if (strcmp(arg[iarg],"maxew") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal pair_style command");
      maxew = utils::inumeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    // reset extrapolation warning counter
    } else if (strcmp(arg[iarg],"resetew") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal pair_style command");
      if (strcmp(arg[iarg+1],"yes") == 0)
        resetew = true;
      else if (strcmp(arg[iarg+1],"no") == 0)
        resetew = false;
      else
        error->all(FLERR,"Illegal pair_style command");
      iarg += 2;
    // length unit conversion factor
    } else if (strcmp(arg[iarg],"cflength") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal pair_style command");
      cflength = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    // energy unit conversion factor
    } else if (strcmp(arg[iarg],"cfenergy") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal pair_style command");
      cfenergy = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;
    } else error->all(FLERR,"Illegal pair_style command");
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNNP::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  if (narg != 3) error->all(FLERR,"Incorrect args for pair coefficients");

  int ilo,ihi,jlo,jhi;

  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  maxCutoffRadius = utils::numeric(FLERR,arg[2],false,lmp); // this the cutoff specified via pair_coeff in the input

  // TODO: Check how this flag is set.
  int count = 0;
  for(int i=ilo; i<=ihi; i++) {
    for(int j=MAX(jlo,i); j<=jhi; j++) {
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairNNP::init_style()
{
  int irequest = neighbor->request((void *) this);
  neighbor->requests[irequest]->pair = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  // Return immediately if NNP setup is already completed.
  if (interface.isInitialized()) return;

  // Activate screen and logfile output only for rank 0.
  if (comm->me == 0) {
    if (lmp->screen != NULL)
      interface.log.registerCFilePointer(&(lmp->screen));
    if (lmp->logfile != NULL)
      interface.log.registerCFilePointer(&(lmp->logfile));
  }

  ///TODO: add nnpType
  // Initialize interface on all processors.
  interface.initialize(directory,
                       emap,
                       showew,
                       resetew,
                       showewsum,
                       maxew,
                       cflength,
                       cfenergy,
                       maxCutoffRadius,
                       atom->ntypes,
                       comm->me);
  // LAMMPS cutoff radius (given via pair_coeff) should not be smaller than
  // maximum symmetry function cutoff radius.
  if (maxCutoffRadius < interface.getMaxCutoffRadius())
    error->all(FLERR,"Inconsistent cutoff radius");

    if (interface.getNnpType() == InterfaceLammps::NNPType::HDNNP_4G)
    {
        isPeriodic(); // check for periodicity here
    }
}

/* ----------------------------------------------------------------------
   init neighbor list(TODO: check this)
------------------------------------------------------------------------- */

void PairNNP::init_list(int /*id*/, NeighList *ptr)
{
    list = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairNNP::init_one(int i, int j)
{
  // TODO: Check how this actually works for different cutoffs.
  return maxCutoffRadius;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairNNP::write_restart(FILE *fp)
{
    return;
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairNNP::read_restart(FILE *fp)
{
    return;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairNNP::write_restart_settings(FILE *fp)
{
    return;
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairNNP::read_restart_settings(FILE *fp)
{
    return;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairNNP::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  int natoms = atom->natoms;
  int nlocal = atom->nlocal;



  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  // TODO: add an if and initialize only for 4G
  // Allocate and initialize 4G-related arrays
  dEdQ = nullptr;
  forceLambda = nullptr;
  memory->create(dEdQ,natoms+1,"pair:dEdQ");
  memory->create(forceLambda,natoms+1,"pair:forceLambda");
  memory->create(dEdLambda_loc,natoms,"pair_nnp:dEdLambda_loc");
  memory->create(dEdLambda_all,natoms,"pair_nnp:dEdLambda_all");
  memory->create(qall_loc,natoms,"pair_nnp:qall_loc");
  memory->create(qall,natoms,"pair_nnp:qall");
  memory->create(type_loc,natoms,"pair_nnp:type_loc");
  memory->create(type_all,natoms,"pair_nnp:type_all");
  memory->create(xx,natoms,"pair_nnp:xx");
  memory->create(xy,natoms,"pair_nnp:xy");
  memory->create(xz,natoms,"pair_nnp:xz");
  memory->create(xx_loc,natoms,"pair_nnp:xx_loc");
  memory->create(xy_loc,natoms,"pair_nnp:xy_loc");
  memory->create(xz_loc,natoms,"pair_nnp:xz_loc");
  memory->create(forceLambda_loc,natoms,"pair_nnp:forceLambda_loc");
  memory->create(forceLambda_all,natoms,"pair_nnp:forceLambda_all");
  
  /*memory->create(kcos,nlocal,natoms,"pair_nnp:kcos");
  memory->create(ksinx,nlocal,natoms,"pair_nnp:ksinx");
  memory->create(ksiny,nlocal,natoms,"pair_nnp:ksiny");
  memory->create(ksinz,nlocal,natoms,"pair_nnp:ksinz");*/

  for (int i = 0; i < natoms+1; i++)
  {
      forceLambda[i] = 0.0;
      dEdQ[i] = 0.0;
  }

    /*for (int i = 0; i < nlocal; i++)
    {
        for (int j = 0; j < natoms; j++)
        {
            kcos[i][j] = 0.0;
            ksinx[i][j] = 0.0;
            ksiny[i][j] = 0.0;
            ksinz[i][j] = 0.0;
        }
    }*/

}

// Transfers neighbor lists to n2p2
void PairNNP::transferNeighborList()
{
  // Transfer neighbor list to NNP.
  double rc2 = maxCutoffRadius * maxCutoffRadius;
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
        interface.addNeighbor(i,j,atom->tag[j],atom->type[j],dx,dy,dz,d2);
      }
    }
  }
}

void PairNNP::handleExtrapolationWarnings()
{
  // Get number of extrapolation warnings for local atoms.
  // TODO: Is the conversion from std::size_t to long ok?
  long numCurrentEW = (long)interface.getNumExtrapolationWarnings();

  // Update (or set, resetew == true) total warnings counter.
  if (resetew) numExtrapolationWarningsTotal = numCurrentEW;
  else numExtrapolationWarningsTotal += numCurrentEW;

  // Update warnings summary counter.
  if(showewsum > 0) {
    numExtrapolationWarningsSummary += numCurrentEW;
  }

  // If requested write extrapolation warnings.
  // Requires communication of all symmetry functions statistics entries to
  // rank 0.
  if(showew > 0) {
    // First collect an overview of extrapolation warnings per process.
    long* numEWPerProc = NULL;
    if(comm->me == 0) numEWPerProc = new long[comm->nprocs];
    MPI_Gather(&numCurrentEW, 1, MPI_LONG, numEWPerProc, 1, MPI_LONG, 0, world);

    if(comm->me == 0) {
      for(int i=1;i<comm->nprocs;i++) {
        if(numEWPerProc[i] > 0) {
          long bs = 0;
          MPI_Status ms;
          // Get buffer size.
          MPI_Recv(&bs, 1, MPI_LONG, i, 0, world, &ms);
          char* buf = new char[bs];
          // Receive buffer.
          MPI_Recv(buf, bs, MPI_BYTE, i, 0, world, &ms);
          interface.extractEWBuffer(buf, bs);
          delete[] buf;
        }
      }
      interface.writeExtrapolationWarnings();
    }
    else if(numCurrentEW > 0) {
      // Get desired buffer length for all extrapolation warning entries.
      long bs = interface.getEWBufferSize();
      // Allocate and fill buffer.
      char* buf = new char[bs];
      interface.fillEWBuffer(buf, bs);
      // Send buffer size and buffer.
      MPI_Send(&bs, 1, MPI_LONG, 0, 0, world);
      MPI_Send(buf, bs, MPI_BYTE, 0, 0, world);
      delete[] buf;
    }

    if(comm->me == 0) delete[] numEWPerProc;
  }

  // If requested gather number of warnings to display summary.
  if(showewsum > 0 && update->ntimestep % showewsum == 0) {
    long globalEW = 0;
    // Communicate the sum over all processors to proc 0.
    MPI_Reduce(&numExtrapolationWarningsSummary,
               &globalEW, 1, MPI_LONG, MPI_SUM, 0, world);
    // Write to screen or logfile.
    if(comm->me == 0) {
      if(screen) {
        fprintf(screen,
                "### NNP EW SUMMARY ### TS: %10ld EW %10ld EWPERSTEP %10.3E\n",
                update->ntimestep,
                globalEW,
                double(globalEW) / showewsum);
      }
      if(logfile) {
        fprintf(logfile,
                "### NNP EW SUMMARY ### TS: %10ld EW %10ld EWPERSTEP %10.3E\n",
                update->ntimestep,
                globalEW,
                double(globalEW) / showewsum);
      }
    }
    // Reset summary counter.
    numExtrapolationWarningsSummary = 0;
  }

  // Stop if maximum number of extrapolation warnings is exceeded.
  if (numExtrapolationWarningsTotal > maxew) {
    error->one(FLERR,"Too many extrapolation warnings");
  }

  // Reset internal extrapolation warnings counters.
  interface.clearExtrapolationWarnings();
}

// Write atomic charges into n2p2
void PairNNP::transferCharges()
{
    for (int i = 0; i < atom->nlocal; ++i) {
        interface.addCharge(i,atom->q[i]);
    }
}

// forceLambda function
double PairNNP::forceLambda_f(const gsl_vector *v)
{
    int i,j,jmap;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    int *tag = atom->tag;
    int nall = atom->natoms;

    double **x = atom->x;
    double E_real, E_recip, E_self;
    double E_lambda,E_lambda_loc;
    double iiterm,ijterm;

    double eta;

    if (periodic) eta = 1 / kspacennp->g_ewald; // LAMMPS truncation

    E_lambda = 0.0;
    E_lambda_loc = 0.0;
    if (periodic)
    {
        double sqrt2eta = (sqrt(2.0) * eta);
        E_recip = 0.0;
        E_real = 0.0;
        E_self = 0.0;
        for (i = 0; i < nlocal; i++) // over local atoms
        {
            double const lambda_i = gsl_vector_get(v, tag[i]-1);
            double lambda_i2 = lambda_i * lambda_i;

            // Self interactionsterm
            E_self += lambda_i2 * (1 / (2.0 * sigmaSqrtPi[type[i]-1]) - 1 / (sqrt(2.0 * M_PI) * eta));
            E_lambda_loc += dEdQ[i] * lambda_i + 0.5 * hardness[type[i]-1] * lambda_i2;

            // Real space
            // TODO: we loop over the full neighbor list, could this be optimized ?
            for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                j = list->firstneigh[i][jj];
                j &= NEIGHMASK;
                double const lambda_j = gsl_vector_get(v, tag[j]-1);
                //jmap = atom->map(tag[j]);
                //double const dx = x[i][0] - x[j][0];
                //double const dy = x[i][1] - x[j][1];
                //double const dz = x[i][2] - x[j][2];
                //double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz)) * cflength;
                //double erfcRij = (erfc(rij / sqrt2eta) - erfc(rij / gammaSqrt2[type[i]-1][type[jmap]-1])) / rij;
                //double real = 0.5 * lambda_i * lambda_j * erfcRij;
                double real = 0.5 * lambda_i * lambda_j * erfc_val[i][jj];
                E_real += real;
            }
        }
        // Reciprocal space
        for (int k = 0; k < kspacennp->kcount; k++) // over k-space
        {
            double sf_real_loc = 0.0;
            double sf_im_loc = 0.0;
            kspacennp->sf_real[k] = 0.0;
            kspacennp->sf_im[k] = 0.0;
            for (i = 0; i < nlocal; i++) //TODO: check
            {
                double const lambda_i = gsl_vector_get(v,tag[i]-1);
                sf_real_loc += lambda_i * kspacennp->sfexp_rl[k][i];
                sf_im_loc += lambda_i * kspacennp->sfexp_im[k][i];
            }
            MPI_Allreduce(&(sf_real_loc),&(kspacennp->sf_real[k]),1,MPI_DOUBLE,MPI_SUM,world);
            MPI_Allreduce(&(sf_im_loc),&(kspacennp->sf_im[k]),1,MPI_DOUBLE,MPI_SUM,world);
            E_recip += kspacennp->kcoeff[k] * (pow(kspacennp->sf_real[k],2) + pow(kspacennp->sf_im[k],2));
        }
        E_lambda_loc += E_real + E_self;
    }else
    {
        // first loop over local atoms
        for (i = 0; i < nlocal; i++) {
            double const lambda_i = gsl_vector_get(v,i);
            // add i terms here
            iiterm = lambda_i * lambda_i / (2.0 * sigmaSqrtPi[type[i]-1]);
            E_lambda += iiterm + dEdQ[i]*lambda_i + 0.5*hardness[type[i]-1]*lambda_i*lambda_i;
            // second loop over 'all' atoms
            for (j = i + 1; j < nall; j++) {
                double const lambda_j = gsl_vector_get(v, j);
                double const dx = x[j][0] - x[i][0];
                double const dy = x[j][1] - x[i][1];
                double const dz = x[j][2] - x[i][2];
                double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz)) * cflength;
                ijterm = lambda_i * lambda_j * (erf(rij / gammaSqrt2[type[i]-1][type[j]-1]) / rij);
                E_lambda += ijterm;
            }
        }
    }

    MPI_Allreduce(&E_lambda_loc,&E_lambda,1,MPI_DOUBLE,MPI_SUM,world); // MPI_SUM of local QEQ contributions
    E_lambda += E_recip; // adding already all-reduced reciprocal part

    return E_lambda;
}

// forceLambda function - wrapper
double PairNNP::forceLambda_f_wrap(const gsl_vector *v, void *params)
{
    return static_cast<PairNNP*>(params)->forceLambda_f(v);
}

// forceLambda gradient
void PairNNP::forceLambda_df(const gsl_vector *v, gsl_vector *dEdLambda)
{
    int i,j,jmap;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;
    int *tag = atom->tag;
    int *type = atom->type;

    double **x = atom->x;
    double val;
    double grad;
    double grad_sum,grad_sum_loc,grad_i;
    double local_sum;

    double eta;

    if (periodic) eta = 1 / kspacennp->g_ewald; // LAMMPS truncation

    grad_sum = 0.0;
    grad_sum_loc = 0.0;
    if (periodic)
    {
        double sqrt2eta = (sqrt(2.0) * eta);
        for (i = 0; i < nlocal; i++) // over local atoms
        {
            double const lambda_i = gsl_vector_get(v,tag[i]-1);

            // Reciprocal space
            double ksum = 0.0;
            for (int k = 0; k < kspacennp->kcount; k++) // over k-space
            {
                ksum += 2.0 * kspacennp->kcoeff[k] *
                        (kspacennp->sf_real[k] * kspacennp->sfexp_rl[k][i] +
                         kspacennp->sf_im[k] * kspacennp->sfexp_im[k][i]);
            }

            // Real space
            double jsum = 0.0;
            for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                j = list->firstneigh[i][jj];
                j &= NEIGHMASK;
                double const lambda_j = gsl_vector_get(v, tag[j]-1);
                //jmap = atom->map(tag[j]);
                //double const dx = x[i][0] - x[j][0];
                //double const dy = x[i][1] - x[j][1];
                //double const dz = x[i][2] - x[j][2];
                //double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz)) * cflength;
                //double erfcRij = (erfc(rij / sqrt2eta) - erfc(rij / gammaSqrt2[type[i]-1][type[jmap]-1]));
                //jsum += lambda_j * erfcRij / rij;
                jsum += lambda_j * erfc_val[i][jj];
            }
            grad = jsum + ksum + dEdQ[i] + hardness[type[i]-1]*lambda_i +
                   lambda_i * (1/(sigmaSqrtPi[type[i]-1])- 2/(eta * sqrt(2.0 * M_PI)));
            grad_sum_loc += grad;
            dEdLambda_loc[tag[i]-1] = grad; // fill gradient array based on tags instead of local IDs
        }
    }else
    {
        // first loop over local atoms
        for (i = 0; i < nlocal; i++) { // TODO: indices
            double const lambda_i = gsl_vector_get(v,i);
            local_sum = 0.0;
            // second loop over 'all' atoms
            for (j = 0; j < nall; j++) {
                if (j != i) {
                    double const lambda_j = gsl_vector_get(v, j);
                    double const dx = x[j][0] - x[i][0];
                    double const dy = x[j][1] - x[i][1];
                    double const dz = x[j][2] - x[i][2];
                    double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz)) * cflength;
                    local_sum += lambda_j * erf(rij / gammaSqrt2[type[i]-1][type[j]-1]) / rij;
                }
            }
            val = dEdQ[i] + hardness[type[i]-1]*lambda_i +
                  lambda_i/(sigmaSqrtPi[type[i]-1]) + local_sum;
            grad_sum = grad_sum + val;
            gsl_vector_set(dEdLambda,i,val);
        }
    }

    MPI_Allreduce(dEdLambda_loc,dEdLambda_all,atom->natoms,MPI_DOUBLE,MPI_SUM,world);
    MPI_Allreduce(&grad_sum_loc,&grad_sum,1,MPI_DOUBLE,MPI_SUM,world);

    // Gradient projection //TODO: communication ?
    for (i = 0; i < nall; i++){
        grad_i = dEdLambda_all[i];
        gsl_vector_set(dEdLambda,i,grad_i - (grad_sum)/nall);
    }

}

// forceLambda gradient - wrapper
void PairNNP::forceLambda_df_wrap(const gsl_vector *v, void *params, gsl_vector *df)
{
    static_cast<PairNNP*>(params)->forceLambda_df(v, df);
}

// forceLambda f*df
void PairNNP::forceLambda_fdf(const gsl_vector *v, double *f, gsl_vector *df)
{
    *f = forceLambda_f(v);
    forceLambda_df(v, df);
}

// forceLambda f*df - wrapper
void PairNNP::forceLambda_fdf_wrap(const gsl_vector *v, void *params, double *f, gsl_vector *df)
{
    static_cast<PairNNP*>(params)->forceLambda_fdf(v, f, df);
}

// Calculate forcelambda vector $\lambda_i$ that is required for optimized force calculation
void PairNNP::calculateForceLambda()
{
    size_t iter = 0;
    int i,j;
    int nsize,status;
    int nlocal;

    double psum_it;
    double gradsum;
    double lambda_i;

    nsize = atom->natoms; // total number of atoms
    nlocal = atom->nlocal; // total number of atoms

    gsl_vector *x; // charge vector in our case

    forceLambda_minimizer.n = nsize;
    forceLambda_minimizer.f = &forceLambda_f_wrap;
    forceLambda_minimizer.df = &forceLambda_df_wrap;
    forceLambda_minimizer.fdf = &forceLambda_fdf_wrap;
    forceLambda_minimizer.params = this;

    for (int i =0; i < atom->natoms; i++) {
        dEdLambda_all[i] = 0.0;
        dEdLambda_loc[i] = 0.0;
        forceLambda_loc[i] = 0.0;
        forceLambda_all[i] = 0.0;
    }

    // Communicate forceLambda
    for (int i = 0; i < atom->nlocal; i++){
        forceLambda_loc[atom->tag[i]-1] = forceLambda[i];
    }
    MPI_Allreduce(forceLambda_loc,forceLambda_all,atom->natoms,MPI_DOUBLE,MPI_SUM,world);


    // TODO : check
    x = gsl_vector_alloc(nsize);
    for (i = 0; i < nsize; i++) {
        gsl_vector_set(x,i,forceLambda_all[i]);
    }

    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T, nsize);

    gsl_multimin_fdfminimizer_set(s, &forceLambda_minimizer, x, step, min_tol); // TODO would tol = 0 be expensive ??
    do
    {
        iter++;
        psum_it = 0.0;
        status = gsl_multimin_fdfminimizer_iterate(s);

        // Projection
        for(i = 0; i < nsize; i++) {
            psum_it = psum_it + gsl_vector_get(s->x, i);
        }
        for(i = 0; i < nsize; i++) {
            lambda_i = gsl_vector_get(s->x,i);
            gsl_vector_set(s->x,i, lambda_i - psum_it/nsize); // projection
        }

        status = gsl_multimin_test_gradient(s->gradient, grad_tol);

        if (status == GSL_SUCCESS)
            printf ("Minimum forceLambda is found at iteration: %zu\n",iter);

    }
    while (status == GSL_CONTINUE && iter < maxit);

    // Read charges before deallocating x
    for (i = 0; i < nlocal; i++) {
        forceLambda[i] = gsl_vector_get(s->x,atom->tag[i]-1);
    }

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);
}

void PairNNP::calculate_kspace_terms()
{
    int i,j,k;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;

    double **x = atom->x;
    double dx,dy,dz;
    double kdum_cos,kdum_sinx,kdum_siny,kdum_sinz;

    // This term is used in dEdQ calculation and can be calculated beforehand
    for (k = 0; k < kspacennp->kcount; k++)
    {
        kcoeff_sum += 2 * kspacennp->kcoeff[k];
    }

    for (i = 0; i < nlocal; i++)
    {
        for (j = 0; j < nall; j++)
        {
            kdum_cos = 0.0;
            kdum_sinx = 0.0;
            kdum_siny = 0.0;
            kdum_sinz = 0.0;

            /*dx = x[i][0] - xx[j];
            dy = x[i][1] - xy[j];
            dz = x[i][2] - xz[j];

            for (int k = 0; k < kspacennp->kcount; k++) {
                double kx = kspacennp->kxvecs[k] * kspacennp->unitk[0];
                double ky = kspacennp->kyvecs[k] * kspacennp->unitk[1];
                double kz = kspacennp->kzvecs[k] * kspacennp->unitk[2];
                double kdr = (dx * kx + dy * ky + dz * kz) * cflength;

                    // Cos term
                    if (dx != 0.0) kdum_cos += 2.0 * kspacennp->kcoeff[k] * cos(kdr);

                    // Sin terms
                    double sinkdr = sin(kdr);
                    kdum_sinx -= 2.0 * kspacennp->kcoeff[k] * sinkdr * kx;
                    kdum_siny -= 2.0 * kspacennp->kcoeff[k] * sinkdr * ky;
                    kdum_sinz -= 2.0 * kspacennp->kcoeff[k] * sinkdr * kz;
            }
            kcos[i][j] = kdum_cos;
            ksinx[i][j] = kdum_sinx;
            ksiny[i][j] = kdum_siny;
            ksinz[i][j] = kdum_sinz;*/
        }
    }
}

// Calculate $dEelec/dQ_i$ and add one part of the $\partial Eelec/\partial Q_i$ contribution to the total force vector
void PairNNP::calculateElecDerivatives(double *dEelecdQ, double **f)
{
    int i,j,jmap;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;
    int *tag = atom->tag;
    int *type = atom->type;

    double **x = atom->x;
    double *q = atom->q;
    double qi,qj;
    double dx,dy,dz;
    double rij,rij2,erfrij;
    double gams2;
    double sij,tij;
    double fsrij,dfsrij; // corrections due to screening

    double eta;

    if (periodic) eta = 1 / kspacennp->g_ewald; // LAMMPS truncation (RuNNer truncation has been removed)

    for (i = 0; i < nlocal; i++)
    {
        qi = q[i];
        if (periodic)
        {
            double sqrt2eta = (sqrt(2.0) * eta);
            double ksum = 0.0;
            for (j = 0; j < nall; j++)
            {
                double Aij = 0.0;
                qj = qall[j];
                dx = x[i][0] - xx[j];
                dy = x[i][1] - xy[j];
                dz = x[i][2] - xz[j];
                // Reciprocal part
                for (int k = 0; k < kspacennp->kcount; k++) {
                    double kdr = (dx * kspacennp->kxvecs[k] * kspacennp->unitk[0] +
                                  dy * kspacennp->kyvecs[k] * kspacennp->unitk[1] +
                                  dz * kspacennp->kzvecs[k] * kspacennp->unitk[2]) * cflength;
                    if (dx != 0.0) Aij += 2.0 * kspacennp->kcoeff[k] * cos(kdr);
                }
                dEelecdQ[i] += qj * Aij;
                //dEelecdQ[i] += qj * kcos[i][j];
            }
            // Add remaining k-space contributions here
            dEelecdQ[i] += qi * (kcoeff_sum - 2 / (sqrt2eta * sqrt(M_PI)));

            // Real space
            for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                j = list->firstneigh[i][jj];
                j &= NEIGHMASK;
                jmap = atom->map(tag[j]);
                qj = qall[tag[j]-1];
                dx = x[i][0] - x[j][0];
                dy = x[i][1] - x[j][1];
                dz = x[i][2] - x[j][2];
                rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                rij = sqrt(rij2) * cflength;

                gams2 = gammaSqrt2[type[i]-1][type[jmap]-1];

                //double Aij = (erfc(rij / sqrt2eta) - erfc(rij / gams2)) / rij;
                double Aij = erfc_val[i][jj];

                dEelecdQ[i] += qj * Aij; // Add Aij contribution regardless of screening

                if (rij < screening_info[2])
                {
                    erfrij = erf(rij / gams2);
                    fsrij = screening_f(rij);
                    dfsrij = screening_df(rij);

                    sij = erfrij * (fsrij - 1) / rij;

                    tij = (2 / (sqrt(M_PI)*gams2) * exp(- pow(rij / gams2,2)) *
                           (fsrij - 1) + erfrij*dfsrij - erfrij*(fsrij-1)/rij) / rij2;

                    dEelecdQ[i] += qj * (sij); // Add sij contributions if inside screening cutoff

                    f[i][0] -= 0.5 * (qi * qj * dx * tij) / cfenergy;
                    f[i][1] -= 0.5 * (qi * qj * dy * tij) / cfenergy;
                    f[i][2] -= 0.5 * (qi * qj * dz * tij) / cfenergy;

                    f[j][0] += 0.5 * (qi * qj * dx * tij) / cfenergy;
                    f[j][1] += 0.5 * (qi * qj * dy * tij) / cfenergy;
                    f[j][2] += 0.5 * (qi * qj * dz * tij) / cfenergy;
                }
            }
        }else
        {
            for (j = 0; j < nall; j++)
            {
                // Retrieve position, type and charge from global arrays
                //TODO check
                dx = x[i][0] - xx[j];
                dy = x[i][1] - xy[j];
                dz = x[i][2] - xz[j];
                qj = qall[j];
                rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                rij = sqrt(rij2) * cflength;

                if (rij != 0.0)
                {
                    gams2 = gammaSqrt2[type[i]-1][type_all[j]-1];

                    erfrij = erf(rij / gams2);
                    fsrij = screening_f(rij);
                    dfsrij = screening_df(rij);

                    sij = erfrij * (fsrij - 1) / rij;
                    tij = (2 / (sqrt(M_PI)*gams2) * exp(- pow(rij / gams2,2)) *
                           (fsrij - 1) + erfrij*dfsrij - erfrij*(fsrij-1)/rij) / rij2;

                    dEelecdQ[i] += qj * ((erfrij/rij) + sij);
                    f[i][0] -= qi * qj * dx * tij;
                    f[i][1] -= qi * qj * dy * tij;
                    f[i][2] -= qi * qj * dz * tij;
                }
            }
        }
    }
    kcoeff_sum = 0.0; // re-initialize to 0.0 here
}

// Calculate electrostatic forces and add to atomic force vectors
// TODO: clean-up & non-periodic
void PairNNP::calculateElecForce(double **f)
{

    int i,j,k;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;
    int *tag = atom->tag;
    int *type = atom->type;

    double rij;
    double qi,qj,qk;
    double *q = atom->q;
    double **x = atom->x;
    double delr,gams2;
    double dx,dy,dz;

    double eta;

    if (periodic) eta = 1 / kspacennp->g_ewald; // LAMMPS truncation

    // lambda_i * dChidr contributions are added into the force vectors in n2p2
    interface.getForcesChi(forceLambda_all, atom->f);

    for (i = 0; i < nlocal; i++) {
        qi = q[i];
        double lambdai = forceLambda[i];
        if (periodic) {
            double sqrt2eta = (sqrt(2.0) * eta);
            double dAdrx = 0.0;
            double dAdry = 0.0;
            double dAdrz = 0.0;

            // Real space
            for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                k = list->firstneigh[i][jj];
                k &= NEIGHMASK;
                int jmap = atom->map(tag[k]);
                qk = qall[tag[k]-1];
                double lambdak = forceLambda_all[tag[k]-1];
                dx = x[i][0] - x[k][0];
                dy = x[i][1] - x[k][1];
                dz = x[i][2] - x[k][2];
                double rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                rij = sqrt(rij2) * cflength;

                if (rij < kspacennp->ewald_real_cutoff) {
                    gams2 = gammaSqrt2[type[i]- 1][type[jmap]-1];
                    //delr = (2 / sqrt(M_PI) * (-exp(-pow(rij / sqrt2eta, 2))
                    //                          / sqrt2eta + exp(-pow(rij / gams2, 2)) / gams2)
                    //        - 1 / rij * (erfc(rij / sqrt2eta) - erfc(rij / gams2))) / rij2;
                    delr = (2 / sqrt(M_PI) * (-exp(-pow(rij / sqrt2eta, 2))
                                              / sqrt2eta + exp(-pow(rij / gams2, 2)) / gams2)
                            - erfc_val[i][jj]) / rij2;
                    dAdrx = dx * delr;
                    dAdry = dy * delr;
                    dAdrz = dz * delr;

                    // Contributions to the local atom i
                    f[i][0] -= 0.5 * (lambdai * qk + lambdak * qi) * dAdrx / cfenergy;
                    f[i][1] -= 0.5 * (lambdai * qk + lambdak * qi) * dAdry / cfenergy;
                    f[i][2] -= 0.5 * (lambdai * qk + lambdak * qi) * dAdrz / cfenergy;

                    // Contributions to the neighbors of the local atom i
                    f[k][0] += 0.5 * (lambdai * qk + lambdak * qi) * dAdrx / cfenergy;
                    f[k][1] += 0.5 * (lambdai * qk + lambdak * qi) * dAdry / cfenergy;
                    f[k][2] += 0.5 * (lambdai * qk + lambdak * qi) * dAdrz / cfenergy;

                    // Contributions to the local atom i (pEelecpr)
                    f[i][0] -= (0.5 * qk * dAdrx * qi) / cfenergy;
                    f[i][1] -= (0.5 * qk * dAdry * qi) / cfenergy;
                    f[i][2] -= (0.5 * qk * dAdrz * qi) / cfenergy;

                    f[k][0] += (0.5 * qk * dAdrx * qi) / cfenergy;
                    f[k][1] += (0.5 * qk * dAdry * qi) / cfenergy;
                    f[k][2] += (0.5 * qk * dAdrz * qi) / cfenergy;
                }
            }
            // Reciprocal space
            for (j = 0; j < nall; j++){
                qj = qall[j];
                double lambdaj = forceLambda_all[j];
                dx = x[i][0] - xx[j];
                dy = x[i][1] - xy[j];
                dz = x[i][2] - xz[j];
                double ksx = 0;
                double ksy = 0;
                double ksz = 0;
                for (int kk = 0; kk < kspacennp->kcount; kk++) {
                    double kx = kspacennp->kxvecs[kk] * kspacennp->unitk[0];
                    double ky = kspacennp->kyvecs[kk] * kspacennp->unitk[1];
                    double kz = kspacennp->kzvecs[kk] * kspacennp->unitk[2];
                    double kdr = (dx * kx + dy * ky + dz * kz) * cflength;
                    ksx -= 2.0 * kspacennp->kcoeff[kk] * sin(kdr) * kx;
                    ksy -= 2.0 * kspacennp->kcoeff[kk] * sin(kdr) * ky;
                    ksz -= 2.0 * kspacennp->kcoeff[kk] * sin(kdr) * kz;
                }
                dAdrx = ksx;
                dAdry = ksy;
                dAdrz = ksz;
                //dAdrx = ksinx[i][j];
                //dAdry = ksiny[i][j];
                //dAdrz = ksinz[i][j];

                // Contributions to the local atom i
                f[i][0] -= (lambdai * qj + lambdaj * qi) * dAdrx * (cflength/cfenergy);
                f[i][1] -= (lambdai * qj + lambdaj * qi) * dAdry * (cflength/cfenergy);
                f[i][2] -= (lambdai * qj + lambdaj * qi) * dAdrz * (cflength/cfenergy);

                // pEelecpr
                f[i][0] -= (qj * dAdrx * qi) * (cflength/cfenergy);
                f[i][1] -= (qj * dAdry * qi) * (cflength/cfenergy);
                f[i][2] -= (qj * dAdrz * qi) * (cflength/cfenergy);

                // Contributions to the neighbors of the local atom i
                /*f[k][0] += (lambdai * qj + lambdaj * qi) * dAdrx * (cflength/cfenergy);
                f[k][1] += (lambdai * qj + lambdaj * qi) * dAdry * (cflength/cfenergy);
                f[k][2] += (lambdai * qj + lambdaj * qi) * dAdrz * (cflength/cfenergy);*/

                /*f[k][0] += (qj * dAdrx * qi) * (cflength/cfenergy);
                f[k][1] += (qj * dAdry * qi) * (cflength/cfenergy);
                f[k][2] += (qj * dAdrz * qi) * (cflength/cfenergy);*/
            }
        } else {
            // Over all atoms in the system
            for (j = 0; j < nall; j++) {
                double jt0 = 0;
                double jt1 = 0;
                double jt2 = 0;
                //qj = q[j];
                qj = qall[j];
                if (i == j) {
                    // We have to loop over all atoms once again to calculate dAdrQ terms
                    for (k = 0; k < nall; k++) {
                        //qk = q[k];
                        qk = qall[k];
                        //dx = x[i][0] - x[k][0];
                        dx = x[i][0] - xx[k];
                        //dy = x[i][1] - x[k][1];
                        dy = x[i][1] - xy[k];
                        //dz = x[i][2] - x[k][2];
                        dz = x[i][2] - xz[k];
                        double rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                        rij = sqrt(rij2) * cflength;

                        if (rij != 0.0)
                        {
                            //gams2 = gammaSqrt2[type[i]-1][type[k]-1];
                            gams2 = gammaSqrt2[type[i]-1][type_all[k]-1];
                            delr = (2 / (sqrt(M_PI) * gams2) * exp(-pow(rij / gams2, 2)) - erf(rij / gams2) / rij);

                            jt0 += (dx / rij2) * delr * qk;
                            jt1 += (dy / rij2) * delr * qk;
                            jt2 += (dz / rij2) * delr * qk;
                        }
                    }
                } else {
                    //TODO
                    //dx = x[i][0] - x[j][0];
                    dx = x[i][0] - xx[j];
                    //dy = x[i][1] - x[j][1];
                    dy = x[i][1] - xy[j];
                    //dz = x[i][2] - x[j][2];
                    dz = x[i][2] - xz[j];
                    double rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                    rij = sqrt(rij2) * cflength;

                    //gams2 = gammaSqrt2[type[i]-1][type[j]-1];
                    gams2 = gammaSqrt2[type[i]-1][type_all[j]-1];
                    delr = (2 / (sqrt(M_PI) * gams2) * exp(-pow(rij / gams2, 2)) - erf(rij / gams2) / rij);

                    jt0 = (dx / rij2) * delr * qi;
                    jt1 = (dy / rij2) * delr * qi;
                    jt2 = (dz / rij2) * delr * qi;
                }
                f[i][0] -= (forceLambda_all[j] * (jt0) + 0.5 * qj * jt0) / cfenergy;
                f[i][1] -= (forceLambda_all[j] * (jt1) + 0.5 * qj * jt1) / cfenergy;
                f[i][2] -= (forceLambda_all[j] * (jt2) + 0.5 * qj * jt2) / cfenergy;
            }
        }
    }
}

// Calculate screening function
// TODO : add other function types
double PairNNP::screening_f(double r)
{
    double x;

    if (r >= screening_info[2]) return 1.0;
    else if (r <= screening_info[1]) return 0.0;
    else
    {
        x = (r-screening_info[1])*screening_info[3];
        return 1.0 - 0.5*(cos(M_PI*x)+1);
    }
}

// Calculate derivative of the screening function
// TODO : add other function types
double PairNNP::screening_df(double r) {

    double x;

    if (r >= screening_info[2] || r <= screening_info[1]) return 0.0;
    else {
        x = (r - screening_info[1]) * screening_info[3];
        return -screening_info[3] * (-M_PI_2 * sin(M_PI * x));
    }
}

// Check for periodicity
void PairNNP::isPeriodic()
{
    if (domain->nonperiodic == 0) periodic = true;
    else                          periodic = false;
}






