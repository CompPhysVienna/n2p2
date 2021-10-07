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
#include "pair_nnp.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "domain.h" // for the periodicity check
#include "fix_nnp.h"
#include <chrono> //time


#define SQR(x) ((x)*(x))
#define MIN_NBRS 100
#define MIN_CAP  50
#define DANGER_ZONE    0.90
#define SAFE_ZONE      1.2


using namespace LAMMPS_NS;
using namespace std::chrono;
using namespace nnp;

/* ---------------------------------------------------------------------- */

PairNNP::PairNNP(LAMMPS *lmp) : Pair(lmp),
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
    eElec                          (0.0    ),
    dEdQ                           (nullptr),
    forceLambda                    (nullptr),
    dChidxyz                       (nullptr),
    overallCutoff                  (0.0    ),
    grad_tol                       (0.0    ),
    min_tol                        (0.0    ),
    step                           (0.0    ),
    maxit                          (0      ),
    T                              (nullptr),
    s                              (nullptr),
    gsqmx                          (0.0    ),
    volume                         (0.0    ),
    q2                             (0.0    ),
    g_ewald                        (0.0    ),
    ewaldPrecision                 (0.0    ),
    ewaldEta                       (0.0    ),
    recip_cut                      (0.0    ),
    real_cut                       (0.0    ),
    E_elec                         (0.0    ),
    kxvecs                         (nullptr),
    kyvecs                         (nullptr),
    kzvecs                         (nullptr),
    kxmax_orig                     (0      ),
    kymax_orig                     (0      ),
    kzmax_orig                     (0      ),
    kmax_created                   (0      ),
    kxmax                          (0      ),
    kymax                          (0      ),
    kzmax                          (0      ),
    kmax                           (0      ),
    kmax3d                         (0      ),
    kcount                         (0      ),
    nmax                           (0      ),
    kcoeff                         (nullptr),
    eg                             (nullptr),
    vg                             (nullptr),
    ek                             (nullptr),
    sfexp_rl                       (nullptr),
    sfexp_im                       (nullptr),
    sf_real                        (nullptr),
    sf_im                          (nullptr),
    sfexp_rl_all                   (nullptr),
    sfexp_im_all                   (nullptr),
    cs                             (nullptr),
    sn                             (nullptr),
    screening_info                 (nullptr)
{

}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairNNP::~PairNNP()
{
    if (periodic)
    {
        deallocate_kspace();
    }
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

      //auto start = high_resolution_clock::now();
      transferCharges();

      // Run second set of NNs for the short range contributions
      interface.process();

      // Get short-range forces of local and ghost atoms.
      interface.getForces(atom->f);

      // Calculate dEelecdQ and add pEelecpr contribution to the total force vector
      calculateElecDerivatives(dEdQ,atom->f); // TODO: calculate fElec separately ?

      // Read dEdG array from n2p2
      interface.getdEdQ(dEdQ);

      // Calculate lambda vector that is necessary for optimized force calculation
      calculateForceLambda(); // TODO: calculate lambdaElec separately ?

      // Add electrostatic contributions and calculate final force vector
      calculateElecForce(atom->f);

      //auto stop = high_resolution_clock::now();
      //auto duration = duration_cast<microseconds>(stop - start);
      //std::cout << duration.count() << '\n';

      //std::cout << "kmax : " << kcount << '\n';
      //std::cout << "Ewald Pre : " << ewaldPrecision << '\n';

      // Do all stuff related to extrapolation warnings.
      if(showew == true || showewsum > 0 || maxew >= 0) {
          handleExtrapolationWarnings();
      }

  }

  // Add energy contribution to total energy.
  if (eflag_global)
     ev_tally(0,0,atom->nlocal,1,interface.getEnergy(),E_elec,0.0,0.0,0.0,0.0); //TODO: check if Eelec is added

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

  maxCutoffRadius = utils::numeric(FLERR,arg[2],false,lmp);

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
      isPeriodic();
      // TODO: add cutoff update
      if (periodic)
      {
          //maxCutoffRadius = fmax(maxCutoffRadius, screening_info[2]);
          //maxCutoffRadius = getOverallCutoffRadius(maxCutoffRadius);
      }else
      {
          //maxCutoffRadius = fmax(maxCutoffRadius, screening_info[2]);
      }
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
  int natoms = atom->natoms; // TODO : should this be nlocal ?

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  // TODO: add an if an initialize only for 4G
  // Allocate and initialize 4g-related arrays
  dEdQ = nullptr;
  forceLambda = nullptr;
  memory->create(dEdQ,natoms+1,"pair:dEdQ");
  memory->create(forceLambda,natoms+1,"pair:forceLambda");
  memory->create(dChidxyz,natoms+1,3,"pair:dChidxyz");
  for (int i = 0; i < natoms+1; i++) // TODO: do we need these initializations ?
  {
      forceLambda[i] = 0.0;
      dEdQ[i] = 0.0;
  }
  // Allocate and initialize k-space related arrays if periodic
  //if (periodic)
  {
      //allocate_kspace();
      kmax = 0;
      kmax_created = 0;
      kxvecs = kyvecs = kzvecs = nullptr;
      kcoeff = nullptr;
      //sfacrl = sfacim = sfacrl_all = sfacim_all = nullptr; // these are from LAMMPS, might be needed
      cs = sn = nullptr;
      kcount = 0;
  }

}

// TODO: check, j became jmap
void PairNNP::transferNeighborList()
{
  // Transfer neighbor list to NNP.
  double rc2 = maxCutoffRadius * maxCutoffRadius;
  for (int ii = 0; ii < list->inum; ++ii) {
    int i = list->ilist[ii];
    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      int j = list->firstneigh[i][jj];
      j &= NEIGHMASK;
      int jmap = atom->map(atom->tag[j]);
      double dx = atom->x[i][0] - atom->x[j][0];
      double dy = atom->x[i][1] - atom->x[j][1];
      double dz = atom->x[i][2] - atom->x[j][2];
      double d2 = dx * dx + dy * dy + dz * dz;
      if (d2 <= rc2) {
        interface.addNeighbor(i,jmap,atom->tag[j],atom->type[j],dx,dy,dz,d2);
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
    double E_lambda;
    double iiterm,ijterm;

    E_lambda = 0.0;
    if (periodic)
    {
        double sqrt2eta = (sqrt(2.0) * ewaldEta);
        E_recip = 0.0;
        E_real = 0.0;
        E_self = 0.0;
        for (i = 0; i < nlocal; i++) // over local atoms
        {
            double const lambda_i = gsl_vector_get(v, i);
            double lambda_i2 = lambda_i * lambda_i;
            // Self term
            E_self += lambda_i2 * (1 / (2.0 * sigmaSqrtPi[type[i]-1]) - 1 / (sqrt(2.0 * M_PI) * ewaldEta));
            E_lambda += dEdQ[i] * lambda_i + 0.5 * hardness[type[i]-1] * lambda_i2;
            // Real Term
            // TODO: we loop over the full neighbor list, this can be optimized
            for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                j = list->firstneigh[i][jj];
                j &= NEIGHMASK;
                jmap = atom->map(tag[j]); //TODO: check, required for the function minimization ?
                double const lambda_j = gsl_vector_get(v, jmap);
                double const dx = x[i][0] - x[j][0];
                double const dy = x[i][1] - x[j][1];
                double const dz = x[i][2] - x[j][2];
                double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
                double erfcRij = (erfc(rij / sqrt2eta) - erfc(rij / gammaSqrt2[type[i]-1][type[jmap]-1])) / rij;
                double real = 0.5 * lambda_i * lambda_j * erfcRij;
                E_real += real;
            }
        }
        // Reciprocal Term
        for (int k = 0; k < kcount; k++) // over k-space
        {
            sf_real[k] = 0.0;
            sf_im[k] = 0.0;
            for (i = 0; i < nlocal; i++) //TODO: discuss this additional inner loop
            {
                double const lambda_i = gsl_vector_get(v,i);
                sf_real[k] += lambda_i * sfexp_rl[k][i];
                sf_im[k] += lambda_i * sfexp_im[k][i];
            }
            E_recip += kcoeff[k] * (pow(sf_real[k],2) + pow(sf_im[k],2));
        }
        E_lambda += E_real + E_self + E_recip;
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
                double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
                ijterm = lambda_i * lambda_j * (erf(rij / gammaSqrt2[type[i]-1][type[j]-1]) / rij);
                E_lambda += ijterm;
            }
        }
    }

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
    double grad_sum,grad_i;
    double local_sum;

    grad_sum = 0.0;
    if (periodic)
    {
        double sqrt2eta = (sqrt(2.0) * ewaldEta);
        for (i = 0; i < nlocal; i++) // over local atoms
        {
            double const lambda_i = gsl_vector_get(v,i);
            // Reciprocal contribution
            double ksum = 0.0;
            for (int k = 0; k < kcount; k++) // over k-space
            {
                ksum += 2.0 * kcoeff[k] *
                        (sf_real[k] * sfexp_rl[k][i] + sf_im[k] * sfexp_im[k][i]);
            }
            // Real contribution - over neighbors
            double jsum = 0.0;
            for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                j = list->firstneigh[i][jj];
                j &= NEIGHMASK;
                jmap = atom->map(tag[j]);
                double const lambda_j = gsl_vector_get(v, jmap);
                double const dx = x[i][0] - x[j][0];
                double const dy = x[i][1] - x[j][1];
                double const dz = x[i][2] - x[j][2];
                double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
                double erfcRij = (erfc(rij / sqrt2eta) - erfc(rij / gammaSqrt2[type[i]-1][type[jmap]-1]));
                jsum += lambda_j * erfcRij / rij;
            }
            grad = jsum + ksum + dEdQ[i] + hardness[type[i]-1]*lambda_i +
                   lambda_i * (1/(sigmaSqrtPi[type[i]-1])- 2/(ewaldEta * sqrt(2.0 * M_PI)));
            grad_sum += grad;
            gsl_vector_set(dEdLambda,i,grad);
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
                    double const rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
                    local_sum += lambda_j * erf(rij / gammaSqrt2[type[i]-1][type[j]-1]) / rij;
                }
            }
            val = dEdQ[i] + hardness[type[i]-1]*lambda_i +
                  lambda_i/(sigmaSqrtPi[type[i]-1]) + local_sum;
            grad_sum = grad_sum + val;
            gsl_vector_set(dEdLambda,i,val);
        }
    }

    // Gradient projection //TODO: communication ?
    for (i = 0; i < nall; i++){
        grad_i = gsl_vector_get(dEdLambda,i);
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

    double psum_it;
    double gradsum;
    double lambda_i;

    nsize = atom->natoms;
    gsl_vector *x; // charge vector in our case

    forceLambda_minimizer.n = nsize;
    forceLambda_minimizer.f = &forceLambda_f_wrap;
    forceLambda_minimizer.df = &forceLambda_df_wrap;
    forceLambda_minimizer.fdf = &forceLambda_fdf_wrap;
    forceLambda_minimizer.params = this;

    // TODO : check
    x = gsl_vector_alloc(nsize);
    for (i = 0; i < nsize; i++) {
        gsl_vector_set(x,i,forceLambda[i]);
    }

    //T = gsl_multimin_fdfminimizer_conjugate_fr;
    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T, nsize);

    gsl_multimin_fdfminimizer_set(s, &forceLambda_minimizer, x, step, min_tol); // tol = 0 might be expensive ???
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

    // read charges before deallocating x - be careful with indices !
    for (i = 0; i < nsize; i++) {
        forceLambda[i] = gsl_vector_get(s->x,i);
    }
    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);
}

// Calculate $dEelec/dQ_i$ and add $\partial Eelec/\partial Q_i$ contribution to the total force vector
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

    // TODO: parallelization
    for (i = 0; i < nlocal; i++)
    {
        qi = q[i];
        if (periodic)
        {
            double sqrt2eta = (sqrt(2.0) * ewaldEta);
            double ksum = 0.0;
            //TODO: do we need this loop?
            for (j = 0; j < nall; j++)
            {
                double Aij = 0.0;
                qj = q[j];
                if (i != j)
                {
                    dx = x[i][0] - x[j][0];
                    dy = x[i][1] - x[j][1];
                    dz = x[i][2] - x[j][2];
                    // Reciprocal part
                    for (int k = 0; k < kcount; k++) {
                        double kdr = dx*kxvecs[k]*unitk[0] + dy*kyvecs[k]*unitk[1]
                                + dz*kzvecs[k]*unitk[2];
                        Aij += 2.0 * kcoeff[k] * cos(kdr);
                    }
                    dEelecdQ[i] += qj * Aij;
                }
            }
            // Kspace term // TODO: could this be done in above loop ?
            for (int k = 0; k < kcount; k++)
            {
                ksum += 2 * kcoeff[k];
            }
            dEelecdQ[i] += qi * (ksum - 2 / (sqrt2eta * sqrt(M_PI)));
            // Real term over neighbors
            for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                j = list->firstneigh[i][jj];
                j &= NEIGHMASK;
                jmap = atom->map(tag[j]);
                qj = q[jmap];
                dx = x[i][0] - x[j][0];
                dy = x[i][1] - x[j][1];
                dz = x[i][2] - x[j][2];
                rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                rij = sqrt(rij2);

                //if (rij >= screening_info[2]) break; // TODO: check

                gams2 = gammaSqrt2[type[i]-1][type[jmap]-1];
                erfrij = erf(rij / gams2);
                fsrij = screening_f(rij);
                dfsrij = screening_df(rij);

                sij = erfrij * (fsrij - 1) / rij;
                tij = (2 / (sqrt(M_PI)*gams2) * exp(- pow(rij / gams2,2)) *
                       (fsrij - 1) + erfrij*dfsrij - erfrij*(fsrij-1)/rij) / rij2;
                double Aij = (erfc(rij / sqrt2eta) - erfc(rij / gams2)) / rij;

                dEelecdQ[i] += qj * (sij + Aij);
                /*pEelecpr[i][0] += qi * qj * dx * tij;
                pEelecpr[i][1] += qi * qj * dy * tij;
                pEelecpr[i][2] += qi * qj * dz * tij;*/
                f[i][0] -= qi * qj * dx * tij;
                f[i][1] -= qi * qj * dy * tij;
                f[i][2] -= qi * qj * dz * tij;
            }
        }else
        {
            for (j = 0; j < nall; j++)
            {
                if (i != j)
                {
                    dx = x[i][0] - x[j][0];
                    dy = x[i][1] - x[j][1];
                    dz = x[i][2] - x[j][2];
                    qj = q[j];
                    rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                    rij = sqrt(rij2);
                    gams2 = gammaSqrt2[type[i]-1][type[j]-1];

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
}

// Calculate electrostatic forces and add to atomic force vectors
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

    // TODO:parallelization
    for (i = 0; i < nlocal; i++) {
        qi = q[i];
        int ne = 0;
        reinitialize_dChidxyz();
        interface.getdChidxyz(i, dChidxyz);
        if (periodic) {
            double sqrt2eta = (sqrt(2.0) * ewaldEta);
            for (j = 0; j < nall; j++) {
                double jt0 = 0;
                double jt1 = 0;
                double jt2 = 0;
                qj = q[j];
                if (i == j) {
                    /// Reciprocal Contribution
                    for (k = 0; k < nall; k++) {
                        if (k != i) {
                            qk = q[k];
                            dx = x[i][0] - x[k][0];
                            dy = x[i][1] - x[k][1];
                            dz = x[i][2] - x[k][2];
                            double ksx = 0;
                            double ksy = 0;
                            double ksz = 0;
                            for (int kk = 0; kk < kcount; kk++) {
                                double kx = kxvecs[kk] * unitk[0];
                                double ky = kyvecs[kk] * unitk[1];
                                double kz = kzvecs[kk] * unitk[2];
                                double kdr = dx * kx + dy * ky + dz * kz;
                                ksx -= 2.0 * kcoeff[kk] * sin(kdr) * kx;
                                ksy -= 2.0 * kcoeff[kk] * sin(kdr) * ky;
                                ksz -= 2.0 * kcoeff[kk] * sin(kdr) * kz;
                            }
                            jt0 += ksx * qk;
                            jt1 += ksy * qk;
                            jt2 += ksz * qk;
                        }
                    }
                    /// Real contribution
                    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                        k = list->firstneigh[i][jj]; // use k here as j was used above
                        k &= NEIGHMASK;
                        int jmap = atom->map(tag[k]); // local index of a global neighbor
                        qk = q[jmap]; // neighbor charge
                        dx = x[i][0] - x[k][0];
                        dy = x[i][1] - x[k][1];
                        dz = x[i][2] - x[k][2];
                        double rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                        rij = sqrt(rij2);
                        gams2 = gammaSqrt2[type[i]-1][type[jmap]-1];
                        delr = (2 / sqrt(M_PI) * (-exp(-pow(rij / sqrt2eta, 2))
                                                  / sqrt2eta + exp(-pow(rij / gams2, 2)) / gams2)
                                - 1 / rij * (erfc(rij / sqrt2eta) - erfc(rij / gams2))) / rij2;
                        jt0 += dx * delr * qk;
                        jt1 += dy * delr * qk;
                        jt2 += dz * delr * qk;
                    }

                } else {
                    /// Reciprocal Contribution
                    dx = x[i][0] - x[j][0];
                    dy = x[i][1] - x[j][1];
                    dz = x[i][2] - x[j][2];
                    double ksx = 0;
                    double ksy = 0;
                    double ksz = 0;
                    for (int kk = 0; kk < kcount; kk++) {
                        double kx = kxvecs[kk] * unitk[0];
                        double ky = kyvecs[kk] * unitk[1];
                        double kz = kzvecs[kk] * unitk[2];
                        double kdr = dx * kx + dy * ky + dz * kz;
                        ksx -= 2.0 * kcoeff[kk] * sin(kdr) * kx;
                        ksy -= 2.0 * kcoeff[kk] * sin(kdr) * ky;
                        ksz -= 2.0 * kcoeff[kk] * sin(kdr) * kz;
                    }
                    jt0 += ksx * qi;
                    jt1 += ksy * qi;
                    jt2 += ksz * qi;
                    /// Real Contribution
                    for (int jj = 0; jj < list->numneigh[j]; ++jj) {
                        k = list->firstneigh[j][jj]; // use k here as j was used above
                        k &= NEIGHMASK;
                        int jmap = atom->map(tag[k]); // local index of a global neighbor
                        if (jmap == i) {
                            //qk = q[jmap]; // neighbor charge
                            dx = x[k][0] - x[j][0];
                            dy = x[k][1] - x[j][1];
                            dz = x[k][2] - x[j][2];
                            double rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                            rij = sqrt(rij2);
                            if (rij >= real_cut) break;
                            gams2 = gammaSqrt2[type[i]-1][type[j]-1];
                            delr = (2 / sqrt(M_PI) * (-exp(-pow(rij / sqrt2eta, 2))
                                                      / sqrt2eta + exp(-pow(rij / gams2, 2)) / gams2)
                                    - 1 / rij * (erfc(rij / sqrt2eta) - erfc(rij / gams2))) / rij2;
                            jt0 += dx * delr * qi;
                            jt1 += dy * delr * qi;
                            jt2 += dz * delr * qi;
                        }
                    }
                }
                //pEelecpr[i][0] += 0.5 * qj * jt0;
                //pEelecpr[i][1] += 0.5 * qj * jt1;
                //pEelecpr[i][2] += 0.5 * qj * jt2;
                f[i][0] -= (forceLambda[j] * (jt0 + dChidxyz[j][0]) + 0.5 * qj * jt0);
                f[i][1] -= (forceLambda[j] * (jt1 + dChidxyz[j][1]) + 0.5 * qj * jt1);
                f[i][2] -= (forceLambda[j] * (jt2 + dChidxyz[j][2]) + 0.5 * qj * jt2);
            }
        } else {
            // Over all atoms in the system
            for (j = 0; j < nall; j++) {
                double jt0 = 0;
                double jt1 = 0;
                double jt2 = 0;
                qj = q[j];
                if (i == j) {
                    // We have to loop over all atoms once again to calculate dAdrQ terms
                    for (k = 0; k < nall; k++) {
                        if (k != i) {
                            qk = q[k];
                            dx = x[i][0] - x[k][0];
                            dy = x[i][1] - x[k][1];
                            dz = x[i][2] - x[k][2];
                            double rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                            rij = sqrt(rij2);

                            gams2 = gammaSqrt2[type[i]-1][type[k]-1];
                            delr = (2 / (sqrt(M_PI) * gams2) * exp(-pow(rij / gams2, 2)) - erf(rij / gams2) / rij);

                            jt0 += (dx / rij2) * delr * qk;
                            jt1 += (dy / rij2) * delr * qk;
                            jt2 += (dz / rij2) * delr * qk;
                        }
                    }
                } else {
                    dx = x[i][0] - x[j][0];
                    dy = x[i][1] - x[j][1];
                    dz = x[i][2] - x[j][2];
                    double rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                    rij = sqrt(rij2);

                    gams2 = gammaSqrt2[type[i]-1][type[j]-1];
                    delr = (2 / (sqrt(M_PI) * gams2) * exp(-pow(rij / gams2, 2)) - erf(rij / gams2) / rij);

                    jt0 = (dx / rij2) * delr * qi;
                    jt1 = (dy / rij2) * delr * qi;
                    jt2 = (dz / rij2) * delr * qi;
                }
                f[i][0] -= (forceLambda[j] * (jt0 + dChidxyz[j][0]) + 0.5 * qj * jt0);
                f[i][1] -= (forceLambda[j] * (jt1 + dChidxyz[j][1]) + 0.5 * qj * jt1);
                f[i][2] -= (forceLambda[j] * (jt2 + dChidxyz[j][2]) + 0.5 * qj * jt2);
            }
        }
    }

    if (periodic) deallocate_kspace(); //TODO: remove this, it is a workaround
}

// Re-initialize $\partial \Chi/\partial x_i$ array (derivatives of electronegativities w.r.t. positions)
// TODO: check, do we really need this ?
void PairNNP::reinitialize_dChidxyz()
{
    for (int i = 0; i < atom->nlocal; i++)
        for (int j = 0; j < 3; j++)
            dChidxyz[i][j] = 0.0;
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

// Calculate overall cutoff radius for 4G-neighlist
// TODO: not implemented yet
double PairNNP::getOverallCutoffRadius(double sfCutoff, int numAtoms)
{
    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd;

    double volume = xprd * yprd * zprd;
    double eta = 1.0 / sqrt(2.0 * M_PI);

    double precision = interface.getEwaldPrecision(); //TODO: remove this from fix_nnp?

    // Regular Ewald eta.
    if (numAtoms != 0) eta *= pow(volume * volume / numAtoms, 1.0 / 6.0);
        // Matrix version eta.
    else eta *= pow(volume, 1.0 / 3.0);
    //TODO: in RuNNer they take eta = max(eta, maxval(sigma))

    double rcutReal = sqrt(-2.0 * log(precision)) * eta;
    if (rcutReal > sfCutoff) return rcutReal;
    else return sfCutoff;

}

// Allocate k-space arrays
void PairNNP::allocate_kspace()
{
    int nloc = atom->nlocal;

    memory->create(kxvecs,kmax3d,"ewald:kxvecs");
    memory->create(kyvecs,kmax3d,"ewald:kyvecs");
    memory->create(kzvecs,kmax3d,"ewald:kzvecs");

    memory->create(kcoeff,kmax3d,"ewald:kcoeff");

    //memory->create(eg,kmax3d,3,"ewald:eg");
    //memory->create(vg,kmax3d,6,"ewald:vg");

    memory->create(sfexp_rl,kmax3d,nloc,"ewald:sfexp_rl");
    memory->create(sfexp_im,kmax3d,nloc,"ewald:sfexp_im");

    memory->create(sf_real,kmax3d,"ewald:sfexp_im");
    memory->create(sf_im,kmax3d,"ewald:sfexp_im");
    //sfacrl_all = new double[kmax3d];
    //sfacim_all = new double[kmax3d];
}

// Deallocate k-space arrays
void PairNNP::deallocate_kspace()
{
    memory->destroy(kxvecs);
    memory->destroy(kyvecs);
    memory->destroy(kzvecs);

    memory->destroy(kcoeff);
    //memory->destroy(eg);
    //memory->destroy(vg);

    memory->destroy(sfexp_rl);
    memory->destroy(sfexp_im);

    memory->destroy(sf_real);
    memory->destroy(sf_im);
    //delete [] sfacrl_all;
    //delete [] sfacim_all;
}

// Setup k-space grid and run necessary subroutines
void PairNNP::kspace_setup() {
    //allocate_kspace();
    //deallocate_kspace();

    // TODO: 'called initially and whenever the volume is changed' ? from LAMMPS version

    int natoms = atom->natoms;

    // WARNING: Immediately convert to NNP units!
    // volume-dependent factors
    double const xprd = domain->xprd * cflength;
    double const yprd = domain->yprd * cflength;
    double const zprd = domain->zprd * cflength;

    // adjustment of z dimension for 2d slab Ewald
    // 3d Ewald just uses zprd since slab_volfactor = 1.0

    //double zprd_slab = zprd*slab_volfactor;
    volume = xprd * yprd * zprd;

    unitk[0] = 2.0 * M_PI / xprd;
    unitk[1] = 2.0 * M_PI / yprd;
    unitk[2] = 2.0 * M_PI / zprd;
    fprintf(stderr, "unitk[0] = %24.16E\n", unitk[0]);
    fprintf(stderr, "unitk[1] = %24.16E\n", unitk[1]);
    fprintf(stderr, "unitk[2] = %24.16E\n", unitk[2]);
    //unitk[2] = 2.0*MY_PI/zprd_slab;

    ewaldEta = 1.0 / sqrt(2.0 * M_PI);
    // Regular Ewald eta.
    //if (natoms != 0) ewaldEta *= pow(volume * volume / natoms, 1.0 / 6.0);
    // Matrix version eta. ????
    //else ewaldEta *= pow(volume, 1.0 / 3.0);
    ewaldEta *= pow(volume, 1.0 / 3.0);
    //TODO: in RuNNer they take eta = max(eta, maxval(sigma))

    // Cutoff radii
    recip_cut = sqrt(-2.0 * log(ewaldPrecision)) / ewaldEta;
    real_cut = sqrt(-2.0 * log(ewaldPrecision)) * ewaldEta; // ???

    // TODO: calculate PBC copies
    int kmax_old = kmax;
    kspace_pbc(recip_cut); // get kxmax, kymax and kzmax

    kmax = MAX(kxmax, kymax);
    kmax = MAX(kmax, kzmax);
    kmax3d = 4 * kmax * kmax * kmax + 6 * kmax * kmax + 3 * kmax;

    double gsqxmx = unitk[0] * unitk[0] * kxmax * kxmax;
    double gsqymx = unitk[1] * unitk[1] * kymax * kymax;
    double gsqzmx = unitk[2] * unitk[2] * kzmax * kzmax;
    gsqmx = MAX(gsqxmx, gsqymx);
    gsqmx = MAX(gsqmx, gsqzmx);

    // LAMMPS cutoff criterion seems to be different, hence we feed now "our"
    // cutoff radius:
    gsqmx = recip_cut * recip_cut;
    //kmax3d = 9;

    // size change ?
    kxmax_orig = kxmax;
    kymax_orig = kymax;
    kzmax_orig = kzmax;

    deallocate_kspace();
    allocate_kspace();

    //TODO: if size has grown ?
    if (kmax > kmax_old) {
        //memory->destroy(ek);
        memory->destroy3d_offset(cs, -kmax_created);
        memory->destroy3d_offset(sn, -kmax_created);
        nmax = atom->nmax;
        //memory->create(ek,nmax,3,"ewald:ek");
        memory->create3d_offset(cs, -kmax, kmax, 3, nmax, "ewald:cs");
        memory->create3d_offset(sn, -kmax, kmax, 3, nmax, "ewald:sn");
        kmax_created = kmax;
    }

    kspace_coeffs();
    kspace_sfexp();

    std::cout << "Box vol :" << volume << '\n';
    std::cout << "Real cut :" << real_cut << '\n';
    fprintf(stderr, "Recip cut : %24.16E\n", recip_cut);
    fprintf(stderr, "KXMAX     : %d\n", kxmax);
    fprintf(stderr, "KYMAX     : %d\n", kymax);
    fprintf(stderr, "KZMAX     : %d\n", kzmax);
    fprintf(stderr, "kcount    : %d\n", kcount);
    std::cout << "KXMAX :" << kxmax << '\n';
    std::cout << "KYMAX :" << kymax << '\n';
    std::cout << "KZMAX :" << kzmax << '\n';
    std::cout << "kcount :" << kcount << '\n';

}

// Calculate k-space coefficients
//TODO: add vg and eg arrays if necessary (virial ?)
void PairNNP::kspace_coeffs()
{
    int k,l,m;
    double sqk;
    double preu = 4.0*M_PI/volume;
    double etasq = ewaldEta*ewaldEta;

    kcount = 0;

    fprintf(stderr, "gsqmx = %24.16E\n", sqrt(gsqmx));
    // (k,0,0), (0,l,0), (0,0,m)

    for (m = 1; m <= kmax; m++) {
        sqk = (m*unitk[0]) * (m*unitk[0]);
        if (sqk <= gsqmx) {
            fprintf(stderr, "sqk 1x = %24.16E, m %d\n", sqrt(sqk), m);
            kxvecs[kcount] = m;
            kyvecs[kcount] = 0;
            kzvecs[kcount] = 0;
            kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
            kcount++;
        }
        sqk = (m*unitk[1]) * (m*unitk[1]);
        if (sqk <= gsqmx) {
            fprintf(stderr, "sqk 1y = %24.16E, m %d\n", sqrt(sqk), m);
            kxvecs[kcount] = 0;
            kyvecs[kcount] = m;
            kzvecs[kcount] = 0;
            kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
            kcount++;
        }
        sqk = (m*unitk[2]) * (m*unitk[2]);
        if (sqk <= gsqmx) {
            fprintf(stderr, "sqk 1z = %24.16E, m %d\n", sqrt(sqk), m);
            kxvecs[kcount] = 0;
            kyvecs[kcount] = 0;
            kzvecs[kcount] = m;
            kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
            kcount++;
        }
    }

    // 1 = (k,l,0), 2 = (k,-l,0)

    for (k = 1; k <= kxmax; k++) {
        for (l = 1; l <= kymax; l++) {
            sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l);
            if (sqk <= gsqmx) {
                fprintf(stderr, "sqk 2 = %24.16E, k %d  l %d\n", sqrt(sqk), k, l);
                kxvecs[kcount] = k;
                kyvecs[kcount] = l;
                kzvecs[kcount] = 0;
                kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                kcount++;

                fprintf(stderr, "sqk 2 = %24.16E, k %d -l %d\n", sqrt(sqk), k, l);
                kxvecs[kcount] = k;
                kyvecs[kcount] = -l;
                kzvecs[kcount] = 0;
                kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                kcount++;;
            }
        }
    }

    // 1 = (0,l,m), 2 = (0,l,-m)

    for (l = 1; l <= kymax; l++) {
        for (m = 1; m <= kzmax; m++) {
            sqk = (unitk[1]*l) * (unitk[1]*l) + (unitk[2]*m) * (unitk[2]*m);
            if (sqk <= gsqmx) {
                fprintf(stderr, "sqk 3 = %24.16E, l %d  m %d\n", sqrt(sqk), l, m);
                kxvecs[kcount] = 0;
                kyvecs[kcount] = l;
                kzvecs[kcount] = m;
                kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                kcount++;

                fprintf(stderr, "sqk 3 = %24.16E, l %d -m %d\n", sqrt(sqk), l, m);
                kxvecs[kcount] = 0;
                kyvecs[kcount] = l;
                kzvecs[kcount] = -m;
                kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                kcount++;
            }
        }
    }

    // 1 = (k,0,m), 2 = (k,0,-m)

    for (k = 1; k <= kxmax; k++) {
        for (m = 1; m <= kzmax; m++) {
            sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[2]*m) * (unitk[2]*m);
            if (sqk <= gsqmx) {
                fprintf(stderr, "sqk 4 = %24.16E, k %d  m %d\n", sqrt(sqk), k, m);
                kxvecs[kcount] = k;
                kyvecs[kcount] = 0;
                kzvecs[kcount] = m;
                kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                kcount++;

                fprintf(stderr, "sqk 4 = %24.16E, k %d -m %d\n", sqrt(sqk), k, m);
                kxvecs[kcount] = k;
                kyvecs[kcount] = 0;
                kzvecs[kcount] = -m;
                kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                kcount++;
            }
        }
    }

    // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

    for (k = 1; k <= kxmax; k++) {
        for (l = 1; l <= kymax; l++) {
            for (m = 1; m <= kzmax; m++) {
                sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l) +
                      (unitk[2]*m) * (unitk[2]*m);
                if (sqk <= gsqmx) {
                    fprintf(stderr, "sqk 5 = %24.16E, k %d  l %d  m %d\n", sqrt(sqk), k, l, m);
                    kxvecs[kcount] = k;
                    kyvecs[kcount] = l;
                    kzvecs[kcount] = m;
                    kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                    kcount++;

                    fprintf(stderr, "sqk 5 = %24.16E, k %d -l %d  m %d\n", sqrt(sqk), k, l, m);
                    kxvecs[kcount] = k;
                    kyvecs[kcount] = -l;
                    kzvecs[kcount] = m;
                    kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                    kcount++;

                    fprintf(stderr, "sqk 5 = %24.16E, k %d  l %d -m %d\n", sqrt(sqk), k, l, m);
                    kxvecs[kcount] = k;
                    kyvecs[kcount] = l;
                    kzvecs[kcount] = -m;
                    kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                    kcount++;

                    fprintf(stderr, "sqk 5 = %24.16E, k %d -l %d -m %d\n", sqrt(sqk), k, l, m);
                    kxvecs[kcount] = k;
                    kyvecs[kcount] = -l;
                    kzvecs[kcount] = -m;
                    kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                    kcount++;
                }
            }
        }
    }

}

// Calculate k-space structure factor
void PairNNP::kspace_sfexp()
{
    int i,k,l,m,n,ic;
    double sqk,clpm,slpm;

    double **x = atom->x;
    int nlocal = atom->nlocal;

    n = 0;

    // (k,0,0), (0,l,0), (0,0,m)

    for (ic = 0; ic < 3; ic++) {
        sqk = unitk[ic]*unitk[ic];
        if (sqk <= gsqmx) {
            for (i = 0; i < nlocal; i++) {
                cs[0][ic][i] = 1.0;
                sn[0][ic][i] = 0.0;
                cs[1][ic][i] = cos(unitk[ic]*x[i][ic]*cflength);
                sn[1][ic][i] = sin(unitk[ic]*x[i][ic]*cflength);
                cs[-1][ic][i] = cs[1][ic][i];
                sn[-1][ic][i] = -sn[1][ic][i];
                sfexp_rl[n][i] = cs[1][ic][i];
                sfexp_im[n][i] = sn[1][ic][i];
            }
            n++;
        }
    }

    for (m = 2; m <= kmax; m++) {
        for (ic = 0; ic < 3; ic++) {
            sqk = m*unitk[ic] * m*unitk[ic];
            if (sqk <= gsqmx) {
                for (i = 0; i < nlocal; i++) {
                    cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
                                   sn[m-1][ic][i]*sn[1][ic][i];
                    sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
                                   cs[m-1][ic][i]*sn[1][ic][i];
                    cs[-m][ic][i] = cs[m][ic][i];
                    sn[-m][ic][i] = -sn[m][ic][i];
                    sfexp_rl[n][i] = cs[m][ic][i];
                    sfexp_im[n][i] = sn[m][ic][i];
                }
                n++;
            }
        }
    }

    // 1 = (k,l,0), 2 = (k,-l,0)

    for (k = 1; k <= kxmax; k++) {
        for (l = 1; l <= kymax; l++) {
            sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
            if (sqk <= gsqmx) {
                for (i = 0; i < nlocal; i++) {
                    sfexp_rl[n][i] = (cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i]);
                    sfexp_im[n][i] = (sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i]);
                }
                n++;
                for (i = 0; i < nlocal; i++) {
                    sfexp_rl[n][i] = (cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i]);
                    sfexp_im[n][i] = (sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i]);
                }
                n++;
            }
        }
    }

    // 1 = (0,l,m), 2 = (0,l,-m)

    for (l = 1; l <= kymax; l++) {
        for (m = 1; m <= kzmax; m++) {
            sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
            if (sqk <= gsqmx) {
                for (i = 0; i < nlocal; i++) {
                    sfexp_rl[n][i] = (cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i]);
                    sfexp_im[n][i] = (sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i]);
                }
                n++;
                for (i = 0; i < nlocal; i++) {
                    sfexp_rl[n][i] = (cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i]);
                    sfexp_im[n][i] = (sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i]);
                }
                n++;
            }
        }
    }

    // 1 = (k,0,m), 2 = (k,0,-m)

    for (k = 1; k <= kxmax; k++) {
        for (m = 1; m <= kzmax; m++) {
            sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
            if (sqk <= gsqmx) {
                for (i = 0; i < nlocal; i++) {
                    sfexp_rl[n][i] = (cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i]);
                    sfexp_im[n][i] = (sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i]);
                }
                n++;
                for (i = 0; i < nlocal; i++) {
                    sfexp_rl[n][i] = (cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i]);
                    sfexp_im[n][i] = (sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i]);
                }
                n++;
            }
        }
    }

    // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

    for (k = 1; k <= kxmax; k++) {
        for (l = 1; l <= kymax; l++) {
            for (m = 1; m <= kzmax; m++) {
                sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
                      (m*unitk[2] * m*unitk[2]);
                if (sqk <= gsqmx) {
                    for (i = 0; i < nlocal; i++) {
                        clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
                        slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
                        sfexp_rl[n][i] = (cs[k][0][i]*clpm - sn[k][0][i]*slpm);
                        sfexp_im[n][i] = (sn[k][0][i]*clpm + cs[k][0][i]*slpm);
                    }
                    n++;
                    for (i = 0; i < nlocal; i++) {
                        clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
                        slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
                        sfexp_rl[n][i] = (cs[k][0][i]*clpm - sn[k][0][i]*slpm);
                        sfexp_im[n][i] = (sn[k][0][i]*clpm + cs[k][0][i]*slpm);
                    }
                    n++;
                    for (i = 0; i < nlocal; i++) {
                        clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
                        slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
                        sfexp_rl[n][i] = (cs[k][0][i]*clpm - sn[k][0][i]*slpm);
                        sfexp_im[n][i] = (sn[k][0][i]*clpm + cs[k][0][i]*slpm);
                    }
                    n++;
                    for (i = 0; i < nlocal; i++) {
                        clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
                        slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
                        sfexp_rl[n][i] = (cs[k][0][i]*clpm - sn[k][0][i]*slpm);
                        sfexp_im[n][i] = (sn[k][0][i]*clpm + cs[k][0][i]*slpm);
                    }
                    n++;
                }
            }
        }
    }
}

// Generate k-space grid
void PairNNP::kspace_pbc(double rcut)
{

    double proja = fabs(unitk[0]);
    double projb = fabs(unitk[1]);
    double projc = fabs(unitk[2]);
    kxmax = 0;
    kymax = 0;
    kzmax = 0;
    while (kxmax * proja <= rcut) kxmax++;
    while (kymax * projb <= rcut) kymax++;
    while (kzmax * projc <= rcut) kzmax++;

    return;
}

//TODO: check, this was the original subroutine in LAMMPS that sets the K-space grid
double PairNNP::kspace_rms(int km, double prd, bigint natoms, double q2)
{

    /*if (natoms == 0) natoms = 1;   // avoid division by zero
    double value = 2.0*q2*g_ewald/prd *
                   sqrt(1.0/(M_PI*km*natoms)) *
                   exp(-M_PI*M_PI*km*km/(g_ewald*g_ewald*prd*prd));

    return value;
     */
    return 0.0;
}
