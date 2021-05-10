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


#define SQR(x) ((x)*(x))
#define MIN_NBRS 100
#define MIN_CAP  50
#define DANGER_ZONE    0.90
#define SAFE_ZONE      1.2


using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairNNP::PairNNP(LAMMPS *lmp) : Pair(lmp)
{

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

  // TODO: cleanup
  if (interface.getNnpType() == 2) //2G-HDNNPs
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

  }else if (interface.getNnpType() == 4) //4G-HDNNPs
  {
      transferCharges();

      // run second NN for the short range contributions
      interface.process();

      // get short-range forces of local and ghost atoms.
      interface.getForces(atom->f);

      // calculates dEelecdQ vector
      calculateElecDerivatives(dEdQ);

      // Add dEdG term from n2p2
      interface.getdEdQ(dEdQ); //TODO: dEdG slightly different, normal ?

      for(int i=0; i < atom->nlocal; i++)
      {
          std::cout << "Short : " << '\t' << atom->f[i][0] << '\t' << atom->f[i][1] << '\t' << atom->f[i][2] << '\n';
      }

      // Calculates lambda vector that is necessary for force calculation using minimization
      calculateForceLambda();

      for(int i=0; i < atom->nlocal; i++)
      {
          //std::cout << forceLambda[i] << '\n';
      }

      //TODO: these are also different, error accumulation ??
      calculateElecForceTerm(atom->f);

      for(int i=0; i < atom->nlocal; i++)
      {
          std::cout << "Elec : " << '\t' << atom->f[i][0] << '\t' << atom->f[i][1] << '\t' << atom->f[i][2] << '\n';
      }

      exit(0);


      // Do all stuff related to extrapolation warnings.
      if(showew == true || showewsum > 0 || maxew >= 0) {
          handleExtrapolationWarnings();
      }

      // get short-range forces of local and ghost atoms.
      interface.getForces(atom->f);


  }
  // Add energy contribution to total energy.
  if (eflag_global)
     ev_tally(0,0,atom->nlocal,1,interface.getEnergy(),eElec,0.0,0.0,0.0,0.0);

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

  isPeriodic = false; //TODO: add a proper check

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

  //if(interface.getNnpType() == 4) ///TODO: this if does not work in this way, change initialize()
  {
      dEdQ = NULL;
      forceLambda = NULL;

      memory->create(dEdQ,natoms+1,"pair:dEdQ");
      memory->create(pEelecpr,natoms+1,3,"pair:pEelecpr");
      memory->create(forceLambda,natoms+1,"pair:forceLambda");
      memory->create(dChidxyz,natoms+1,3,"pair:dChidxyz");
      for (int i = 0; i < natoms+1; i++)
      {
          forceLambda[i] = 0.0;
          dEdQ[i] = 0.0;
          for (int j = 0; j < 3; j++)
              pEelecpr[i][j] = 0.0;
      }
  }

}

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

// TODO: is indexing correct ?
void PairNNP::transferCharges()
{
    // Transfer charges into n2p2
    for (int i = 0; i < atom->nlocal; ++i) {
        interface.addCharge(i,atom->q[i]);
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

//// Minimization - forceLambda
double PairNNP::forceLambda_f(const gsl_vector *v)
{
    int i,j;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;

    double **x = atom->x;
    double dx, dy, dz, rij;
    double lambda_i,lambda_j;
    double E_lambda;
    double iiterm,ijterm;

    // TODO: indices
    E_lambda = 0.0;
    // first loop over local atoms
    for (i = 0; i < nlocal; i++) {
        lambda_i = gsl_vector_get(v,i);
        // add i terms here
        iiterm = lambda_i * lambda_i / (2.0 * sigma[i] * sqrt(M_PI));
        E_lambda += iiterm + dEdQ[i]*lambda_i + 0.5*hardness[i]*lambda_i*lambda_i;
        // second loop over 'all' atoms
        for (j = i + 1; j < nall; j++) {
            lambda_j = gsl_vector_get(v, j);
            dx = x[j][0] - x[i][0];
            dy = x[j][1] - x[i][1];
            dz = x[j][2] - x[i][2];
            rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
            ijterm = lambda_i * lambda_j * (erf(rij / sqrt(2.0 *
                    (pow(sigma[i], 2) + pow(sigma[j], 2)))) / rij);
            E_lambda += ijterm;
        }
    }

    return E_lambda;
}

double PairNNP::forceLambda_f_wrap(const gsl_vector *v, void *params)
{
    return static_cast<PairNNP*>(params)->forceLambda_f(v);
}

void PairNNP::forceLambda_df(const gsl_vector *v, gsl_vector *dEdLambda)
{
    int i,j;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;

    double dx, dy, dz, rij;
    double lambda_i,lambda_j;
    double **x = atom->x;

    double val;
    double grad_sum,grad_i;
    double local_sum;

    grad_sum = 0.0;
    // first loop over local atoms
    for (i = 0; i < nlocal; i++) { // TODO: indices
        lambda_i = gsl_vector_get(v,i);
        local_sum = 0.0;
        // second loop over 'all' atoms
        for (j = 0; j < nall; j++) {
            if (j != i) {
                lambda_j = gsl_vector_get(v, j);
                dx = x[j][0] - x[i][0];
                dy = x[j][1] - x[i][1];
                dz = x[j][2] - x[i][2];
                rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
                local_sum += lambda_j * erf(rij / sqrt(2.0 *
                        (pow(sigma[i], 2) + pow(sigma[j], 2)))) / rij;
            }
        }
        val = dEdQ[i] + hardness[i]*lambda_i +
                lambda_i/(sigma[i]*sqrt(M_PI)) + local_sum;
        grad_sum = grad_sum + val;
        gsl_vector_set(dEdLambda,i,val);
    }

    // Gradient projection //TODO: communication ?
    for (i = 0; i < nall; i++){
        //grad_i = gsl_vector_get(dEdLambda,i);
        //gsl_vector_set(dEdLambda,i,grad_i - (grad_sum)/nall);
    }

}

void PairNNP::forceLambda_df_wrap(const gsl_vector *v, void *params, gsl_vector *df)
{
    static_cast<PairNNP*>(params)->forceLambda_df(v, df);
}

void PairNNP::forceLambda_fdf(const gsl_vector *v, double *f, gsl_vector *df)
{
    *f = forceLambda_f(v);
    forceLambda_df(v, df);
}

void PairNNP::forceLambda_fdf_wrap(const gsl_vector *v, void *params, double *f, gsl_vector *df)
{
    static_cast<PairNNP*>(params)->forceLambda_fdf(v, f, df);
}

void PairNNP::calculateForceLambda()
{
    size_t iter = 0;
    int i,j;
    int nsize,maxit,status;

    double psum_it;
    double gradsum;
    double lambda_i;
    double grad_tol,min_tol,step;

    // TODO: backward/forward communication ??

    nsize = atom->natoms;
    gsl_vector *x; // charge vector in our case

    forceLambda_minimizer.n = nsize;
    forceLambda_minimizer.f = &forceLambda_f_wrap;
    forceLambda_minimizer.df = &forceLambda_df_wrap;
    forceLambda_minimizer.fdf = &forceLambda_fdf_wrap;
    forceLambda_minimizer.params = this;


    // TODO:how should we initialize these ?
    x = gsl_vector_alloc(nsize);
    for (i = 0; i < nsize; i++) {
        gsl_vector_set(x,i,forceLambda[i]);
    }

    //T = gsl_multimin_fdfminimizer_conjugate_fr;
    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T, nsize);

    // Minimizer Params TODO: user-defined ?
    grad_tol = 1e-5;
    min_tol = 1e-5;
    step = 1e-2;
    maxit = 100;

    gsl_multimin_fdfminimizer_set(s, &forceLambda_minimizer, x, step, min_tol); // tol = 0 might be expensive ???
    do
    {
        iter++;
        psum_it = 0.0;
        std::cout << "iter : " << iter << '\n';
        std::cout << "-------------" << '\n';

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
            printf ("Minimum found\n");

    }
    while (status == GSL_CONTINUE && iter < maxit);

    // read charges before deallocating x - be careful with indices !
    for (i = 0; i < nsize; i++) {
        forceLambda[i] = gsl_vector_get(s->x,i);
    }
    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);
}


//// Electrostatic Force
void PairNNP::calculateElecDerivatives(double *dEelecdQ)
{

    int i,j;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;

    double **x = atom->x;
    double *q = atom->q;
    double dx,dy,dz;
    double rij,erfrij;
    double gams2;
    double sij,tij;
    double fsrij,dfsrij; // corrections due to screening

    // TODO: indices and parallelization
    for (i = 0; i < nlocal; i++)
    {
        for (j = 0; j < nall; j++)
        {
            dx = x[i][0] - x[j][0];
            dy = x[i][1] - x[j][1];
            dz = x[i][2] - x[j][2];
            rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
            gams2 = sqrt(2.0 * (pow(sigma[i], 2) + pow(sigma[j], 2)));
            // Diagonal terms contain self-interaction
            if (i != j)
            {
                erfrij = erf(rij / gams2);
                fsrij = screen_f(rij);
                dfsrij = screen_df(rij);

                sij = erfrij * (fsrij - 1) / rij;
                tij = (2 / (sqrt(M_PI)*gams2) * exp(- pow(rij / gams2,2)) *
                        (fsrij - 1) + erfrij*dfsrij - erfrij*(fsrij-1)/rij);

                dEelecdQ[i] += q[j] * ((erfrij/rij) + sij);

                pEelecpr[i][0] += q[i] * q[j] * (dx/pow(rij,2)) * tij;
                pEelecpr[i][1] += q[i] * q[j] * (dy/pow(rij,2)) * tij;
                pEelecpr[i][2] += q[i] * q[j] * (dz/pow(rij,2)) * tij;

            }
            else if (isPeriodic) //TODO: add later
            {
                //dEelecdQ[i] += atom->q[i] * (0.0- hardness[i] - 1 / (sigma[i]*sqrt(M_PI));
            }
        }
    }
}

void PairNNP::calculateElecForceTerm(double **f)
{

    int i,j,k;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;

    double rij,rij2;
    double qi,qj,qk;
    double *q = atom->q;
    double **x = atom->x;
    double delr,gams2;
    double dx,dy,dz;

    double jterm0,jterm1,jterm2;


    // TODO: check indices, consider parallelization..
    for (i = 0; i < nlocal; i++) {
        qi = q[i];
        initializeChi();
        interface.getdChidxyz(i, dChidxyz);
        // second loop over 'all' atoms
        //TODO: check this
        for (j = 0; j < nall; j++) {
            jterm0 = 0;
            jterm1 = 0;
            jterm2 = 0;
            qj = q[j];
            if (i == j)
            {
                // We have to loop over all atoms once again to calculate dAdrQ terms
                for (k = 0; k < nall; k++)
                {
                    if (k != i)
                    {
                        qk = q[k];
                        dx = x[i][0] - x[k][0];
                        dy = x[i][1] - x[k][1];
                        dz = x[i][2] - x[k][2];
                        rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                        rij = sqrt(rij2);

                        gams2 = sqrt(2.0 * (pow(sigma[i], 2) + pow(sigma[k], 2)));
                        delr = (2 / (sqrt(M_PI) * gams2) * exp(-pow(rij / gams2, 2)) - erf(rij / gams2) / rij);

                        jterm0 += (dx/rij2) * delr * qk;
                        jterm1 += (dy/rij2) * delr * qk;
                        jterm2 += (dz/rij2) * delr * qk;
                    }
                }

            }else
            {
                dx = x[i][0] - x[j][0];
                dy = x[i][1] - x[j][1];
                dz = x[i][2] - x[j][2];
                rij2 = SQR(dx) + SQR(dy) + SQR(dz);
                rij = sqrt(rij2);

                gams2 = sqrt(2.0 * (pow(sigma[i], 2) + pow(sigma[j], 2)));
                delr = (2 / (sqrt(M_PI) * gams2) * exp(-pow(rij / gams2, 2)) - erf(rij / gams2) / rij);

                jterm0 = (dx/rij2) * delr * qi;
                jterm1 = (dy/rij2) * delr * qi;
                jterm2 = (dz/rij2) * delr * qi;
            }
            //f[i][0] -= forceLambda[k] * (delr*drdx*qj + dChidxyz[k][0]);
            //f[i][1] -= forceLambda[k] * (delr*drdy*qj + dChidxyz[k][1]);
            //f[i][2] -= forceLambda[k] * (delr*drdz*qj + dChidxyz[k][2]);
            pEelecpr[i][0] += 0.5 * qj * jterm0;
            pEelecpr[i][1] += 0.5 * qj * jterm1;
            pEelecpr[i][2] += 0.5 * qj * jterm2;
            f[i][0] -= (jterm0 + dChidxyz[j][0]);
            f[i][1] -= (jterm1 + dChidxyz[j][1]);
            f[i][2] -= (jterm2 + dChidxyz[j][2]);
        }
    }
}

void PairNNP::initializeChi()
{
    for (int i = 0; i < atom->nlocal; i++)
        for (int j = 0; j < 3; j++)
            dChidxyz[i][j] = 0.0;
}

// TODO : add other function types (only cos now)
double PairNNP::screen_f(double r)
{
    double x;

    if (r >= screenInfo[2]) return 1.0;
    else if (r <= screenInfo[1]) return 0.0;
    else
    {
        x = (r-screenInfo[1])*screenInfo[3];
        return 1.0 - 0.5*(cos(M_PI*x)+1);
    }

}
/* ---------------------------------------------------------------------- */

// TODO : add other function types (only cos now)
double PairNNP::screen_df(double r)
{
    double x;

    if (r >= screenInfo[2] || r <= screenInfo[1]) return 0.0;
    else
    {
        x = (r-screenInfo[1])*screenInfo[3];
        return -screenInfo[3] * ( -M_PI_2 * sin(M_PI*x));
    }
}

