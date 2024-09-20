// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifdef PAIR_CLASS

PairStyle(nnp,PairNNP)

#else

#ifndef LMP_PAIR_NNP_H
#define LMP_PAIR_NNP_H

#include "pair.h"
#include "InterfaceLammps.h"
#include <gsl/gsl_multimin.h>

namespace LAMMPS_NS {

class PairNNP : public Pair {
    friend class FixNNP;
    friend class KSpaceNNP;
 public:

  PairNNP(class LAMMPS *);
  virtual ~PairNNP();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  virtual void coeff(int, char **);
  virtual void init_style();
  virtual double init_one(int, int);
  void init_list(int,class NeighList *);
  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);


protected:

    class KSpaceNNP *kspacennp; // interface to NNP kspace_style

    int me,nprocs;
    bool periodic;
    bool showew;
    bool resetew;
    int showewsum;
    int maxew;
    long numExtrapolationWarningsTotal;
    long numExtrapolationWarningsSummary;
    double cflength;
    double cfenergy;
    double maxCutoffRadius;
    char* directory;
    char* emap;
    class NeighList *list;

    nnp::InterfaceLammps interface;

    double *chi,*hardness,*sigmaSqrtPi,**gammaSqrt2; // QEq arrays
    double *dEdQ,*forceLambda;
    double grad_tol,min_tol,step; // user-defined minimization parameters
    int maxit;
    int minim_init_style; // initialization style for the minimization algorithm, 0: from zero or 1: from the final step

    virtual void allocate();
    void transferNeighborList();
    void isPeriodic();

    void transferCharges();
    void handleExtrapolationWarnings();

    void deallocateQEq();

    // Minimization Setup for Force Lambda
    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;

    gsl_multimin_function_fdf forceLambda_minimizer;

    double forceLambda_f(const gsl_vector*);
    void forceLambda_df(const gsl_vector*, gsl_vector*);
    void forceLambda_fdf(const gsl_vector*, double*, gsl_vector*);
    static double forceLambda_f_wrap(const gsl_vector*, void*);
    static void forceLambda_df_wrap(const gsl_vector*, void*, gsl_vector*);
    static void forceLambda_fdf_wrap(const gsl_vector*, void*, double*, gsl_vector*);

    double E_elec;
    double kcoeff_sum; // used in dEdQ calculation

    int nmax;

    int *type_all,*type_loc;
    double *dEdLambda_loc,*dEdLambda_all;
    double *qall_loc,*qall;
    double *xx,*xy,*xz; // global positions
    double *xx_loc,*xy_loc,*xz_loc; // sparse local positions
    double *forceLambda_loc,*forceLambda_all;

    double **erfc_val;
    double **kcos,**ksinx,**ksiny,**ksinz;

    void calculateForceLambda();
    void calculateElecDerivatives(double*,double**);
    void calculateElecForce(double**);
    void calculate_kspace_terms();


    // Screening
    double *screening_info;
    double screening_f(double);
    double screening_df(double);

};

}

#endif
#endif
