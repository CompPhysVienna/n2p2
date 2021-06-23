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

    class FixNNP *fix_nnp;

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

    double *chi,*hardness,*sigma; // QEq arrays
    double eElec; // electrostatic contribution to total energy (calculated in fix_nnp.cpp
    double *dEdQ,*forceLambda,**dChidxyz;
    double overallCutoff; // TODO
    double grad_tol,min_tol,step; // user-defined minimization parameters
    int maxit;

    virtual void allocate();
    void transferNeighborList();
    void isPeriodic();
    double getOverallCutoffRadius(double, int natoms = 0); // TODO

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

    // Electrostatics
    double gsqmx,volume;
    double unitk[3];
    double q2,g_ewald;
    double ewaldPrecision; // 'accuracy' in LAMMPS
    double ewaldEta; //  '1/g_ewald' in LAMMPS
    double recip_cut,real_cut;
    double E_elec;

    int *kxvecs,*kyvecs,*kzvecs;
    int kxmax_orig,kymax_orig,kzmax_orig,kmax_created;
    int kxmax,kymax,kzmax,kmax,kmax3d;
    int kcount;
    int nmax;
    double *kcoeff;
    double **eg,**vg; // forces and virial
    double **ek; // forces ?
    double **sfexp_rl,**sfexp_im;
    double **sfexp_rl_all,*sfexp_im_all; // structure factors after communications ?
    double ***cs,***sn; // cosine and sine grid, TODO: should we change this ?

    void calculateForceLambda();
    void calculateElecDerivatives(double*,double**);
    void calculateElecForce(double**);
    void reinitialize_dChidxyz();

    void kspace_setup();
    void kspace_coeffs();
    void kspace_sfexp();
    void kspace_pbc(double);
    double kspace_rms(int, double, bigint, double);

    void allocate_kspace();
    void deallocate_kspace();

    // Screening
    double *screening_info;
    double screening_f(double);
    double screening_df(double);

};

}

#endif
#endif
