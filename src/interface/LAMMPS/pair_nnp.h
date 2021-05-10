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

    // QEq arrays
    double *chi,*hardness,*sigma;

    double eElec; // electrostatic contribution to total energy (calculated in fix_nnp.cpp

    double *dEdQ,*forceLambda;
    double **dChidxyz,**pEelecpr;

    bool isPeriodic;
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

    virtual void allocate();
    void transferNeighborList();

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
    void calculateForceLambda();
    void calculateElecDerivatives(double*);
    void calculateElecForceTerm(double**);
    void initializeChi();

    double *screenInfo; // info array
    double screen_f(double);
    double screen_df(double);

};

}

#endif
#endif
