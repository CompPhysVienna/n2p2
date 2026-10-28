/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   This file initially came from n2p2 (https://github.com/CompPhysVienna/n2p2)
   Copyright (2018) Andreas Singraber (University of Vienna)

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Emir Kocer
                         Andreas Singraber
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(hdnnp/4g,PairHDNNP4G)
// clang-format on
#else

#ifndef LMP_PAIR_HDNNP4G_H
#define LMP_PAIR_HDNNP4G_H

#include "pair.h"
#include "InterfaceLammps.h"
#include <gsl/gsl_vector.h>
#include <cmath>

namespace LAMMPS_NS {

/* Conjugate gradient solution of the minimum of the quadratic function
   1/2 x^T A x + b^T x (A symmetric positive definite) with the constraint
   that the sum of x is fixed (charge equilibration, force lambda).
   gradient(v, g) must store the gradient projected on the constraint,
   g = P (A v + b) with P removing the mean, as a global vector on all MPI
   processes. Since gradient(v) - gradient(0) = P A v, each iteration needs one
   gradient evaluation. On entry x is the initial guess (its sum is kept), on
   exit the solution. Stops when |g| < tolerance or after maxit iterations;
   returns the number of iterations and |g| in norm. */
template <typename Gradient>
int projectedConjugateGradient(Gradient gradient, gsl_vector *x,
                               double tolerance, int maxit, double *norm)
{
    size_t const n = x->size;
    gsl_vector *zero = gsl_vector_calloc(n);
    gsl_vector *b = gsl_vector_alloc(n);
    gsl_vector *g = gsl_vector_alloc(n);
    gsl_vector *p = gsl_vector_alloc(n);
    gsl_vector *ap = gsl_vector_alloc(n);

    gradient(zero, b); // P b
    gradient(x, g);
    double gg = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double const gi = gsl_vector_get(g, i);
        gsl_vector_set(p, i, -gi);
        gg += gi * gi;
    }
    int iter = 0;
    while (std::sqrt(gg) >= tolerance && iter < maxit)
    {
        iter++;
        gradient(p, ap);
        double pap = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double const api = gsl_vector_get(ap, i) - gsl_vector_get(b, i);
            gsl_vector_set(ap, i, api); // P A p
            pap += gsl_vector_get(p, i) * api;
        }
        if (pap <= 0.0) break; // solution reached within round-off
        double const alpha = gg / pap;
        double ggNew = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            gsl_vector_set(x, i, gsl_vector_get(x, i)
                                 + alpha * gsl_vector_get(p, i));
            double const gi = gsl_vector_get(g, i)
                            + alpha * gsl_vector_get(ap, i);
            gsl_vector_set(g, i, gi);
            ggNew += gi * gi;
        }
        double const beta = ggNew / gg;
        gg = ggNew;
        for (size_t i = 0; i < n; ++i)
        {
            gsl_vector_set(p, i, -gsl_vector_get(g, i)
                                 + beta * gsl_vector_get(p, i));
        }
    }
    *norm = std::sqrt(gg);

    gsl_vector_free(zero);
    gsl_vector_free(b);
    gsl_vector_free(g);
    gsl_vector_free(p);
    gsl_vector_free(ap);

    return iter;
}

class PairHDNNP4G : public Pair {
    friend class FixHDNNP;
    friend class KSpaceHDNNP;
 public:

  PairHDNNP4G(class LAMMPS *);
  virtual ~PairHDNNP4G();
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

    class KSpaceHDNNP *kspacehdnnp; // interface to HDNNP kspace_style

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
    // cfenergy times 4 pi epsilon in the units of the training data: energy
    // conversion of the electrostatics, which are evaluated with
    // 4 pi epsilon = 1 (see init_style()).
    double cfenergyElec;
    double maxCutoffRadius;
    char* directory;
    char* emap;
    class NeighList *list;
    // Neighbor list for the real-space electrostatics (owned by fix hdnnp)
    // and its cutoff in LAMMPS length units, see init_style().
    class NeighList *elecList;
    double elecCutoff;
    double E_recip_global;

    nnp::InterfaceLammps interface;

    double *chi,*hardness,*sigmaSqrtPi,**gammaSqrt2; // QEq arrays
    double *dEdQ,*forceLambda;
    double grad_tol; // convergence criterion of the charge and force lambda solvers (norm of the projected gradient)
    double min_tol,step; // parameters of the former GSL minimizer, read by fix hdnnp, unused (deprecated, warning)
    int maxit;
    int minim_init_style; // initialization style for the minimization algorithm, 0: from zero or 1: from the final step

    virtual void allocate();
    void transferNeighborList();
    void isPeriodic();
    void setElecCutoff();
    void requestNeighborList();

    void transferCharges();
    void handleExtrapolationWarnings();

    void deallocateQEq();

    // Force lambda: minimum of 1/2 lambda^T A lambda + dEdQ^T lambda with
    // sum(lambda) = 0, solved by projectedConjugateGradient().
    void forceLambda_structureFactors(const gsl_vector*); // reciprocal-space terms used by forceLambda_df
    void forceLambda_df(const gsl_vector*, gsl_vector*); // projected gradient

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
