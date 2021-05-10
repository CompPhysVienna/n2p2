/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
#ifdef FIX_CLASS

FixStyle(nnp,FixNNP)

#else

#ifndef LMP_FIX_NNP_H
#define LMP_FIX_NNP_H

#include <gsl/gsl_multimin.h>
#include "fix.h"
#include "InterfaceLammps.h"


namespace LAMMPS_NS {

    class FixNNP : public Fix {
        friend class PairNNP;
    public:
        FixNNP(class LAMMPS *, int, char **);
        ~FixNNP();

        int setmask();
        virtual void post_constructor();
        virtual void init();
        void init_list(int,class NeighList *);
        void setup_pre_force(int);
        virtual void pre_force(int);

        //void setup_pre_force_respa(int, int);
        //void pre_force_respa(int, int, int);

        void min_setup_pre_force(int);
        void min_pre_force(int);

        int matvecs;
        double qeq_time;

    protected:
        int nevery,nnpflag;
        int n, N, m_fill;
        int n_cap, nmax, m_cap;
        int pack_flag;
        class NeighList *list;
        class PairNNP *nnp;


        double tolerance;     // tolerance for the norm of the rel residual in CG
        bool periodic; // check for periodicity

        bigint ngroup;
        int nprev,dum1,dum2;

        // charges
        double *Q;
        double **Q_hist; // do we need this ???

        void allocate_QEq();
        void deallocate_QEq();
        double qRef;

        void calculate_electronegativities();

        void run_network();

        //TODO:to be removed
        char *pertype_option;
        virtual void pertype_parameters(char*);

        //// Function Minimization Approach

        double *xall,*yall,*zall; // all atoms (parallelization)

        gsl_multimin_function_fdf QEq_minimizer; // find a better name



        const gsl_multimin_fdfminimizer_type *T;
        gsl_multimin_fdfminimizer *s;

        // Minimization Setup for QEq energy
        double QEq_f(const gsl_vector*);
        void QEq_df(const gsl_vector*, gsl_vector*);
        void QEq_fdf(const gsl_vector*, double*, gsl_vector*);

        static double QEq_f_wrap(const gsl_vector*, void*);
        static void QEq_df_wrap(const gsl_vector*, void*, gsl_vector*);
        static void QEq_fdf_wrap(const gsl_vector*, void*, double*, gsl_vector*);

        void calculate_QEqCharges();

        //// Electrostatics
        double E_elec; // electrostatic energy


        /// Matrix Approach (DEPRECATED and to be cleaned up)

        //CG storage
        double *p, *q, *r, *d;


        virtual int pack_forward_comm(int, int *, double *, int, int *);
        virtual void unpack_forward_comm(int, int, double *);
        virtual int pack_reverse_comm(int, int, double *);
        virtual void unpack_reverse_comm(int, int *, double *);
        virtual double memory_usage();
        virtual void grow_arrays(int);
        virtual void copy_arrays(int, int, int);
        virtual int pack_exchange(int, double *);
        virtual int unpack_exchange(int, double *);

        virtual double parallel_norm( double*, int );
        virtual double parallel_dot( double*, double*, int );
        virtual double parallel_vector_acc( double*, int );

        virtual void vector_sum(double*,double,double*,double,double*,int);
        virtual void vector_add(double*, double, double*,int);

        void   isPeriodic(); // periodicity check to decide on how to fill A matrix

        //TODO:old versions
        void QEqSerial();

    };

}

#endif
#endif
