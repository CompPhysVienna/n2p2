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
    public:
        FixNNP(class LAMMPS *, int, char **);
        ~FixNNP();

        int setmask();
        virtual void post_constructor();
        virtual void init();
        void init_list(int,class NeighList *);
        virtual void init_storage();
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
        bool periodic;

        // QEq arrays
        double *chi,*hardness,*sigma;

        // Forces
        double *fLambda,*dchidxyz,*dEdQ;

        bigint ngroup;
        int nprev,dum1,dum2;

        // charges
        double *Q;
        double **Q_hist; // do we need this ???

        void allocate_qeq();
        void deallocate_qeq();
        void setup_qeq();
        void pre_force_qeq();
        void run_network();

        //TODO:to be removed
        char *pertype_option;
        virtual void pertype_parameters(char*);
        virtual void allocate_storage();
        virtual void deallocate_storage();
        void reallocate_storage();
        virtual void allocate_matrix();
        void deallocate_matrix();
        void reallocate_matrix();


        //// Function Minimization Approach

        double *xall,*yall,*zall; // all atoms (parallelization)

        gsl_multimin_function_fdf QEq_minimizer; // find a better name
        gsl_multimin_function_fdf fLambda_minimizer; // fLambda calculator

        double qref; // TODO: total ref charge of the system, normally it will be read from n2p2 ?

        const gsl_multimin_fdfminimizer_type *T;
        gsl_multimin_fdfminimizer *s;

        // Minimization Setup for Force Lambda
        double fLambda_f(const gsl_vector*);
        void fLambda_df(const gsl_vector*, gsl_vector*);
        void fLambda_fdf(const gsl_vector*, double*, gsl_vector*);

        static double fLambda_f_wrap(const gsl_vector*, void*);
        static void fLambda_df_wrap(const gsl_vector*, void*, gsl_vector*);
        static void fLambda_fdf_wrap(const gsl_vector*, void*, double*, gsl_vector*);

        void calculate_fLambda();

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

        double *rscr; // screening cutoff radii

        double screen_f(double);
        double screen_df(double);
        void screen_fdf();

        /// Matrix Approach (DEPRECATED and to be cleaned up)

        typedef struct{
            int n, m;
            int *firstnbr;
            int *numnbrs;
            int *jlist;
            double *val;
            double **val2d;
            double ***dvalq; // dAdxyzQ
        } sparse_matrix;

        sparse_matrix A;
        double *Adia_inv;
        double *b,*b_der; // b_der is the b vector during the derivative calculation, it has more entries than b
        double *b_prc, *b_prm;

        //CG storage
        double *p, *q, *r, *d;

        virtual void init_matvec();
        void init_A();
        virtual void compute_A();
        double calculate_A(int, int, double);

        void compute_dAdxyzQ();
        double calculate_dAdxyzQ(double, double, int, int);

        virtual int CG(double*,double*);

        virtual void sparse_matvec(sparse_matrix*,double*,double*);

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
