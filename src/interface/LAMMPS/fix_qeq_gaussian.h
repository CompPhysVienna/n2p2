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

FixStyle(qeq/gaussian,FixQEqGaussian)

#else

#ifndef LMP_FIX_QEQ_GAUSSIAN_H
#define LMP_FIX_QEQ_GAUSSIAN_H

#include "fix.h"

namespace LAMMPS_NS {

    class FixQEqGaussian : public Fix {
        friend class PairNNP;
    public:
        FixQEqGaussian(class LAMMPS *, int, char **);
        ~FixQEqGaussian();
        int setmask();
        virtual void post_constructor();
        virtual void init();
        void init_list(int,class NeighList *);
        virtual void init_storage();
        void setup_pre_force(int);
        virtual void pre_force(int);

        void setup_pre_force_respa(int, int);
        void pre_force_respa(int, int, int);

        void min_setup_pre_force(int);
        void min_pre_force(int);

        int test_dummy();

        int matvecs;
        double qeq_time;


    protected:
        int nevery,reaxflag;
        int nprev;
        int n, N, m_fill;
        int n_cap, nmax, m_cap;
        int pack_flag;
        int nlevels_respa;
        class NeighList *list;

        double tolerance;     // tolerance for the norm of the rel residual in CG

        double *chi,*hardness,*sigma;  // qeq parameters

        bigint ngroup;

        // fictitious charges
        double *Q;
        double **Q_hist;


        typedef struct{
            int n, m;
            int *firstnbr;
            int *numnbrs;
            int *jlist;
            double *val;
        } sparse_matrix;

        sparse_matrix A;
        double *Adia_inv;
        double *b;
        double *b_prc, *b_prm;

        //CG storage
        double *p, *q, *r, *d;

        // TODO: this is for testing/benchmarking
        // a serial version of QEq
        void QEq_serial(double *,double *, double *, double *);
        void init_storage_serial();
        void runQEq_serial();
        void init_matvec_serial();
        int  CG_serial(double*,double*);
        void calculate_Q_serial();
        double serial_norm( double*, int );
        double serial_dot( double*, double*, int );
        double serial_vector_acc( double*, int );

        // when these are virtual then we receive segmentation faults
        void allocate_storage();
        void deallocate_storage();
        void reallocate_storage();
        void allocate_matrix();
        void deallocate_matrix();
        void reallocate_matrix();

        char *pertype_option;  // argument to determine how per-type info is obtained


        virtual void init_matvec();
        void init_H();
        virtual void compute_A();
        double calculate_A(double,double,double);
        virtual void calculate_Q();

        virtual int CG(double*,double*);
        //int GMRES(double*,double*);
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

        // dual CG support
        int dual_enabled;  // 0: Original, separate s & t optimization; 1: dual optimization

    };

}

#endif
#endif
