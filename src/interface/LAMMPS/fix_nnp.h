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

        class PairNNP *nnp; // interface to NNP pair_style
        class NeighList *list;
        char *pertype_option;

        bool periodic; // true if periodic
        double qRef; // total reference charge of the system
        double *Q;

        virtual void pertype_parameters(char*);
        void   isPeriodic(); // true if periodic
        void calculate_electronegativities();
        void process_first_network(); // run first NN and calculate atomic electronegativities
        void allocate_QEq(); // allocate QEq arrays
        void deallocate_QEq(); // deallocate QEq arrays
        void map_localids();

        /// QEq energy minimization via gsl
        gsl_multimin_function_fdf QEq_minimizer; // find a better name
        const gsl_multimin_fdfminimizer_type *T;
        gsl_multimin_fdfminimizer *s;
        double QEq_f(const gsl_vector*); // f : QEq energy as a function of atomic charges
        void QEq_df(const gsl_vector*, gsl_vector*); // df : Gradient of QEq energy with respect to atomic charges
        void QEq_fdf(const gsl_vector*, double*, gsl_vector*); // fdf
        static double QEq_f_wrap(const gsl_vector*, void*); // wrapper function of f
        static void QEq_df_wrap(const gsl_vector*, void*, gsl_vector*); // wrapper function of df
        static void QEq_fdf_wrap(const gsl_vector*, void*, double*, gsl_vector*); // wrapper function of fdf
        void calculate_QEqCharges(); // QEq minimizer

        /// Global storage
        double *coords,*xf,*yf,*zf; // global arrays for atom positions
        double *xbuf; // memory for atom positions
        int ntotal;
        int xbufsize;

        void pack_positions(); // pack atom->x into xbuf
        void gather_positions();


        /// Matrix Approach (DEPRECATED and to be deleted)
        int nevery,nnpflag;
        int n, N, m_fill;
        int n_cap, m_cap;
        int pack_flag;
        bigint ngroup;
        int nprev;
        double **Q_hist;
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

    };

}

#endif
#endif
