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

FixStyle(hdnnp,FixHDNNP)

#else

#ifndef LMP_FIX_HDNNP_H
#define LMP_FIX_HDNNP_H

#include <gsl/gsl_vector.h>
#include "fix.h"
#include "InterfaceLammps.h"


namespace LAMMPS_NS {

    class FixHDNNP : public Fix {
        friend class PairHDNNP4G;
    public:
        FixHDNNP(class LAMMPS *, int, char **);
        ~FixHDNNP();

        int setmask();
        virtual void post_constructor();
        virtual void init();
        void init_list(int,class NeighList *);
        void setup_pre_force(int);
        virtual void pre_force(int);

        void min_setup_pre_force(int);
        void min_pre_force(int);

    protected:

        class PairHDNNP4G *hdnnp; // interface to HDNNP pair_style
        class KSpaceHDNNP *kspacehdnnp; // interface to HDNNP kspace_style
        class NeighList *list;
        char *pertype_option;

        int hdnnpflag;
        int kspaceflag; // 0:Ewald Sum, 1:PPPM
        int ngroup;

        bool periodic; // true if periodic
        double qRef; // total reference charge of the system
        double *Q;

        virtual void pertype_parameters(char*);
        void isPeriodic(); // check for periodicity
        void calculate_electronegativities(); // calculates electronegatives via running first set of NNs
        void process_first_network(); // interfaces to n2p2 and runs first NN
        void allocate_QEq(); // allocates QEq arrays
        void deallocate_QEq(); // deallocates QEq arrays

        /// QEq energy minimization with fixed total charge, solved by
        /// projectedConjugateGradient() (pair_hdnnp_4g.h)

        double QEq_f(const gsl_vector*); // f : QEq energy as a function of atomic charges (also stores E_elec, E_recip_global)
        void QEq_df(const gsl_vector*, gsl_vector*); // df : Gradient of QEq energy with respect to atomic charges (projected)
        void QEq_reciprocal(const gsl_vector*); // reciprocal-space terms of the charges used by QEq_df

        void calculate_QEqCharges(); // main function where minimization happens

        void calculate_erfc_terms(); // loops over neighbors of local atoms and calculates erfc terms
                                     // these consume the most time. Storing them speeds up calculations

        /// Global storage
        int *type_all,*type_loc;
        double *qall,*qall_loc;
        double *dEdQ_all,*dEdQ_loc; // gradient of the charge equilibration energy wrt charges
        double *xx,*xy,*xz; // global positions (here only for nonperiodic case)
        double *xx_loc,*xy_loc,*xz_loc; // sparse local positions (here only for nonperiodic case)


        /* Matrix Approach (DEPRECATED and to be deleted)
        int nevery,hdnnpflag;
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
        virtual void vector_add(double*, double, double*,int);*/

    };

}

#endif
#endif
