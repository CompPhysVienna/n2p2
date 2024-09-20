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

#ifdef KSPACE_CLASS

KSpaceStyle(nnp,KSpaceNNP)

#else

#ifndef LMP_KSPACE_NNP_H
#define LMP_KSPACE_NNP_H

#include <gsl/gsl_multimin.h>
#include "kspace.h"

#if defined(FFT_FFTW3)
#define LMP_FFT_LIB "FFTW3"
#elif defined(FFT_MKL)
#define LMP_FFT_LIB "MKL FFT"
#elif defined(FFT_CUFFT)
#define LMP_FFT_LIB "cuFFT"
#else
#define LMP_FFT_LIB "KISS FFT"
#endif

#ifdef FFT_SINGLE
typedef float FFT_SCALAR;
#define LMP_FFT_PREC "single"
#define MPI_FFT_SCALAR MPI_FLOAT
#else

typedef double FFT_SCALAR;
#define LMP_FFT_PREC "double"
#define MPI_FFT_SCALAR MPI_DOUBLE
#endif

#include "InterfaceLammps.h"


namespace LAMMPS_NS {

    class KSpaceNNP : public KSpace {
        friend class PairNNP;
        friend class FixNNP;
    public:
        KSpaceNNP(class LAMMPS *);
        ~KSpaceNNP();

        virtual void settings(int, char **);
        virtual void init();
        virtual void setup();
        //virtual void setup_grid();
        virtual void compute(int, int);
        //virtual int timing_1d(int, double &);
        //virtual int timing_3d(int, double &);
        //virtual double memory_usage();

    protected:

        class PairNNP *nnp; // interface to NNP pair_style

        int ewaldflag,pppmflag;

        int triclinic;               // domain settings, orthog or triclinic

        double cutoff;
        double gsqmx;

        //// EWALD SUM
        int kewaldflag; // O if no kspace_modify

        double unitk[3];
        int *kxvecs,*kyvecs,*kzvecs;
        int kxmax_orig,kymax_orig,kzmax_orig,kmax_created;
        int kxmax,kymax,kzmax,kmax,kmax3d;
        int kcount;

        int ewald_truncation_method; // truncation method (RuNNer)

        double ewald_eta;
        double s_ewald;
        double ewald_recip_cutoff;
        double ewald_real_cutoff;

        double ewald_max_charge;
        double ewald_max_sigma;

        double *ug;
        double **eg,**vg;
        double **ek;

        double *kcoeff;
        double **sfexp_rl,**sfexp_im;
        double *sf_real, *sf_im;
        double *sfexp_rl_all,*sfexp_im_all; // structure factors after communications ?
        double ***cs,***sn; // cosine and sine grid

        void calculate_ewald_eta(int);
        void calculate_ewald_eta_efficient();

        void ewald_coeffs();
        void ewald_sfexp();

        void ewald_pbc(double); // TODO: this is from RuNNer

        double rms(int, double, bigint, double);

        double compute_ewald_eqeq(const gsl_vector*);

        //// PPPM

        int me,nprocs;
        int nfactors;
        int *factors;

        double volume;
        double delxinv,delyinv,delzinv,delvolinv;
        double h_x,h_y,h_z;
        double shift,shiftone;
        int peratom_allocate_flag;
        int nxlo_in,nylo_in,nzlo_in,nxhi_in,nyhi_in,nzhi_in;
        int nxlo_out,nylo_out,nzlo_out,nxhi_out,nyhi_out,nzhi_out;
        int nxlo_ghost,nxhi_ghost,nylo_ghost,nyhi_ghost,nzlo_ghost,nzhi_ghost;
        int nxlo_fft,nylo_fft,nzlo_fft,nxhi_fft,nyhi_fft,nzhi_fft;
        int nlower,nupper;
        int ngrid,nfft,nfft_both;

        FFT_SCALAR ***density_brick;
        FFT_SCALAR ***vdx_brick,***vdy_brick,***vdz_brick;
        FFT_SCALAR ***vx_brick,***vy_brick,***vz_brick;
        FFT_SCALAR ***u_brick;
        FFT_SCALAR ***v0_brick,***v1_brick,***v2_brick;
        FFT_SCALAR ***v3_brick,***v4_brick,***v5_brick;
        double *greensfn;
        double *fkx,*fky,*fkz;
        FFT_SCALAR *density_fft;
        FFT_SCALAR *work1,*work2;

        double *gf_b;
        FFT_SCALAR **rho1d,**rho_coeff,**drho1d,**drho_coeff;
        double *sf_precoeff1, *sf_precoeff2, *sf_precoeff3;
        double *sf_precoeff4, *sf_precoeff5, *sf_precoeff6;
        double sf_coeff[6];          // coefficients for calculating ad self-forces
        double **acons;

        // FFTs and grid communication

        class FFT3d *fft1,*fft2;
        class Remap *remap;
        class GridComm *gc;

        FFT_SCALAR *gc_buf1,*gc_buf2;
        int ngc_buf1,ngc_buf2,npergrid;

        int **part2grid;             // storage for particle -> grid mapping
        int nmax;

        double *boxlo;

        virtual void set_grid_global();
        void set_grid_local();
        void adjust_gewald();
        virtual double newton_raphson_f();
        double derivf();
        double final_accuracy();

        virtual void allocate();
        //virtual void allocate_peratom();
        virtual void deallocate();
        //virtual void deallocate_peratom();


        int factorable(int);
        double compute_df_kspace();
        double estimate_ik_error(double, double, bigint);
        //virtual double compute_qopt();
        virtual void compute_gf_denom();
        virtual void compute_gf_ik();


        void make_rho_qeq(const gsl_vector*); // charge density (rho) / charge
        virtual void particle_map();
        double compute_pppm_eqeq();
        double compute_pppm_dEdQ(int);

        //virtual void make_rho();

        virtual void brick2fft();

        virtual void poisson(); // Poisson solver for P3M (differentiation_flag == 0)

        /*void compute_sf_precoeff();

        virtual void fieldforce();
        virtual void fieldforce_ik();
        virtual void fieldforce_ad();

        virtual void poisson_peratom();
        virtual void fieldforce_peratom();*/
        void procs2grid2d(int,int,int,int *, int*);
        void compute_rho1d(const FFT_SCALAR &, const FFT_SCALAR &, const FFT_SCALAR &);
        //void compute_drho1d(const FFT_SCALAR &, const FFT_SCALAR &,const FFT_SCALAR &);
        void compute_rho_coeff();


        // grid communication
        /*
        virtual void pack_forward_grid(int, void *, int, int *);
        virtual void unpack_forward_grid(int, void *, int, int *);
        virtual void pack_reverse_grid(int, void *, int, int *);
        virtual void unpack_reverse_grid(int, void *, int, int *);

        // triclinic

        void setup_triclinic();
        void compute_gf_ik_triclinic();
        void poisson_ik_triclinic();
        void poisson_groups_triclinic();

        // group-group interactions

        virtual void allocate_groups();
        virtual void deallocate_groups();
        virtual void make_rho_groups(int, int, int);
        virtual void poisson_groups(int);
        virtual void slabcorr_groups(int,int,int);*/

/* ----------------------------------------------------------------------
   denominator for Hockney-Eastwood Green's function
     of x,y,z = sin(kx*deltax/2), etc

            inf                 n-1
   S(n,k) = Sum  W(k+pi*j)**2 = Sum b(l)*(z*z)**l
           j=-inf               l=0

          = -(z*z)**n /(2n-1)! * (d/dx)**(2n-1) cot(x)  at z = sin(x)
   gf_b = denominator expansion coeffs
------------------------------------------------------------------------- */

        inline double gf_denom(const double &x, const double &y,
                               const double &z) const {
            double sx,sy,sz;
            sz = sy = sx = 0.0;
            for (int l = order-1; l >= 0; l--) {
                sx = gf_b[l] + sx*x;
                sy = gf_b[l] + sy*y;
                sz = gf_b[l] + sz*z;
            }
            double s = sx*sy*sz;
            return s*s;
        };


    };

}

#endif
#endif
