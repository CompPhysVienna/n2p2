/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "kspace_nnp.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fft3d_wrap.h"
#include "force.h"
#include "gridcomm.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "neighbor.h"
#include "pair.h"
#include "remap_wrap.h"
#include "pair_nnp.h"
#include "fix_nnp.h"

#include <cmath>
#include <cstring>
#include <iostream>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;
using namespace std;

#define MAXORDER 7
#define OFFSET 16384
#define LARGE 10000.0
#define SMALL 0.00001
#define EPS_HOC 1.0e-7

enum{REVERSE_RHO};
enum{FORWARD_IK,FORWARD_AD,FORWARD_IK_PERATOM,FORWARD_AD_PERATOM};

#ifdef FFT_SINGLE
#define ZEROF 0.0f
#define ONEF  1.0f
#else
#define ZEROF 0.0
#define ONEF  1.0
#endif

/* ---------------------------------------------------------------------- */

KSpaceNNP::KSpaceNNP(LAMMPS *lmp) : KSpace(lmp),
      kxvecs(nullptr), kyvecs(nullptr), kzvecs(nullptr), ug(nullptr), eg(nullptr), vg(nullptr),
      ek(nullptr), sfexp_rl(nullptr), sfexp_im(nullptr), sf_real(nullptr), sf_im(nullptr),
      sfexp_rl_all(nullptr),sfexp_im_all(nullptr),
      cs(nullptr), sn(nullptr), factors(nullptr),
      density_brick(nullptr), vdx_brick(nullptr),
      vdy_brick(nullptr), vdz_brick(nullptr),u_brick(nullptr), v0_brick(nullptr), v1_brick(nullptr),
      v2_brick(nullptr), v3_brick(nullptr),v4_brick(nullptr), v5_brick(nullptr), greensfn(nullptr),
      fkx(nullptr), fky(nullptr), fkz(nullptr), density_fft(nullptr), work1(nullptr),
      work2(nullptr), gf_b(nullptr), rho1d(nullptr),
      rho_coeff(nullptr), drho1d(nullptr), drho_coeff(nullptr),
      sf_precoeff1(nullptr), sf_precoeff2(nullptr), sf_precoeff3(nullptr),
      sf_precoeff4(nullptr), sf_precoeff5(nullptr), sf_precoeff6(nullptr),
      acons(nullptr), fft1(nullptr), fft2(nullptr), remap(nullptr), gc(nullptr),
      gc_buf1(nullptr), gc_buf2(nullptr), part2grid(nullptr), boxlo(nullptr)
{
    nnp = nullptr;
    nnp = (PairNNP *) force->pair_match("^nnp",0);

    ewaldflag = pppmflag = 0; // TODO check

    eflag_global = 1; // calculate global energy

    //// EWALD
    kewaldflag = 0;
    kmax_created = 0;

    kmax = 0;
    kxvecs = kyvecs = kzvecs = nullptr;
    ug = nullptr;
    eg = vg = nullptr;
    ek = nullptr;

    sfexp_rl = sfexp_im = nullptr;
    sf_im = sf_real = nullptr;
    sfexp_rl_all = sfexp_im_all = nullptr;
    cs = sn = nullptr;

    kcount = 0;

    //// PPPM
    peratom_allocate_flag = 0;
    //group_allocate_flag = 0;

    group_group_enable = 1;
    triclinic = domain->triclinic;

    //TODO: add a flag-check for these initializations
    nfactors = 3;
    factors = new int[nfactors];
    factors[0] = 2;
    factors[1] = 3;
    factors[2] = 5;

    MPI_Comm_rank(world,&me);
    MPI_Comm_size(world,&nprocs);

    nfft_both = 0;
    nxhi_in = nxlo_in = nxhi_out = nxlo_out = 0;
    nyhi_in = nylo_in = nyhi_out = nylo_out = 0;
    nzhi_in = nzlo_in = nzhi_out = nzlo_out = 0;

    density_brick = vdx_brick = vdy_brick = vdz_brick = nullptr;
    vx_brick = vy_brick = vz_brick = nullptr;
    density_fft = nullptr;
    u_brick = nullptr;
    v0_brick = v1_brick = v2_brick = v3_brick = v4_brick = v5_brick = nullptr;
    greensfn = nullptr;
    work1 = work2 = nullptr;
    vg = nullptr;
    fkx = fky = fkz = nullptr;

    sf_precoeff1 = sf_precoeff2 = sf_precoeff3 =
    sf_precoeff4 = sf_precoeff5 = sf_precoeff6 = nullptr;

    gf_b = nullptr;
    rho1d = rho_coeff = drho1d = drho_coeff = nullptr;

    fft1 = fft2 = nullptr;
    remap = nullptr;
    gc = nullptr;
    gc_buf1 = gc_buf2 = nullptr;

    nmax = 0;
    part2grid = nullptr;

    // define acons coefficients for estimation of kspace errors
    // see JCP 109, pg 7698 for derivation of coefficients
    // higher order coefficients may be computed if needed

    memory->create(acons,8,7,"pppm:acons");
    acons[1][0] = 2.0 / 3.0;
    acons[2][0] = 1.0 / 50.0;
    acons[2][1] = 5.0 / 294.0;
    acons[3][0] = 1.0 / 588.0;
    acons[3][1] = 7.0 / 1440.0;
    acons[3][2] = 21.0 / 3872.0;
    acons[4][0] = 1.0 / 4320.0;
    acons[4][1] = 3.0 / 1936.0;
    acons[4][2] = 7601.0 / 2271360.0;
    acons[4][3] = 143.0 / 28800.0;
    acons[5][0] = 1.0 / 23232.0;
    acons[5][1] = 7601.0 / 13628160.0;
    acons[5][2] = 143.0 / 69120.0;
    acons[5][3] = 517231.0 / 106536960.0;
    acons[5][4] = 106640677.0 / 11737571328.0;
    acons[6][0] = 691.0 / 68140800.0;
    acons[6][1] = 13.0 / 57600.0;
    acons[6][2] = 47021.0 / 35512320.0;
    acons[6][3] = 9694607.0 / 2095994880.0;
    acons[6][4] = 733191589.0 / 59609088000.0;
    acons[6][5] = 326190917.0 / 11700633600.0;
    acons[7][0] = 1.0 / 345600.0;
    acons[7][1] = 3617.0 / 35512320.0;
    acons[7][2] = 745739.0 / 838397952.0;
    acons[7][3] = 56399353.0 / 12773376000.0;
    acons[7][4] = 25091609.0 / 1560084480.0;
    acons[7][5] = 1755948832039.0 / 36229939200000.0;
    acons[7][6] = 4887769399.0 / 37838389248.0;
}

/* ---------------------------------------------------------------------- */

void KSpaceNNP::settings(int narg, char **arg)
{
    if (narg < 2) error->all(FLERR,"Illegal kspace_style nnp command"); // we have two params, not one (style + FP)

    if (strcmp(arg[0],"pppm") == 0) pppmflag = 1;
    else if (strcmp(arg[0],"ewald") == 0) ewaldflag = 1;
    else error->all(FLERR,"Illegal kspace_style nnp command");

    accuracy_relative = fabs(utils::numeric(FLERR,arg[1],false,lmp));
}

/* ----------------------------------------------------------------------
   free all memory
------------------------------------------------------------------------- */

KSpaceNNP::~KSpaceNNP()
{
    if (pppmflag)
    {
        deallocate();
        if (copymode) return;
        delete [] factors;
        //if (peratom_allocate_flag) deallocate_peratom();
        //if (group_allocate_flag) deallocate_groups();
        memory->destroy(part2grid);
        memory->destroy(acons);
    }
    else if (ewaldflag)
    {
        deallocate();
        //if (group_allocate_flag) deallocate_groups();
        memory->destroy(ek);
	//memory->destroy(acons);
        memory->destroy3d_offset(cs,-kmax_created);
        memory->destroy3d_offset(sn,-kmax_created);
    }
}

/* ----------------------------------------------------------------------
   called once before run
------------------------------------------------------------------------- */

void KSpaceNNP::init()
{
    // Make initializations based on the selected k-space style
    if (pppmflag)
    {
        if (me == 0) utils::logmesg(lmp,"PPPM initialization ...\n");

        // TODO: add other error handlings in LAMMPS

        if (order < 2 || order > MAXORDER)
            error->all(FLERR,fmt::format("PPPM order cannot be < 2 or > {}",MAXORDER));

        // compute two charge force (?)
        two_charge();

        triclinic = domain->triclinic;

        // extract short-range Coulombic cutoff from pair style
        cutoff = nnp->maxCutoffRadius;

        // compute qsum & qsqsum and warn if not charge-neutral
        scale = 1.0;
        qqrd2e = force->qqrd2e;
        qsum_qsq();
        natoms_original = atom->natoms;

        if (accuracy_absolute >= 0.0) accuracy = accuracy_absolute;
        else accuracy = accuracy_relative * two_charge_force;

        // free all arrays previously allocated
        deallocate();

        //if (peratom_allocate_flag) deallocate_peratom();
        //if (group_allocate_flag) deallocate_groups();

        GridComm *gctmp = nullptr;
        int iteration = 0;

        while (order >= minorder) {
            if (iteration && me == 0)
                error->warning(FLERR, "Reducing PPPM order b/c stencil extends "
                                      "beyond nearest neighbor processor");

            //compute_gf_denom(); //TODO: stagger ?
            set_grid_global();
            set_grid_local();
            // overlap not allowed

            gctmp = new GridComm(lmp, world, nx_pppm, ny_pppm, nz_pppm,
                                 nxlo_in, nxhi_in, nylo_in, nyhi_in, nzlo_in, nzhi_in,
                                 nxlo_out, nxhi_out, nylo_out, nyhi_out, nzlo_out, nzhi_out);

            int tmp1, tmp2;
            gctmp->setup(tmp1, tmp2);
            if (gctmp->ghost_adjacent()) break;
            delete gctmp;

            order--;
            iteration++;
        }

        if (order < minorder) error->all(FLERR,"PPPM order < minimum allowed order");
        if (!overlap_allowed && !gctmp->ghost_adjacent())
            error->all(FLERR,"PPPM grid stencil extends "
                             "beyond nearest neighbor processor");
        if (gctmp) delete gctmp;

        // adjust g_ewald TODO(kspace_modify)
        if (!gewaldflag) adjust_gewald();

        // calculate the final accuracy
        double estimated_accuracy = final_accuracy();

        // print stats
        int ngrid_max,nfft_both_max;
        MPI_Allreduce(&ngrid,&ngrid_max,1,MPI_INT,MPI_MAX,world);
        MPI_Allreduce(&nfft_both,&nfft_both_max,1,MPI_INT,MPI_MAX,world);

        if (me == 0) {
            std::string mesg = fmt::format("  G vector (1/distance) = {:.8g}\n",g_ewald);
            mesg += fmt::format("  grid = {} {} {}\n",nx_pppm,ny_pppm,nz_pppm);
            mesg += fmt::format("  stencil order = {}\n",order);
            mesg += fmt::format("  estimated absolute RMS force accuracy = {:.8g}\n",
                                estimated_accuracy);
            mesg += fmt::format("  estimated relative force accuracy = {:.8g}\n",
                                estimated_accuracy/two_charge_force);
            mesg += "  using " LMP_FFT_PREC " precision " LMP_FFT_LIB "\n";
            mesg += fmt::format("  3d grid and FFT values/proc = {} {}\n",
                                ngrid_max,nfft_both_max);
            utils::logmesg(lmp,mesg);
        }

        // allocate K-space dependent memory
        // don't invoke allocate peratom() or group(), will be allocated when needed
        allocate();

        // pre-compute Green's function denomiator expansion
        // pre-compute 1d charge distribution coefficients
        // (differentiation_flag == 0)
        compute_gf_denom();
        compute_rho_coeff();
    }
    else if (ewaldflag)
    {
        if (comm->me == 0) utils::logmesg(lmp,"Ewald initialization ...\n");

        // error check
        //triclinic_check();
        if (domain->dimension == 2)
            error->all(FLERR,"Cannot use Ewald with 2d simulation");
        if (!atom->q_flag) error->all(FLERR,"Kspace style requires atom attribute q");
        if (slabflag == 0 && domain->nonperiodic > 0)
            error->all(FLERR,"Cannot use non-periodic boundaries with Ewald");

        //TODO: no slab yet

        // compute two charge force TODO:check
        two_charge();

        // extract short-range Coulombic cutoff from pair style
        triclinic = domain->triclinic;

        // extract short-range Coulombic cutoff from pair style
        cutoff = nnp->maxCutoffRadius;

        // compute qsum & qsqsum and warn if not charge-neutral TODO:check
        scale = 1.0;
        qqrd2e = force->qqrd2e;
        qsum_qsq();
        natoms_original = atom->natoms;

        // Via the method in RuNNer (umgekehrt)
        // LAMMPS uses a different methodology to calculate g_ewald
        accuracy = accuracy_relative;
        g_ewald = sqrt(-2.0 * log(accuracy)) / (cutoff*nnp->cflength);

        // setup Ewald coefficients
        setup();
    }
}

/* ----------------------------------------------------------------------
   adjust coeffs, called initially and whenever volume has changed
------------------------------------------------------------------------- */

void KSpaceNNP::setup()
{
    if (pppmflag)
    {
        // TODO: trcilinic & slab
        // perform some checks to avoid illegal boundaries with read_data

        if (slabflag == 0 && domain->nonperiodic > 0)
            error->all(FLERR,"Cannot use non-periodic boundaries with PPPM");
        if (slabflag) {
            if (domain->xperiodic != 1 || domain->yperiodic != 1 ||
                domain->boundary[2][0] != 1 || domain->boundary[2][1] != 1)
                error->all(FLERR,"Incorrect boundaries with slab PPPM");
        }

        int i,j,k,n;
        double *prd;

        // volume-dependent factors
        // adjust z dimension for 2d slab PPPM
        // z dimension for 3d PPPM is zprd since slab_volfactor = 1.0

        //triclinic = 0 TODO:check

        // WARNING: Immediately convert to NNP units!
        // volume-dependent factors
        prd = domain->prd;
        double xprd = prd[0] * nnp->cflength;
        double yprd = prd[1] * nnp->cflength;
        double zprd = prd[2] * nnp->cflength;
        volume = xprd * yprd * zprd;

        delxinv = nx_pppm/xprd;
        delyinv = ny_pppm/yprd;
        delzinv = nz_pppm/zprd;

        delvolinv = delxinv*delyinv*delzinv;

        double unitkx = (MY_2PI/xprd);
        double unitky = (MY_2PI/yprd);
        double unitkz = (MY_2PI/zprd);

        // fkx,fky,fkz for my FFT grid pts

        double per;

        for (i = nxlo_fft; i <= nxhi_fft; i++) {
            per = i - nx_pppm*(2*i/nx_pppm);
            fkx[i] = unitkx*per;
        }

        for (i = nylo_fft; i <= nyhi_fft; i++) {
            per = i - ny_pppm*(2*i/ny_pppm);
            fky[i] = unitky*per;
        }

        for (i = nzlo_fft; i <= nzhi_fft; i++) {
            per = i - nz_pppm*(2*i/nz_pppm);
            fkz[i] = unitkz*per;
        }

        // virial coefficients TODO

        /*double sqk,vterm;

        n = 0;
        for (k = nzlo_fft; k <= nzhi_fft; k++) {
            for (j = nylo_fft; j <= nyhi_fft; j++) {
                for (i = nxlo_fft; i <= nxhi_fft; i++) {
                    sqk = fkx[i]*fkx[i] + fky[j]*fky[j] + fkz[k]*fkz[k];
                    if (sqk == 0.0) {
                        vg[n][0] = 0.0;
                        vg[n][1] = 0.0;
                        vg[n][2] = 0.0;
                        vg[n][3] = 0.0;
                        vg[n][4] = 0.0;
                        vg[n][5] = 0.0;
                    } else {
                        vterm = -2.0 * (1.0/sqk + 0.25/(g_ewald*g_ewald));
                        vg[n][0] = 1.0 + vterm*fkx[i]*fkx[i];
                        vg[n][1] = 1.0 + vterm*fky[j]*fky[j];
                        vg[n][2] = 1.0 + vterm*fkz[k]*fkz[k];
                        vg[n][3] = vterm*fkx[i]*fky[j];
                        vg[n][4] = vterm*fkx[i]*fkz[k];
                        vg[n][5] = vterm*fky[j]*fkz[k];
                    }
                    n++;
                }
            }
        }*/

        compute_gf_ik(); // diff option 'ik' is selected
    }
    else if (ewaldflag)
    {
        // WARNING: Immediately convert to NNP units!
        // volume-dependent factors
        double const xprd = domain->xprd * nnp->cflength;
        double const yprd = domain->yprd * nnp->cflength;
        double const zprd = domain->zprd * nnp->cflength;
        volume = xprd * yprd * zprd;

        //TODO: slab feature

        unitk[0] = 2.0*MY_PI/xprd;
        unitk[1] = 2.0*MY_PI/yprd;
        unitk[2] = 2.0*MY_PI/zprd;

        // Get k-space resolution via the method in RuNNer
        ewald_recip_cutoff = sqrt(-2.0 * log(accuracy)) * g_ewald;
        //ewald_real_cutoff = sqrt(-2.0 * log(accuracy)) / g_ewald;
        ewald_real_cutoff = cutoff * nnp->cflength; // we skip the calculation

        //std::cout << "recip cut:" << ewald_recip_cutoff << '\n';
        //std::cout << "real cut: " << ewald_real_cutoff << '\n';
        //std::cout << "eta: " << 1 / g_ewald << '\n';

        int kmax_old = kmax;
        kxmax = kymax = kzmax = 1;

        ewald_pbc(ewald_recip_cutoff); // get kxmax, kymax and kzmax
        gsqmx = ewald_recip_cutoff * ewald_recip_cutoff;

        kmax = MAX(kxmax,kymax);
        kmax = MAX(kmax,kzmax);
        kmax3d = 4*kmax*kmax*kmax + 6*kmax*kmax + 3*kmax;

        // size change ?
        kxmax_orig = kxmax;
        kymax_orig = kymax;
        kzmax_orig = kzmax;

	//allocate();

        // if size has grown, reallocate k-dependent and nlocal-dependent arrays
        if (kmax > kmax_old) {

            deallocate();
            allocate();

            memory->destroy(ek);
            memory->destroy3d_offset(cs,-kmax_created);
            memory->destroy3d_offset(sn,-kmax_created);
            nmax = atom->nmax;
            memory->create(ek,nmax,3,"ewald:ek");
            memory->create3d_offset(cs,-kmax,kmax,3,nmax,"ewald:cs");
            memory->create3d_offset(sn,-kmax,kmax,3,nmax,"ewald:sn");
            kmax_created = kmax;
        }

        // pre-compute Ewald coefficients and structure factor arrays
        ewald_coeffs();
        //ewald_sfexp();
    }
}

// compute RMS accuracy for a dimension TODO: do we need this ?
double KSpaceNNP::rms(int km, double prd, bigint natoms, double q2)
{

    if (natoms == 0) natoms = 1;   // avoid division by zero
    double value = 2.0*q2*g_ewald/prd *
                   sqrt(1.0/(MY_PI*km*natoms)) *
                   exp(-MY_PI*MY_PI*km*km/(g_ewald*g_ewald*prd*prd));

    return value;
}

// Calculate Ewald coefficients
void KSpaceNNP::ewald_coeffs()
{
    int k,l,m;
    double sqk,vterm;
    double preu = 4.0*M_PI/volume;
    double etasq;

    etasq = 1.0 / (g_ewald*g_ewald); // LAMMPS truncation (RuNNer truncations has been removed)

    kcount = 0;

    // (k,0,0), (0,l,0), (0,0,m)

    for (m = 1; m <= kmax; m++) {
        sqk = (m*unitk[0]) * (m*unitk[0]);
        if (sqk <= gsqmx) {
            //fprintf(stderr, "sqk 1x = %24.16E, m %d\n", sqrt(sqk), m);
            kxvecs[kcount] = m;
            kyvecs[kcount] = 0;
            kzvecs[kcount] = 0;
            kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
            kcount++;
        }
        sqk = (m*unitk[1]) * (m*unitk[1]);
        if (sqk <= gsqmx) {
            //fprintf(stderr, "sqk 1y = %24.16E, m %d\n", sqrt(sqk), m);
            kxvecs[kcount] = 0;
            kyvecs[kcount] = m;
            kzvecs[kcount] = 0;
            kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
            kcount++;
        }
        sqk = (m*unitk[2]) * (m*unitk[2]);
        if (sqk <= gsqmx) {
            //fprintf(stderr, "sqk 1z = %24.16E, m %d\n", sqrt(sqk), m);
            kxvecs[kcount] = 0;
            kyvecs[kcount] = 0;
            kzvecs[kcount] = m;
            kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
            kcount++;
        }
    }

    // 1 = (k,l,0), 2 = (k,-l,0)

    for (k = 1; k <= kxmax; k++) {
        for (l = 1; l <= kymax; l++) {
            sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l);
            if (sqk <= gsqmx) {
                //fprintf(stderr, "sqk 2 = %24.16E, k %d  l %d\n", sqrt(sqk), k, l);
                kxvecs[kcount] = k;
                kyvecs[kcount] = l;
                kzvecs[kcount] = 0;
                kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                kcount++;

                //fprintf(stderr, "sqk 2 = %24.16E, k %d -l %d\n", sqrt(sqk), k, l);
                kxvecs[kcount] = k;
                kyvecs[kcount] = -l;
                kzvecs[kcount] = 0;
                kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                kcount++;;
            }
        }
    }

    // 1 = (0,l,m), 2 = (0,l,-m)

    for (l = 1; l <= kymax; l++) {
        for (m = 1; m <= kzmax; m++) {
            sqk = (unitk[1]*l) * (unitk[1]*l) + (unitk[2]*m) * (unitk[2]*m);
            if (sqk <= gsqmx) {
                //fprintf(stderr, "sqk 3 = %24.16E, l %d  m %d\n", sqrt(sqk), l, m);
                kxvecs[kcount] = 0;
                kyvecs[kcount] = l;
                kzvecs[kcount] = m;
                kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                kcount++;

                //fprintf(stderr, "sqk 3 = %24.16E, l %d -m %d\n", sqrt(sqk), l, m);
                kxvecs[kcount] = 0;
                kyvecs[kcount] = l;
                kzvecs[kcount] = -m;
                kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                kcount++;
            }
        }
    }

    // 1 = (k,0,m), 2 = (k,0,-m)

    for (k = 1; k <= kxmax; k++) {
        for (m = 1; m <= kzmax; m++) {
            sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[2]*m) * (unitk[2]*m);
            if (sqk <= gsqmx) {
                //fprintf(stderr, "sqk 4 = %24.16E, k %d  m %d\n", sqrt(sqk), k, m);
                kxvecs[kcount] = k;
                kyvecs[kcount] = 0;
                kzvecs[kcount] = m;
                kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                kcount++;

                //fprintf(stderr, "sqk 4 = %24.16E, k %d -m %d\n", sqrt(sqk), k, m);
                kxvecs[kcount] = k;
                kyvecs[kcount] = 0;
                kzvecs[kcount] = -m;
                kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                kcount++;
            }
        }
    }

    // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

    for (k = 1; k <= kxmax; k++) {
        for (l = 1; l <= kymax; l++) {
            for (m = 1; m <= kzmax; m++) {
                sqk = (unitk[0]*k) * (unitk[0]*k) + (unitk[1]*l) * (unitk[1]*l) +
                      (unitk[2]*m) * (unitk[2]*m);
                if (sqk <= gsqmx) {
                    //fprintf(stderr, "sqk 5 = %24.16E, k %d  l %d  m %d\n", sqrt(sqk), k, l, m);
                    kxvecs[kcount] = k;
                    kyvecs[kcount] = l;
                    kzvecs[kcount] = m;
                    kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                    kcount++;

                    //fprintf(stderr, "sqk 5 = %24.16E, k %d -l %d  m %d\n", sqrt(sqk), k, l, m);
                    kxvecs[kcount] = k;
                    kyvecs[kcount] = -l;
                    kzvecs[kcount] = m;
                    kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                    kcount++;

                    //fprintf(stderr, "sqk 5 = %24.16E, k %d  l %d -m %d\n", sqrt(sqk), k, l, m);
                    kxvecs[kcount] = k;
                    kyvecs[kcount] = l;
                    kzvecs[kcount] = -m;
                    kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                    kcount++;

                    //fprintf(stderr, "sqk 5 = %24.16E, k %d -l %d -m %d\n", sqrt(sqk), k, l, m);
                    kxvecs[kcount] = k;
                    kyvecs[kcount] = -l;
                    kzvecs[kcount] = -m;
                    kcoeff[kcount] = preu*exp(-0.5*sqk*etasq)/sqk;
                    kcount++;
                }
            }
        }
    }
}

// Calculate Ewald structure factors
void KSpaceNNP::ewald_sfexp()
{
    int i,k,l,m,n,ic;
    double sqk,clpm,slpm;

    double **x = atom->x;
    int nlocal = atom->nlocal;
    double cflength;

    n = 0;
    cflength = nnp->cflength; // RuNNer truncation (conversion required)

    // (k,0,0), (0,l,0), (0,0,m)

    for (ic = 0; ic < 3; ic++) {
        sqk = unitk[ic]*unitk[ic];
        if (sqk <= gsqmx) {
            for (i = 0; i < nlocal; i++) {
                cs[0][ic][i] = 1.0;
                sn[0][ic][i] = 0.0;
                cs[1][ic][i] = cos(unitk[ic]*x[i][ic]*cflength);
                sn[1][ic][i] = sin(unitk[ic]*x[i][ic]*cflength);
                cs[-1][ic][i] = cs[1][ic][i];
                sn[-1][ic][i] = -sn[1][ic][i];
                sfexp_rl[n][i] = cs[1][ic][i];
                sfexp_im[n][i] = sn[1][ic][i];
            }
            n++;
        }
    }

    for (m = 2; m <= kmax; m++) {
        for (ic = 0; ic < 3; ic++) {
            sqk = m*unitk[ic] * m*unitk[ic];
            if (sqk <= gsqmx) {
                for (i = 0; i < nlocal; i++) {
                    cs[m][ic][i] = cs[m-1][ic][i]*cs[1][ic][i] -
                                   sn[m-1][ic][i]*sn[1][ic][i];
                    sn[m][ic][i] = sn[m-1][ic][i]*cs[1][ic][i] +
                                   cs[m-1][ic][i]*sn[1][ic][i];
                    cs[-m][ic][i] = cs[m][ic][i];
                    sn[-m][ic][i] = -sn[m][ic][i];
                    sfexp_rl[n][i] = cs[m][ic][i];
                    sfexp_im[n][i] = sn[m][ic][i];
                }
                n++;
            }
        }
    }

    // 1 = (k,l,0), 2 = (k,-l,0)

    for (k = 1; k <= kxmax; k++) {
        for (l = 1; l <= kymax; l++) {
            sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]);
            if (sqk <= gsqmx) {
                for (i = 0; i < nlocal; i++) {
                    sfexp_rl[n][i] = (cs[k][0][i]*cs[l][1][i] - sn[k][0][i]*sn[l][1][i]);
                    sfexp_im[n][i] = (sn[k][0][i]*cs[l][1][i] + cs[k][0][i]*sn[l][1][i]);
                }
                n++;
                for (i = 0; i < nlocal; i++) {
                    sfexp_rl[n][i] = (cs[k][0][i]*cs[l][1][i] + sn[k][0][i]*sn[l][1][i]);
                    sfexp_im[n][i] = (sn[k][0][i]*cs[l][1][i] - cs[k][0][i]*sn[l][1][i]);
                }
                n++;
            }
        }
    }

    // 1 = (0,l,m), 2 = (0,l,-m)

    for (l = 1; l <= kymax; l++) {
        for (m = 1; m <= kzmax; m++) {
            sqk = (l*unitk[1] * l*unitk[1]) + (m*unitk[2] * m*unitk[2]);
            if (sqk <= gsqmx) {
                for (i = 0; i < nlocal; i++) {
                    sfexp_rl[n][i] = (cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i]);
                    sfexp_im[n][i] = (sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i]);
                }
                n++;
                for (i = 0; i < nlocal; i++) {
                    sfexp_rl[n][i] = (cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i]);
                    sfexp_im[n][i] = (sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i]);
                }
                n++;
            }
        }
    }

    // 1 = (k,0,m), 2 = (k,0,-m)

    for (k = 1; k <= kxmax; k++) {
        for (m = 1; m <= kzmax; m++) {
            sqk = (k*unitk[0] * k*unitk[0]) + (m*unitk[2] * m*unitk[2]);
            if (sqk <= gsqmx) {
                for (i = 0; i < nlocal; i++) {
                    sfexp_rl[n][i] = (cs[k][0][i]*cs[m][2][i] - sn[k][0][i]*sn[m][2][i]);
                    sfexp_im[n][i] = (sn[k][0][i]*cs[m][2][i] + cs[k][0][i]*sn[m][2][i]);
                }
                n++;
                for (i = 0; i < nlocal; i++) {
                    sfexp_rl[n][i] = (cs[k][0][i]*cs[m][2][i] + sn[k][0][i]*sn[m][2][i]);
                    sfexp_im[n][i] = (sn[k][0][i]*cs[m][2][i] - cs[k][0][i]*sn[m][2][i]);
                }
                n++;
            }
        }
    }

    // 1 = (k,l,m), 2 = (k,-l,m), 3 = (k,l,-m), 4 = (k,-l,-m)

    for (k = 1; k <= kxmax; k++) {
        for (l = 1; l <= kymax; l++) {
            for (m = 1; m <= kzmax; m++) {
                sqk = (k*unitk[0] * k*unitk[0]) + (l*unitk[1] * l*unitk[1]) +
                      (m*unitk[2] * m*unitk[2]);
                if (sqk <= gsqmx) {
                    for (i = 0; i < nlocal; i++) {
                        clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
                        slpm = sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
                        sfexp_rl[n][i] = (cs[k][0][i]*clpm - sn[k][0][i]*slpm);
                        sfexp_im[n][i] = (sn[k][0][i]*clpm + cs[k][0][i]*slpm);
                    }
                    n++;
                    for (i = 0; i < nlocal; i++) {
                        clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
                        slpm = -sn[l][1][i]*cs[m][2][i] + cs[l][1][i]*sn[m][2][i];
                        sfexp_rl[n][i] = (cs[k][0][i]*clpm - sn[k][0][i]*slpm);
                        sfexp_im[n][i] = (sn[k][0][i]*clpm + cs[k][0][i]*slpm);
                    }
                    n++;
                    for (i = 0; i < nlocal; i++) {
                        clpm = cs[l][1][i]*cs[m][2][i] + sn[l][1][i]*sn[m][2][i];
                        slpm = sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
                        sfexp_rl[n][i] = (cs[k][0][i]*clpm - sn[k][0][i]*slpm);
                        sfexp_im[n][i] = (sn[k][0][i]*clpm + cs[k][0][i]*slpm);
                    }
                    n++;
                    for (i = 0; i < nlocal; i++) {
                        clpm = cs[l][1][i]*cs[m][2][i] - sn[l][1][i]*sn[m][2][i];
                        slpm = -sn[l][1][i]*cs[m][2][i] - cs[l][1][i]*sn[m][2][i];
                        sfexp_rl[n][i] = (cs[k][0][i]*clpm - sn[k][0][i]*slpm);
                        sfexp_im[n][i] = (sn[k][0][i]*clpm + cs[k][0][i]*slpm);
                    }
                    n++;
                }
            }
        }
    }

}

// Compute E_QEQ (recip) in PPPM TODO:WIP
double KSpaceNNP::compute_pppm_eqeq()
{

    // all procs communicate density values from their ghost cells
    //   to fully sum contribution in their 3d bricks
    // remap from 3d decomposition to FFT decomposition

    gc->reverse_comm_kspace(this,1,sizeof(FFT_SCALAR),REVERSE_RHO,
                            gc_buf1,gc_buf2,MPI_FFT_SCALAR);
    brick2fft();

    // compute potential V(r) on my FFT grid and
    //   portion of e_long on this proc's FFT grid
    // return potentials 3d brick decomposition

    std::cout << energy << '\n';
    poisson();

    std::cout << energy << '\n';
    exit(0);

    // all procs communicate E-field values
    // to fill ghost cells surrounding their 3d bricks
    // TODO check
    // differentiation_flag == 0

    gc->forward_comm_kspace(this,3,sizeof(FFT_SCALAR),FORWARD_IK,gc_buf1,gc_buf2,MPI_FFT_SCALAR);

    // sum global energy across procs and add in volume-dependent term

    const double qscale = qqrd2e * scale;

    if (eflag_global) {
        double energy_all;
        MPI_Allreduce(&energy, &energy_all, 1, MPI_DOUBLE, MPI_SUM, world);
        energy = energy_all;

        energy *= 0.5 * volume;
        energy -= g_ewald * qsqsum / MY_PIS +
                  MY_PI2 * qsum * qsum / (g_ewald * g_ewald * volume);
        energy *= qscale;
    }

    std::cout << "here" << '\n';
    //std::cout << qscale << '\n';
    std::cout << MY_PIS << '\n';
    std::cout << MY_PI2 << '\n';
    //std::cout << qsum << '\n';
    //std::cout << g_ewald << '\n';
    //std::cout << energy << '\n';
    exit(0);

    return energy;
}

// Returns dE_qeq/dQ_i in PPPM for a given local atom i (inx)
double KSpaceNNP::compute_pppm_dEdQ(int inx)
{
    int i,l,m,n,nx,ny,nz,mx,my,mz;
    FFT_SCALAR dx,dy,dz,x0,y0,z0;
    FFT_SCALAR dEdQx,dEdQy,dEdQz;

    // loop over my charges, interpolate dEdQ from nearby grid points
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // (dx,dy,dz) = distance to "lower left" grid pt
    // (mx,my,mz) = global coords of moving stencil pt
    // ek = 3 components of E-field on particle

    double *q = atom->q;
    double **x = atom->x;
    double **f = atom->f;

    double dEdQ;

    int nlocal = atom->nlocal;

    nx = part2grid[inx][0];
    ny = part2grid[inx][1];
    nz = part2grid[inx][2];
    dx = nx + shiftone - (x[inx][0] - boxlo[0]) * delxinv;
    dy = ny + shiftone - (x[inx][1] - boxlo[1]) * delyinv;
    dz = nz + shiftone - (x[inx][2] - boxlo[2]) * delzinv;

    compute_rho1d(dx, dy, dz);

    dEdQx = dEdQy = dEdQz = ZEROF;
    for (n = nlower; n <= nupper; n++) {
        mz = n + nz;
        z0 = rho1d[2][n];
        for (m = nlower; m <= nupper; m++) {
            my = m + ny;
            y0 = z0 * rho1d[1][m];
            for (l = nlower; l <= nupper; l++) {
                mx = l + nx;
                x0 = y0 * rho1d[0][l];
                dEdQx -= x0 * vx_brick[mz][my][mx];
                dEdQy -= x0 * vy_brick[mz][my][mx];
                dEdQz -= x0 * vz_brick[mz][my][mx];
            }
        }
    }

    dEdQ = sqrt(dEdQx*dEdQx + dEdQy*dEdQy + dEdQz*dEdQz);

    std::cout << dEdQ << '\n';


    /*for (i = 0; i < nlocal; i++) {
        nx = part2grid[i][0];
        ny = part2grid[i][1];
        nz = part2grid[i][2];
        dx = nx + shiftone - (x[i][0] - boxlo[0]) * delxinv;
        dy = ny + shiftone - (x[i][1] - boxlo[1]) * delyinv;
        dz = nz + shiftone - (x[i][2] - boxlo[2]) * delzinv;

        compute_rho1d(dx, dy, dz);

        dEdQx = dEdQy = dEdQz = ZEROF;
        for (n = nlower; n <= nupper; n++) {
            mz = n + nz;
            z0 = rho1d[2][n];
            for (m = nlower; m <= nupper; m++) {
                my = m + ny;
                y0 = z0 * rho1d[1][m];
                for (l = nlower; l <= nupper; l++) {
                    mx = l + nx;
                    x0 = y0 * rho1d[0][l];
                    dEdQx -= x0 * vx_brick[mz][my][mx];
                    dEdQy -= x0 * vy_brick[mz][my][mx];
                    dEdQz -= x0 * vz_brick[mz][my][mx];
                }
            }
        }
    }*/

    return dEdQ;
}

// Compute E_QEQ (recip) in Ewald
double KSpaceNNP::compute_ewald_eqeq(const gsl_vector *v)
{
    int i;
    int nlocal = atom->nlocal;
    int *tag = atom->tag;
    double E_recip;


    E_recip = 0.0;
    for (int k = 0; k < kcount; k++) // over k-space
    {
        //fprintf(stderr, "kcoeff[%d] = %24.16E\n", k, nnp->kcoeff[k]);
        double sf_real_loc = 0.0;
        double sf_im_loc = 0.0;
        sf_real[k] = 0.0;
        sf_im[k] = 0.0;
        // TODO: this loop over all atoms can be replaced by a MPIallreduce ?
        for (i = 0; i < nlocal; i++)
        {
            double const qi = gsl_vector_get(v,tag[i]-1);
            sf_real_loc += qi * sfexp_rl[k][i];
            sf_im_loc += qi * sfexp_im[k][i];
        }
        //fprintf(stderr, "sfexp %d : %24.16E\n", k, nnp->kcoeff[k] * (pow(sf_real,2) + pow(sf_im,2)));
        MPI_Allreduce(&(sf_real_loc),&(sf_real[k]),1,MPI_DOUBLE,MPI_SUM,world);
        MPI_Allreduce(&(sf_im_loc),&(sf_im[k]),1,MPI_DOUBLE,MPI_SUM,world);
        E_recip += kcoeff[k] * (pow(sf_real[k],2) + pow(sf_im[k],2));
    }

    return E_recip;
}

// TODO: this is called after the force->pair->compute calculations in verlet.cpp
// therefore we cannot make use of it as it is
void KSpaceNNP::compute(int eflag, int vflag)
{
    // Carry out k-space computations based on the selected method
    if (pppmflag)
    {

    }
    else if (ewaldflag)
    {

    }
}

void KSpaceNNP::allocate()
{
    // Make allocations based on the selected K-space method
    if (pppmflag)
    {
        memory->create3d_offset(density_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                                nxlo_out,nxhi_out,"pppm:density_brick");

        memory->create(density_fft,nfft_both,"pppm:density_fft");
        memory->create(greensfn,nfft_both,"pppm:greensfn");
        memory->create(work1,2*nfft_both,"pppm:work1");
        memory->create(work2,2*nfft_both,"pppm:work2");
        //memory->create(vg,nfft_both,6,"pppm:vg"); // TODO virial

        // triclinic = 0 TODO: triclinic systems ?
        memory->create1d_offset(fkx,nxlo_fft,nxhi_fft,"pppm:fkx");
        memory->create1d_offset(fky,nylo_fft,nyhi_fft,"pppm:fky");
        memory->create1d_offset(fkz,nzlo_fft,nzhi_fft,"pppm:fkz");

        // differentiation_flag = 0
        /*memory->create3d_offset(vdx_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                                nxlo_out,nxhi_out,"pppm:vdx_brick");
        memory->create3d_offset(vdy_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                                nxlo_out,nxhi_out,"pppm:vdy_brick");
        memory->create3d_offset(vdz_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                                nxlo_out,nxhi_out,"pppm:vdz_brick");*/

        memory->create3d_offset(vx_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                                nxlo_out,nxhi_out,"pppm:vx_brick");
        memory->create3d_offset(vy_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                                nxlo_out,nxhi_out,"pppm:vy_brick");
        memory->create3d_offset(vz_brick,nzlo_out,nzhi_out,nylo_out,nyhi_out,
                                nxlo_out,nxhi_out,"pppm:vz_brick");

        // summation coeffs
        order_allocated = order;
        memory->create(gf_b,order,"pppm:gf_b");
        memory->create2d_offset(rho1d,3,-order/2,order/2,"pppm:rho1d");
        memory->create2d_offset(drho1d,3,-order/2,order/2,"pppm:drho1d");
        memory->create2d_offset(rho_coeff,order,(1-order)/2,order/2,"pppm:rho_coeff");
        memory->create2d_offset(drho_coeff,order,(1-order)/2,order/2,
                                "pppm:drho_coeff");

        // create 2 FFTs and a Remap
        // 1st FFT keeps data in FFT decomposition
        // 2nd FFT returns data in 3d brick decomposition
        // remap takes data from 3d brick to FFT decomposition

        int tmp;

        fft1 = new FFT3d(lmp,world,nx_pppm,ny_pppm,nz_pppm,
                         nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                         nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                         0,0,&tmp,collective_flag);

        fft2 = new FFT3d(lmp,world,nx_pppm,ny_pppm,nz_pppm,
                         nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                         nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                         0,0,&tmp,collective_flag);

        remap = new Remap(lmp,world,
                          nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                          nxlo_fft,nxhi_fft,nylo_fft,nyhi_fft,nzlo_fft,nzhi_fft,
                          1,0,0,FFT_PRECISION,collective_flag);

        // create ghost grid object for rho and electric field communication
        // also create 2 bufs for ghost grid cell comm, passed to GridComm methods

        gc = new GridComm(lmp,world,nx_pppm,ny_pppm,nz_pppm,
                          nxlo_in,nxhi_in,nylo_in,nyhi_in,nzlo_in,nzhi_in,
                          nxlo_out,nxhi_out,nylo_out,nyhi_out,nzlo_out,nzhi_out);

        gc->setup(ngc_buf1,ngc_buf2);
        npergrid = 3;

        memory->create(gc_buf1,npergrid*ngc_buf1,"pppm:gc_buf1");
        memory->create(gc_buf2,npergrid*ngc_buf2,"pppm:gc_buf2");
    }
    else if (ewaldflag)
    {
       
	//kxvecs = new int[kmax3d];
  	//kyvecs = new int[kmax3d];
  	//kzvecs = new int[kmax3d];
 	//kcoeff = new int[kmax3d];

	//sf_real = new double[kmax3d];
  	//sf_im = new double[kmax3d];
	
	//for(int i = 0; i < kmax3d; ++i){
    	//	sfexp_rl[i] = new int[nloc];
    	//	sfexp_im[i] = new int[nloc];
	//}
	
	memory->create(kxvecs,kmax3d,"ewald:kxvecs");
	memory->create(kyvecs,kmax3d,"ewald:kyvecs");
        memory->create(kzvecs,kmax3d,"ewald:kzvecs");
        memory->create(kcoeff,kmax3d,"ewald:kcoeff");

	memory->create(sfexp_rl,kmax3d,atom->natoms,"ewald:sfexp_rl");
        memory->create(sfexp_im,kmax3d,atom->natoms,"ewald:sfexp_im");
        memory->create(sf_real,kmax3d,"ewald:sf_rl");
        memory->create(sf_im,kmax3d,"ewald:sf_im");
        
	//memory->create(eg,kmax3d,3,"ewald:eg");
        //memory->create(vg,kmax3d,6,"ewald:vg"); // TODO: might be required for pressure

    }
}

void KSpaceNNP::deallocate()
{
    // Make deallocations based on the selected K-space method
    if (pppmflag)
    {
        memory->destroy3d_offset(density_brick,nzlo_out,nylo_out,nxlo_out);

        // differentiation_flag = 0
        /*memory->destroy3d_offset(vdx_brick,nzlo_out,nylo_out,nxlo_out);
        memory->destroy3d_offset(vdy_brick,nzlo_out,nylo_out,nxlo_out);
        memory->destroy3d_offset(vdz_brick,nzlo_out,nylo_out,nxlo_out);*/

        memory->destroy3d_offset(vx_brick,nzlo_out,nylo_out,nxlo_out);
        memory->destroy3d_offset(vy_brick,nzlo_out,nylo_out,nxlo_out);
        memory->destroy3d_offset(vz_brick,nzlo_out,nylo_out,nxlo_out);

        memory->destroy(density_fft);
        memory->destroy(greensfn);
        memory->destroy(work1);
        memory->destroy(work2);
        memory->destroy(vg);

        memory->destroy1d_offset(fkx,nxlo_fft);
        memory->destroy1d_offset(fky,nylo_fft);
        memory->destroy1d_offset(fkz,nzlo_fft);

        memory->destroy(gf_b);

        memory->destroy2d_offset(rho1d,-order_allocated/2);
        memory->destroy2d_offset(drho1d,-order_allocated/2);
        memory->destroy2d_offset(rho_coeff,(1-order_allocated)/2);
        memory->destroy2d_offset(drho_coeff,(1-order_allocated)/2);

        delete fft1;
        delete fft2;
        delete remap;
        delete gc;
        memory->destroy(gc_buf1);
        memory->destroy(gc_buf2);
    }
    else if (ewaldflag)
    {
	int nloc = atom->nlocal;
        
	//delete [] kxvecs;
  	//delete [] kyvecs;
  	//delete [] kzvecs;
	//delete [] kcoeff;
	//delete [] sf_real;
	//delete [] sf_im;

	//for(int i = 0; i < kmax3d; ++i) {
    	//	delete [] sfexp_rl[i];
	//	delete [] sfexp_im[i];
	//}
	//delete [] sfexp_rl;
	//delete [] sfexp_im;
        
	memory->destroy(kxvecs);
	memory->destroy(kyvecs);
	memory->destroy(kzvecs);
	memory->destroy(kcoeff);
	
	memory->destroy(sf_real);
        memory->destroy(sf_im);	
	memory->destroy(sfexp_rl);
        memory->destroy(sfexp_im);
	
	//memory->destroy(ek);
        //memory->destroy3d_offset(cs,-kmax_created);
        //memory->destroy3d_offset(sn,-kmax_created);

        //memory->destroy(eg);
        //memory->destroy(vg);
    }
}

// Maps atoms to corresponding grid points
void KSpaceNNP::particle_map()
{
    int nx,ny,nz;

    double **x = atom->x;
    int nlocal = atom->nlocal;

    int flag = 0;

    // if atom count has changed, update qsum and qsqsum

    if (atom->natoms != natoms_original) {
        qsum_qsq();
        natoms_original = atom->natoms;
    }

    // return if there are no charges

    if (qsqsum == 0.0) return;

    boxlo = domain->boxlo; //triclinic = 0

    if (!std::isfinite(boxlo[0]) || !std::isfinite(boxlo[1]) || !std::isfinite(boxlo[2]))
        error->one(FLERR,"Non-numeric box dimensions - simulation unstable");

    if (atom->nmax > nmax) {
        memory->destroy(part2grid);
        nmax = atom->nmax;
        memory->create(part2grid,nmax,3,"kspacennp:part2grid");
    }

    for (int i = 0; i < nlocal; i++) {

        // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
        // current particle coord can be outside global and local box
        // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1

        nx = static_cast<int> ((x[i][0]-boxlo[0])*delxinv+shift) - OFFSET;
        ny = static_cast<int> ((x[i][1]-boxlo[1])*delyinv+shift) - OFFSET;
        nz = static_cast<int> ((x[i][2]-boxlo[2])*delzinv+shift) - OFFSET;


        // check that entire stencil around nx,ny,nz will fit in my 3d brick

        if (nx+nlower < nxlo_out || nx+nupper > nxhi_out ||
            ny+nlower < nylo_out || ny+nupper > nyhi_out ||
            nz+nlower < nzlo_out || nz+nupper > nzhi_out)
            flag = 1;
    }
    if (flag) error->one(FLERR,"Out of range atoms - cannot compute PPPM");
}

void KSpaceNNP::make_rho_qeq(const gsl_vector *v)
{
    int l,m,n,nx,ny,nz,mx,my,mz;
    FFT_SCALAR dx,dy,dz,x0,y0,z0;

    int *tag = atom->tag;

    // clear 3d density array

    memset(&(density_brick[nzlo_out][nylo_out][nxlo_out]),0,
           ngrid*sizeof(FFT_SCALAR));

    // loop over my charges, add their contribution to nearby grid points
    // (nx,ny,nz) = global coords of grid pt to "lower left" of charge
    // (dx,dy,dz) = distance to "lower left" grid pt
    // (mx,my,mz) = global coords of moving stencil pt

    double **x = atom->x;
    int nlocal = atom->nlocal;

    for (int i = 0; i < nlocal; i++) {
        //TODO:conversion ?
        nx = part2grid[i][0];
        ny = part2grid[i][1];
        nz = part2grid[i][2];
        dx = (nx+shiftone - (x[i][0]-boxlo[0])*delxinv) * nnp->cflength;
        dy = (ny+shiftone - (x[i][1]-boxlo[1])*delyinv) * nnp->cflength;
        dz = (nz+shiftone - (x[i][2]-boxlo[2])*delzinv) * nnp->cflength;

        compute_rho1d(dx,dy,dz);

        double const qi = gsl_vector_get(v, tag[i]-1);

        //z0 = delvolinv * q[i];
        z0 = delvolinv * qi;
        for (n = nlower; n <= nupper; n++) {
            mz = n+nz;
            y0 = z0*rho1d[2][n];
            for (m = nlower; m <= nupper; m++) {
                my = m+ny;
                x0 = y0*rho1d[1][m];
                for (l = nlower; l <= nupper; l++) {
                    mx = l+nx;
                    density_brick[mz][my][mx] += x0*rho1d[0][l];
                }
            }
        }
    }
}

void KSpaceNNP::compute_rho1d(const FFT_SCALAR &dx, const FFT_SCALAR &dy, const FFT_SCALAR &dz)
{
    int k,l;
    FFT_SCALAR r1,r2,r3;

    for (k = (1-order)/2; k <= order/2; k++) {
        r1 = r2 = r3 = ZEROF;
        std::cout << r1 << '\n';
        std::cout << r2 << '\n';
        for (l = order-1; l >= 0; l--) {
            std::cout << l << '\n';
            r1 = rho_coeff[l][k] + r1*dx;
            std::cout << "aaau" << '\n';
            exit(0);
            r2 = rho_coeff[l][k] + r2*dy;
            r3 = rho_coeff[l][k] + r3*dz;
        }
        rho1d[0][k] = r1;
        rho1d[1][k] = r2;
        rho1d[2][k] = r3;
    }
}

// Poission solver in PPPM
void KSpaceNNP::poisson()
{
    int i,j,k,n;
    double eng;

    // transform charge density (r -> k)

    n = 0;
    for (i = 0; i < nfft; i++) {
        work1[n++] = density_fft[i];
        work1[n++] = ZEROF;
    }

    fft1->compute(work1,work1,1);

    // global energy and virial contribution

    double scaleinv = 1.0/(nx_pppm*ny_pppm*nz_pppm);
    double s2 = scaleinv*scaleinv;

    if (eflag_global || vflag_global) {
        if (vflag_global) {
            n = 0;
            for (i = 0; i < nfft; i++) {
                eng = s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
                for (j = 0; j < 6; j++) virial[j] += eng*vg[i][j];
                if (eflag_global) energy += eng;
                n += 2;
            }
        } else {
            n = 0;
            for (i = 0; i < nfft; i++) {
                //std::cout << "Green" << greensfn[i] << '\n';
                //std::cout << "S2" << s2 << '\n';
                //std::cout << "Par" << (work1[n]*work1[n] + work1[n+1]*work1[n+1]) << '\n';
                energy += s2 * greensfn[i] * (work1[n]*work1[n] + work1[n+1]*work1[n+1]);
                n += 2;
            }
        }
    }


    // scale by 1/total-grid-pts to get rho(k)
    // multiply by Green's function to get V(k)

    n = 0;
    for (i = 0; i < nfft; i++) {
        work1[n++] *= scaleinv * greensfn[i];
        work1[n++] *= scaleinv * greensfn[i];
    }

    // extra FFTs for per-atom energy/virial
    //if (evflag_atom) poisson_peratom(); // TODO: do we need ?

    // compute V(r) in each of 3 dims by transformimg V(k)
    // FFT leaves data in 3d brick decomposition
    // copy it into inner portion of vx,vy,vz arrays

    // x direction

    n = 0;
    for (k = nzlo_fft; k <= nzhi_fft; k++)
        for (j = nylo_fft; j <= nyhi_fft; j++)
            for (i = nxlo_fft; i <= nxhi_fft; i++) {
                work2[n] = work1[n+1];
                work2[n+1] = work1[n];
                n += 2;
            }

    fft2->compute(work2,work2,-1);

    n = 0;
    for (k = nzlo_in; k <= nzhi_in; k++)
        for (j = nylo_in; j <= nyhi_in; j++)
            for (i = nxlo_in; i <= nxhi_in; i++) {
                vx_brick[k][j][i] = work2[n];
                n += 2;
            }

    // y direction

    n = 0;
    for (k = nzlo_fft; k <= nzhi_fft; k++)
        for (j = nylo_fft; j <= nyhi_fft; j++)
            for (i = nxlo_fft; i <= nxhi_fft; i++) {
                work2[n] = work1[n+1];
                work2[n+1] = work1[n];
                n += 2;
            }

    fft2->compute(work2,work2,-1);

    n = 0;
    for (k = nzlo_in; k <= nzhi_in; k++)
        for (j = nylo_in; j <= nyhi_in; j++)
            for (i = nxlo_in; i <= nxhi_in; i++) {
                vy_brick[k][j][i] = work2[n];
                n += 2;
            }

    // z direction gradient

    n = 0;
    for (k = nzlo_fft; k <= nzhi_fft; k++)
        for (j = nylo_fft; j <= nyhi_fft; j++)
            for (i = nxlo_fft; i <= nxhi_fft; i++) {
                work2[n] = work1[n+1];
                work2[n+1] = work1[n];
                n += 2;
            }

    fft2->compute(work2,work2,-1);

    n = 0;
    for (k = nzlo_in; k <= nzhi_in; k++)
        for (j = nylo_in; j <= nyhi_in; j++)
            for (i = nxlo_in; i <= nxhi_in; i++) {
                vz_brick[k][j][i] = work2[n];
                n += 2;
            }
}

//remap density from 3d brick decomposition to FFT decomposition
void KSpaceNNP::brick2fft()
{
    int n,ix,iy,iz;

    // copy grabs inner portion of density from 3d brick
    // remap could be done as pre-stage of FFT,
    //   but this works optimally on only double values, not complex values

    n = 0;
    for (iz = nzlo_in; iz <= nzhi_in; iz++)
        for (iy = nylo_in; iy <= nyhi_in; iy++)
            for (ix = nxlo_in; ix <= nxhi_in; ix++)
                density_fft[n++] = density_brick[iz][iy][ix];

    remap->perform(density_fft,density_fft,work1);
}

void KSpaceNNP::set_grid_local()
{
    // global indices of PPPM grid range from 0 to N-1
    // nlo_in,nhi_in = lower/upper limits of the 3d sub-brick of
    //   global PPPM grid that I own without ghost cells
    // for slab PPPM, assign z grid as if it were not extended
    // both non-tiled and tiled proc layouts use 0-1 fractional sumdomain info

    if (comm->layout != Comm::LAYOUT_TILED) {
        nxlo_in = static_cast<int> (comm->xsplit[comm->myloc[0]] * nx_pppm);
        nxhi_in = static_cast<int> (comm->xsplit[comm->myloc[0]+1] * nx_pppm) - 1;

        nylo_in = static_cast<int> (comm->ysplit[comm->myloc[1]] * ny_pppm);
        nyhi_in = static_cast<int> (comm->ysplit[comm->myloc[1]+1] * ny_pppm) - 1;

        nzlo_in = static_cast<int> (comm->zsplit[comm->myloc[2]] * nz_pppm);
        nzhi_in = static_cast<int> (comm->zsplit[comm->myloc[2]+1] * nz_pppm) - 1;

    } else {
        nxlo_in = static_cast<int> (comm->mysplit[0][0] * nx_pppm);
        nxhi_in = static_cast<int> (comm->mysplit[0][1] * nx_pppm) - 1;

        nylo_in = static_cast<int> (comm->mysplit[1][0] * ny_pppm);
        nyhi_in = static_cast<int> (comm->mysplit[1][1] * ny_pppm) - 1;

        nzlo_in = static_cast<int> (comm->mysplit[2][0] * nz_pppm);
        nzhi_in = static_cast<int> (comm->mysplit[2][1] * nz_pppm) - 1;
    }


    // nlower,nupper = stencil size for mapping particles to PPPM grid
    //TODO: conversion ?
    nlower = -(order-1)/2;
    nupper = order/2;

    // shift values for particle <-> grid mapping
    // add/subtract OFFSET to avoid int(-0.75) = 0 when want it to be -1
    //TODO: conversion ?
    if (order % 2) shift = OFFSET + 0.5;
    else shift = OFFSET;
    if (order % 2) shiftone = 0.0;
    else shiftone = 0.5;

    // nlo_out,nhi_out = lower/upper limits of the 3d sub-brick of
    //   global PPPM grid that my particles can contribute charge to
    // effectively nlo_in,nhi_in + ghost cells
    // nlo,nhi = global coords of grid pt to "lower left" of smallest/largest
    //           position a particle in my box can be at
    // dist[3] = particle position bound = subbox + skin/2.0 + qdist
    //   qdist = offset due to TIP4P fictitious charge
    //   convert to triclinic if necessary
    // nlo_out,nhi_out = nlo,nhi + stencil size for particle mapping
    // for slab PPPM, assign z grid as if it were not extended

    double *prd,*sublo,*subhi;

    // triclinic  = 0, no slab
    prd = domain->prd;
    boxlo = domain->boxlo;
    sublo = domain->sublo;
    subhi = domain->subhi;

    // Unit conversions for n2p2
    boxlo[0] = boxlo[0] * nnp->cflength;
    boxlo[1] = boxlo[1] * nnp->cflength;
    boxlo[2] = boxlo[2] * nnp->cflength;

    sublo[0] = sublo[0] * nnp->cflength;
    sublo[1] = sublo[1] * nnp->cflength;
    sublo[2] = sublo[2] * nnp->cflength;

    subhi[0] = subhi[0] * nnp->cflength;
    subhi[1] = subhi[1] * nnp->cflength;
    subhi[2] = subhi[2] * nnp->cflength;

    double xprd = prd[0] * nnp->cflength;
    double yprd = prd[1] * nnp->cflength;
    double zprd = prd[2] * nnp->cflength;

    double dist[3] = {0.0,0.0,0.0};
    double cuthalf = 0.5*neighbor->skin * nnp->cflength;
    dist[0] = dist[1] = dist[2] = cuthalf;

    int nlo,nhi;
    nlo = nhi = 0;

    nlo = static_cast<int> ((sublo[0]-dist[0]-boxlo[0]) *
                            nx_pppm/xprd + shift) - OFFSET;
    nhi = static_cast<int> ((subhi[0]+dist[0]-boxlo[0]) *
                            nx_pppm/xprd + shift) - OFFSET;
    nxlo_out = nlo + nlower;
    nxhi_out = nhi + nupper;

    nlo = static_cast<int> ((sublo[1]-dist[1]-boxlo[1]) *
                            ny_pppm/yprd + shift) - OFFSET;
    nhi = static_cast<int> ((subhi[1]+dist[1]-boxlo[1]) *
                            ny_pppm/yprd + shift) - OFFSET;
    nylo_out = nlo + nlower;
    nyhi_out = nhi + nupper;

    nlo = static_cast<int> ((sublo[2]-dist[2]-boxlo[2]) *
                            nz_pppm/zprd + shift) - OFFSET;
    nhi = static_cast<int> ((subhi[2]+dist[2]-boxlo[2]) *
                            nz_pppm/zprd + shift) - OFFSET;
    nzlo_out = nlo + nlower;
    nzhi_out = nhi + nupper;

    // x-pencil decomposition of FFT mesh
    // global indices range from 0 to N-1
    // each proc owns entire x-dimension, clumps of columns in y,z dimensions
    // npey_fft,npez_fft = # of procs in y,z dims
    // if nprocs is small enough, proc can own 1 or more entire xy planes,
    //   else proc owns 2d sub-blocks of yz plane
    // me_y,me_z = which proc (0-npe_fft-1) I am in y,z dimensions
    // nlo_fft,nhi_fft = lower/upper limit of the section
    //   of the global FFT mesh that I own in x-pencil decomposition

    int npey_fft,npez_fft;
    if (nz_pppm >= nprocs) {
        npey_fft = 1;
        npez_fft = nprocs;
    } else procs2grid2d(nprocs,ny_pppm,nz_pppm,&npey_fft,&npez_fft);

    int me_y = me % npey_fft;
    int me_z = me / npey_fft;

    nxlo_fft = 0;
    nxhi_fft = nx_pppm - 1;
    nylo_fft = me_y*ny_pppm/npey_fft;
    nyhi_fft = (me_y+1)*ny_pppm/npey_fft - 1;
    nzlo_fft = me_z*nz_pppm/npez_fft;
    nzhi_fft = (me_z+1)*nz_pppm/npez_fft - 1;

    // ngrid = count of PPPM grid pts owned by this proc, including ghosts

    ngrid = (nxhi_out-nxlo_out+1) * (nyhi_out-nylo_out+1) *
            (nzhi_out-nzlo_out+1);

    // count of FFT grids pts owned by this proc, without ghosts
    // nfft = FFT points in x-pencil FFT decomposition on this proc
    // nfft_brick = FFT points in 3d brick-decomposition on this proc
    // nfft_both = greater of 2 values

    nfft = (nxhi_fft-nxlo_fft+1) * (nyhi_fft-nylo_fft+1) *
           (nzhi_fft-nzlo_fft+1);
    int nfft_brick = (nxhi_in-nxlo_in+1) * (nyhi_in-nylo_in+1) *
                     (nzhi_in-nzlo_in+1);
    nfft_both = MAX(nfft,nfft_brick);
}

/* ----------------------------------------------------------------------
   set global size of PPPM grid = nx,ny,nz_pppm
   used for charge accumulation, FFTs, and electric field interpolation
------------------------------------------------------------------------- */
void KSpaceNNP::set_grid_global()
{
    // use xprd,yprd,zprd (even if triclinic, and then scale later)
    // adjust z dimension for 2d slab PPPM
    // 3d PPPM just uses zprd since slab_volfactor = 1.0

    double xprd = domain->xprd * nnp->cflength;
    double yprd = domain->yprd * nnp->cflength;
    double zprd = domain->zprd * nnp->cflength;
    //double zprd_slab = zprd*slab_volfactor;

    // make initial g_ewald estimate
    // based on desired accuracy and real space cutoff
    // fluid-occupied volume used to estimate real-space error
    // zprd used rather than zprd_slab

    double h;
    bigint natoms = atom->natoms;

    // TODO: check this, we can also use 'kspace_modify' to pick gewald
    if (!gewaldflag) {
        if (accuracy <= 0.0)
            error->all(FLERR, "KSpace accuracy must be > 0");
        if (q2 == 0.0)
            error->all(FLERR, "Must use kspace_modify gewald for uncharged system");
        g_ewald = accuracy * sqrt(natoms * cutoff * xprd * yprd * zprd) / (2.0 * q2);
        if (g_ewald >= 1.0) g_ewald = (1.35 - 0.15 * log(accuracy)) / cutoff;
        else g_ewald = sqrt(-log(g_ewald)) / cutoff;
    }

    // set optimal nx_pppm,ny_pppm,nz_pppm based on order and accuracy
    // nz_pppm uses extended zprd_slab instead of zprd
    // reduce it until accuracy target is met

    if (!gridflag) { // gridflag = 0 if there is no kspace_modify command

        // differentiation_flag = 0 & stagger_flag = 0 TODO
        double err;
        h_x = h_y = h_z = 1.0/g_ewald;

        nx_pppm = static_cast<int> (xprd/h_x) + 1;
        ny_pppm = static_cast<int> (yprd/h_y) + 1;
        nz_pppm = static_cast<int> (zprd/h_z) + 1;

        err = estimate_ik_error(h_x,xprd,natoms);
        while (err > accuracy) {
            err = estimate_ik_error(h_x,xprd,natoms);
            nx_pppm++;
            h_x = xprd/nx_pppm;
        }

        err = estimate_ik_error(h_y,yprd,natoms);
        while (err > accuracy) {
            err = estimate_ik_error(h_y,yprd,natoms);
            ny_pppm++;
            h_y = yprd/ny_pppm;
        }

        err = estimate_ik_error(h_z,zprd,natoms);
        while (err > accuracy) {
            err = estimate_ik_error(h_z,zprd,natoms);
            nz_pppm++;
            h_z = zprd/nz_pppm;
        }
    }

    // boost grid size until it is factorable

    while (!factorable(nx_pppm)) nx_pppm++;
    while (!factorable(ny_pppm)) ny_pppm++;
    while (!factorable(nz_pppm)) nz_pppm++;

    // triclinic = 0
    h_x = xprd/nx_pppm;
    h_y = yprd/ny_pppm;
    h_z = zprd/nz_pppm;


    if (nx_pppm >= OFFSET || ny_pppm >= OFFSET || nz_pppm >= OFFSET)
        error->all(FLERR,"PPPM grid is too large");
}

void KSpaceNNP::compute_rho_coeff()
{
    int j,k,l,m;
    FFT_SCALAR s;

    FFT_SCALAR **a;
    memory->create2d_offset(a,order,-order,order,"pppm:a");

    for (k = -order; k <= order; k++)
        for (l = 0; l < order; l++)
            a[l][k] = 0.0;

    a[0][0] = 1.0;
    for (j = 1; j < order; j++) {
        for (k = -j; k <= j; k += 2) {
            s = 0.0;
            for (l = 0; l < j; l++) {
                a[l+1][k] = (a[l][k+1]-a[l][k-1]) / (l+1);
#ifdef FFT_SINGLE
                s += powf(0.5,(float) l+1) *
          (a[l][k-1] + powf(-1.0,(float) l) * a[l][k+1]) / (l+1);
#else
                s += pow(0.5,(double) l+1) *
                     (a[l][k-1] + pow(-1.0,(double) l) * a[l][k+1]) / (l+1);
#endif
            }
            a[0][k] = s;
        }
    }

    m = (1-order)/2;
    for (k = -(order-1); k < order; k += 2) {
        for (l = 0; l < order; l++)
            rho_coeff[l][m] = a[l][k];
        for (l = 1; l < order; l++)
            drho_coeff[l-1][m] = l*a[l][k];
        m++;
    }

    memory->destroy2d_offset(a,-order);
}

// pre-compute modified (Hockney-Eastwood) Coulomb Green's function
void KSpaceNNP::compute_gf_ik()
{
    const double * const prd = domain->prd;
    //TODO: conversion ?
    const double xprd = prd[0] * nnp->cflength;
    const double yprd = prd[1] * nnp->cflength;
    const double zprd = prd[2] * nnp->cflength;

    const double unitkx = (MY_2PI/xprd);
    const double unitky = (MY_2PI/yprd);
    const double unitkz = (MY_2PI/zprd);

    double snx,sny,snz;
    double argx,argy,argz,wx,wy,wz,sx,sy,sz,qx,qy,qz;
    double sum1,dot1,dot2;
    double numerator,denominator;
    double sqk;

    int k,l,m,n,nx,ny,nz,kper,lper,mper;

    const int nbx = static_cast<int> ((g_ewald*xprd/(MY_PI*nx_pppm)) *
                                      pow(-log(EPS_HOC),0.25));
    const int nby = static_cast<int> ((g_ewald*yprd/(MY_PI*ny_pppm)) *
                                      pow(-log(EPS_HOC),0.25));
    const int nbz = static_cast<int> ((g_ewald*zprd/(MY_PI*nz_pppm)) *
                                      pow(-log(EPS_HOC),0.25));
    const int twoorder = 2*order;

    n = 0;
    for (m = nzlo_fft; m <= nzhi_fft; m++) {
        mper = m - nz_pppm*(2*m/nz_pppm);
        snz = square(sin(0.5*unitkz*mper*zprd/nz_pppm));

        for (l = nylo_fft; l <= nyhi_fft; l++) {
            lper = l - ny_pppm*(2*l/ny_pppm);
            sny = square(sin(0.5*unitky*lper*yprd/ny_pppm));

            for (k = nxlo_fft; k <= nxhi_fft; k++) {
                kper = k - nx_pppm*(2*k/nx_pppm);
                snx = square(sin(0.5*unitkx*kper*xprd/nx_pppm));

                sqk = square(unitkx*kper) + square(unitky*lper) + square(unitkz*mper);

                if (sqk != 0.0) {
                    numerator = 12.5663706/sqk;
                    denominator = gf_denom(snx,sny,snz);
                    sum1 = 0.0;

                    for (nx = -nbx; nx <= nbx; nx++) {
                        qx = unitkx*(kper+nx_pppm*nx);
                        sx = exp(-0.25*square(qx/g_ewald));
                        argx = 0.5*qx*xprd/nx_pppm;
                        wx = powsinxx(argx,twoorder);

                        for (ny = -nby; ny <= nby; ny++) {
                            qy = unitky*(lper+ny_pppm*ny);
                            sy = exp(-0.25*square(qy/g_ewald));
                            argy = 0.5*qy*yprd/ny_pppm;
                            wy = powsinxx(argy,twoorder);

                            for (nz = -nbz; nz <= nbz; nz++) {
                                qz = unitkz*(mper+nz_pppm*nz);
                                sz = exp(-0.25*square(qz/g_ewald));
                                argz = 0.5*qz*zprd/nz_pppm;
                                wz = powsinxx(argz,twoorder);

                                dot1 = unitkx*kper*qx + unitky*lper*qy + unitkz*mper*qz;
                                dot2 = qx*qx+qy*qy+qz*qz;
                                sum1 += (dot1/dot2) * sx*sy*sz * wx*wy*wz;
                            }
                        }
                    }
                    greensfn[n++] = numerator*sum1/denominator;
                } else greensfn[n++] = 0.0;
            }
        }
    }
}

/* ----------------------------------------------------------------------
   calculate the final estimate of the accuracy
------------------------------------------------------------------------- */

double KSpaceNNP::final_accuracy()
{
    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd;
    bigint natoms = atom->natoms;
    if (natoms == 0) natoms = 1; // avoid division by zero

    double df_kspace = compute_df_kspace();
    double q2_over_sqrt = q2 / sqrt(natoms*cutoff*xprd*yprd*zprd);
    double df_rspace = 2.0 * q2_over_sqrt * exp(-g_ewald*g_ewald*cutoff*cutoff);
    double df_table = estimate_table_accuracy(q2_over_sqrt,df_rspace);
    double estimated_accuracy = sqrt(df_kspace*df_kspace + df_rspace*df_rspace +
                                     df_table*df_table);

    return estimated_accuracy;
}

/* ----------------------------------------------------------------------
   compute estimated kspace force error
------------------------------------------------------------------------- */

double KSpaceNNP::compute_df_kspace()
{
    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd;
    bigint natoms = atom->natoms;
    double df_kspace = 0.0;

    //differentiation_flag = 0
    double lprx = estimate_ik_error(h_x,xprd,natoms);
    double lpry = estimate_ik_error(h_y,yprd,natoms);
    double lprz = estimate_ik_error(h_z,zprd,natoms);
    df_kspace = sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0);

    return df_kspace;
}

/* ----------------------------------------------------------------------
   estimate kspace force error for ik method
------------------------------------------------------------------------- */

double KSpaceNNP::estimate_ik_error(double h, double prd, bigint natoms)
{
    double sum = 0.0;
    if (natoms == 0) return 0.0;
    for (int m = 0; m < order; m++)
        sum += acons[order][m] * pow(h*g_ewald,2.0*m);
    double value = q2 * pow(h*g_ewald,(double)order) *
                   sqrt(g_ewald*prd*sqrt(MY_2PI)*sum/natoms) / (prd*prd);

    return value;
}

void KSpaceNNP::procs2grid2d(int nprocs, int nx, int ny, int *px, int *py)
{
    // loop thru all possible factorizations of nprocs
    // surf = surface area of largest proc sub-domain
    // innermost if test minimizes surface area and surface/volume ratio

    int bestsurf = 2 * (nx + ny);
    int bestboxx = 0;
    int bestboxy = 0;

    int boxx,boxy,surf,ipx,ipy;

    ipx = 1;
    while (ipx <= nprocs) {
        if (nprocs % ipx == 0) {
            ipy = nprocs/ipx;
            boxx = nx/ipx;
            if (nx % ipx) boxx++;
            boxy = ny/ipy;
            if (ny % ipy) boxy++;
            surf = boxx + boxy;
            if (surf < bestsurf ||
                (surf == bestsurf && boxx*boxy > bestboxx*bestboxy)) {
                bestsurf = surf;
                bestboxx = boxx;
                bestboxy = boxy;
                *px = ipx;
                *py = ipy;
            }
        }
        ipx++;
    }
}

// pre-compute Green's function denominator expansion coeffs, Gamma(2n)
void KSpaceNNP::compute_gf_denom()
{
    int k,l,m;

    for (l = 1; l < order; l++) gf_b[l] = 0.0;
    gf_b[0] = 1.0;

    for (m = 1; m < order; m++) {
        for (l = m; l > 0; l--)
            gf_b[l] = 4.0 * (gf_b[l]*(l-m)*(l-m-0.5)-gf_b[l-1]*(l-m-1)*(l-m-1));
        gf_b[0] = 4.0 * (gf_b[0]*(l-m)*(l-m-0.5));
    }

    bigint ifact = 1;
    for (k = 1; k < 2*order; k++) ifact *= k;
    double gaminv = 1.0/ifact;
    for (l = 0; l < order; l++) gf_b[l] *= gaminv;
}

int KSpaceNNP::factorable(int n)
{
    int i;

    while (n > 1) {
        for (i = 0; i < nfactors; i++) {
            if (n % factors[i] == 0) {
                n /= factors[i];
                break;
            }
        }
        if (i == nfactors) return 0;
    }

    return 1;
}

/* ----------------------------------------------------------------------
   adjust the g_ewald parameter to near its optimal value
   using a Newton-Raphson solver
------------------------------------------------------------------------- */

void KSpaceNNP::adjust_gewald()
{
    double dx;

    for (int i = 0; i < LARGE; i++) {
        dx = newton_raphson_f() / derivf();
        g_ewald -= dx;
        if (fabs(newton_raphson_f()) < SMALL) return;
    }
    error->all(FLERR, "Could not compute g_ewald");
}

/* ----------------------------------------------------------------------
   calculate f(x) using Newton-Raphson solver
------------------------------------------------------------------------- */

double KSpaceNNP::newton_raphson_f()
{
    double xprd = domain->xprd;
    double yprd = domain->yprd;
    double zprd = domain->zprd;
    bigint natoms = atom->natoms;

    double df_rspace = 2.0*q2*exp(-g_ewald*g_ewald*cutoff*cutoff) /
                       sqrt(natoms*cutoff*xprd*yprd*zprd);

    double df_kspace = compute_df_kspace();

    return df_rspace - df_kspace;
}

/* ----------------------------------------------------------------------
   calculate numerical derivative f'(x) using forward difference
   [f(x + h) - f(x)] / h
------------------------------------------------------------------------- */

double KSpaceNNP::derivf()
{
    double h = 0.000001;  //Derivative step-size
    double df,f1,f2,g_ewald_old;

    f1 = newton_raphson_f();
    g_ewald_old = g_ewald;
    g_ewald += h;
    f2 = newton_raphson_f();
    g_ewald = g_ewald_old;
    df = (f2 - f1)/h;

    return df;
}

// Calculate Ewald eta param (original method in RuNNer - JACKSON_CATLOW)
void KSpaceNNP::calculate_ewald_eta(int mflag)
{
    ewald_eta = 1.0 / sqrt(2.0 * M_PI);

    //TODO: check
    if (mflag == 0) ewald_eta *= pow(volume * volume / atom->natoms, 1.0 / 6.0); // regular Ewald eta
    else            ewald_eta *= pow(volume, 1.0 / 3.0); // matrix approach

}

// Calculate Ewald eta param (efficient method - KOLAFA_PARRAM)
void KSpaceNNP::calculate_ewald_eta_efficient()
{
    // Ratio of computing times for one real space and k space iteration.
    double TrOverTk = 3.676; // TODO: should it be hardcoded ?

    // Unit Conversion (in KOLAFA-PERRAM method precision has unit of a force) TODO:check
    double fourPiEps = 1.0;

    /*precision *= convEnergy / convLength;
    ewaldMaxCharge *= convCharge;
    fourPiEps = pow(convCharge, 2) / (convLength * convEnergy);*/

    // Initial approximation
    double eta0 = pow(1 / TrOverTk * pow(volume, 2) / pow(2 * M_PI, 3),1.0 / 6.0);

    // Selfconsistent calculation of eta
    ewald_eta = eta0;
    s_ewald = 0.0; // TODO: check
    double relError = 1.0;
    while (relError > 0.01)
    {
        // Calculates S
        double y = accuracy * sqrt(ewald_eta / sqrt(2)) * fourPiEps;
        y /= 2 * sqrt(atom->natoms * 1.0 / volume) * pow(ewald_max_charge, 2);

        double relYError = 1.0;
        if (s_ewald <= 0.0)
            s_ewald = 0.5;
        double step = s_ewald;
        while (abs(step) / s_ewald > 0.01 || relYError > 0.01)
        {
            step = 2 * s_ewald / (4 * pow(s_ewald,2) + 1);
            step *= 1 -  sqrt(s_ewald) * y / exp(-pow(s_ewald,2));
            if (s_ewald <= -step)
            {
                s_ewald /= 2;
                step = 1.0;
            }
            else
                s_ewald += step;
            relYError = (exp(-pow(s_ewald,2)) / sqrt(s_ewald) - y) / y;
        }

        double newEta = eta0 * pow((1 + 1 / (2 * pow(s_ewald, 2))), 1.0 / 6.0);
        relError = abs(newEta - ewald_eta) / ewald_eta;
        ewald_eta = newEta;
    }

    ewald_eta = max(ewald_eta, ewald_max_sigma);
}

// Generate k-space grid in Ewald Sum (RuNNer)
void KSpaceNNP::ewald_pbc(double rcut)
{
    double proja = fabs(unitk[0]);
    double projb = fabs(unitk[1]);
    double projc = fabs(unitk[2]);
    kxmax = 0;
    kymax = 0;
    kzmax = 0;
    while (kxmax * proja <= rcut) kxmax++;
    while (kymax * projb <= rcut) kymax++;
    while (kzmax * projc <= rcut) kzmax++;

    return;
}



