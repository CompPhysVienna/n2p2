/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_qeq_gaussian.h"
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <stdlib.h>  //exit(0);
#include "pair_nnp.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "force.h"
#include "group.h"
#include "pair.h"
#include "respa.h"
#include "memory.h"
#include "citeme.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define EV_TO_KCAL_PER_MOL 14.4
//#define DANGER_ZONE     0.95
//#define LOOSE_ZONE      0.7
#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))
#define MIN_NBRS 100
#define MIN_CAP  50
#define DANGER_ZONE    0.90
#define SAFE_ZONE      1.2

/* ---------------------------------------------------------------------- */

FixQEqGaussian::FixQEqGaussian(LAMMPS *lmp, int narg, char **arg) :
        Fix(lmp, narg, arg), pertype_option(NULL)
{
    //TODO: we plan to invoke this fix without requiring a user-defined fix command
    //if (narg<8 || narg>9) error->all(FLERR,"Illegal fix qeq/reax command");

    //nevery = force->inumeric(FLERR,arg[3]);
    //if (nevery <= 0) error->all(FLERR,"Illegal fix qeq/reax command");

    //swa = force->numeric(FLERR,arg[4]);
    //swb = force->numeric(FLERR,arg[5]);
    //tolerance = force->numeric(FLERR,arg[6]);
    //int len = strlen(arg[7]) + 1;
    //pertype_option = new char[len];
    //strcpy(pertype_option,arg[7]);

    // dual CG support only available for USER-OMP variant
    // check for compatibility is in Fix::post_constructor()
    //dual_enabled = 0;
    //if (narg == 9) {
    //    if (strcmp(arg[8],"dual") == 0) dual_enabled = 1;
    //    else error->all(FLERR,"Illegal fix qeq/reax command");
    //}

    n = n_cap = 0;
    N = nmax = 0;
    m_fill = m_cap = 0;
    pack_flag = 0;
    Q = NULL;

    nprev = 4;

    Adia_inv = NULL;
    b = NULL;
    b_prc = NULL;
    b_prm = NULL;

    // CG
    p = NULL;
    q = NULL;
    r = NULL;
    d = NULL;

    // H matrix
    A.firstnbr = NULL;
    A.numnbrs = NULL;
    A.jlist = NULL;
    A.val = NULL;

    // dual CG support
    // Update comm sizes for this fix
    //if (dual_enabled) comm_forward = comm_reverse = 2;
    //else comm_forward = comm_reverse = 1;

    // perform initial allocation of atom-based arrays
    // register with Atom class

    //nnp = NULL;
    Q_hist = NULL;
    grow_arrays(atom->nmax+1);
    atom->add_callback(0);

    for (int i = 0; i < atom->nmax + 1; i++)
        for (int j = 0; j < nprev; ++j)
            Q_hist[i][j] = 0;
}

/* ---------------------------------------------------------------------- */

FixQEqGaussian::~FixQEqGaussian()
{
    if (copymode) return;

    delete[] pertype_option;

    // unregister callbacks to this fix from Atom class

    atom->delete_callback(id,0);

    memory->destroy(Q_hist);

    deallocate_storage();
    deallocate_matrix();

    memory->destroy(chi);
    memory->destroy(hardness);
    memory->destroy(sigma);
}

/* ---------------------------------------------------------------------- */

//TODO: check this later
void FixQEqGaussian::post_constructor()
{

    Pair *pair = force->pair_match("nnp",0);

    int tmp,ntypes;
    ntypes = atom->ntypes;

    chi = (double *) pair->extract("chi",tmp);
    hardness = (double *) pair->extract("hardness",tmp);
    sigma = (double *) pair->extract("sigma",tmp);
    if (chi == NULL || hardness == NULL || sigma == NULL)
           error->all(FLERR,"Could not extract qeq params from pair nnp");

    MPI_Bcast(&chi[1],ntypes,MPI_DOUBLE,0,world);
    MPI_Bcast(&hardness[1],ntypes,MPI_DOUBLE,0,world);
    MPI_Bcast(&sigma[1],ntypes,MPI_DOUBLE,0,world);

    return;

}

/* ---------------------------------------------------------------------- */

int FixQEqGaussian::setmask()
{
    int mask = 0;
    mask |= POST_FORCE;
    mask |= PRE_FORCE;
    mask |= PRE_FORCE_RESPA;
    mask |= MIN_PRE_FORCE;
    return mask;
}


/* ---------------------------------------------------------------------- */

void FixQEqGaussian::allocate_storage()
{
    std::cout << atom->nmax << '\n';
    exit(0);
    nmax = atom->nmax;
    exit(0);
    std::cout << Q << '\n';

    // TODO: check all of these +1 values (LM)
    memory->create(Q,nmax+1,"qeq_gaussian:q");

    memory->create(Adia_inv,nmax+1,"qeq_gaussian:Adia_inv");
    memory->create(b,nmax+1,"qeq_gaussian:b");
    memory->create(b_prc,nmax+1,"qeq_gaussian:b_prc");
    memory->create(b_prm,nmax+1,"qeq_gaussian:b_prm");

    // dual CG support
    int size = nmax;
    if (dual_enabled) size*= 2;

    size = size + 1;
    memory->create(p,size,"qeq_gaussian:p");
    memory->create(q,size,"qeq_gaussian:q");
    memory->create(r,size,"qeq_gaussian:r");
    memory->create(d,size,"qeq_gaussian:d");
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::deallocate_storage()
{

    memory->destroy(Adia_inv);
    memory->destroy(b);
    memory->destroy( b_prc );
    memory->destroy( b_prm );

    memory->destroy( p );
    memory->destroy( q );
    memory->destroy( r );
    memory->destroy( d );
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::reallocate_storage()
{
    deallocate_storage();
    allocate_storage();
    init_storage();
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::allocate_matrix()
{
    int i,ii,inum,m;
    int *ilist, *numneigh;

    int mincap;
    double safezone;

    mincap = MIN_CAP;
    safezone = SAFE_ZONE;

    n = atom->nlocal;
    n_cap = MAX( (int)(n * safezone), mincap);

    // determine the total space for the A matrix
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;

    m = 0;
    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        m += numneigh[i];
    }
    m_cap = MAX( (int)(m * safezone), mincap * MIN_NBRS);

    A.n = n_cap;
    A.m = m_cap;
    memory->create(A.firstnbr,n_cap,"qeq_gaussian:A.firstnbr");
    memory->create(A.numnbrs,n_cap,"qeq_gaussian:A.numnbrs");
    memory->create(A.jlist,m_cap,"qeq_gaussian:A.jlist");
    memory->create(A.val,m_cap,"qeq_gaussian:A.val");
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::deallocate_matrix()
{
    memory->destroy( A.firstnbr );
    memory->destroy( A.numnbrs );
    memory->destroy( A.jlist );
    memory->destroy( A.val );
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::reallocate_matrix()
{
    deallocate_matrix();
    allocate_matrix();
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::init()
{
    if (!atom->q_flag)
        error->all(FLERR,"Missing atom attribute q");

    // need a half neighbor list w/ Newton off and ghost neighbors
    // built whenever re-neighboring occurs

    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->fix = 1;
    neighbor->requests[irequest]->newton = 2;
    neighbor->requests[irequest]->ghost = 1;

}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::init_list(int /*id*/, NeighList *ptr)
{
    list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::setup_pre_force(int vflag)
{
    deallocate_storage();
    allocate_storage();

    init_storage();

    deallocate_matrix();
    allocate_matrix();

    pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::setup_pre_force_respa(int vflag, int ilevel)
{
    if (ilevel < nlevels_respa-1) return;
    setup_pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::min_setup_pre_force(int vflag)
{
    setup_pre_force(vflag);
}

//TODO: check me
void FixQEqGaussian::init_storage()
{
    int NN;

    NN = list->inum + list->gnum;

    for (int i = 0; i < NN; i++) {
        Adia_inv[i] = 1. / (hardness[atom->type[i]] + (1. / (sigma[atom->type[i]] * sqrt(M_PI))));
        b[i] = -chi[atom->tag[i]]; //TODO: our electronegativities are not type-dependent !!!
        b_prc[i] = 0;
        b_prm[i] = 0;
        Q[i] = 0;
    }
    Adia_inv[NN] = 0.0;
    b[NN] = 0.0; //TODO: Qref normally !!
}

/* ---------------------------------------------------------------------- */
//TODO: this is the main control routine
void FixQEqGaussian::pre_force(int /*vflag*/)
{

    double t_start, t_end;

    if (update->ntimestep % nevery) return;
    if (comm->me == 0) t_start = MPI_Wtime();

    n = atom->nlocal;
    N = atom->nlocal + atom->nghost;

    // grow arrays if necessary
    // need to be atom->nmax in length

    if (atom->nmax > nmax) reallocate_storage();
    if (n > n_cap*DANGER_ZONE || m_fill > m_cap*DANGER_ZONE)
        reallocate_matrix();

    init_matvec();
    matvecs = CG(b, Q);  // CG on parallel
    calculate_Q();

    if (comm->me == 0) {
        t_end = MPI_Wtime();
        qeq_time = t_end - t_start;
    }
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::pre_force_respa(int vflag, int ilevel, int /*iloop*/)
{
    if (ilevel == nlevels_respa-1) pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::min_pre_force(int vflag)
{
    pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::init_matvec()
{
    /* fill-in A matrix */
    compute_A();

    int nn, ii, i;
    int *ilist;

    nn = list->inum;
    ilist = list->ilist;

    for (ii = 0; ii < nn; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit) {

            //TODO: discuss this
            /* init pre-conditioner for H and init solution vectors */
            Adia_inv[i] = 1. / (hardness[atom->type[i]] + (1. / (sigma[atom->type[i]] * sqrt(M_PI))));
            b[i]      = -chi[ atom->tag[i] ]; //TODO:check this tag (it was type!)

            /* quadratic extrapolation for s & t from previous solutions */
            //t[i] = t_hist[i][2] + 3 * ( t_hist[i][0] - t_hist[i][1]);

            //TODO: discuss this
            /* cubic extrapolation for q from previous solutions */
            Q[i] = 4*(Q_hist[i][0]+Q_hist[i][2])-(6*Q_hist[i][1]+Q_hist[i][3]);
        }
    }
    // add lm here ?
    Adia_inv[nn] = 0.0;
    b[nn] = 0.0;

    pack_flag = 2;
    comm->forward_comm_fix(this); //Dist_vector( s );
    pack_flag = 3;
    comm->forward_comm_fix(this); //Dist_vector( t );
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::compute_A()
{
    int inum, jnum, *ilist, *jlist, *numneigh, **firstneigh;
    int i, j, ii, jj, flag;
    double dx, dy, dz, r_sqr;
    const double SMALL = 0.0001;

    int *type = atom->type;
    tagint *tag = atom->tag;
    double **x = atom->x;
    int *mask = atom->mask;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    // fill in the A matrix
    //TODO: check if everything is correct with regard to the LM
    m_fill = 0;
    r_sqr = 0;
    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        if (mask[i] & groupbit) {
            jlist = firstneigh[i];
            jnum = numneigh[i];
            A.firstnbr[i] = m_fill;

            for (jj = 0; jj < jnum; jj++) {
                j = jlist[jj];
                j &= NEIGHMASK;

                dx = x[j][0] - x[i][0];
                dy = x[j][1] - x[i][1];
                dz = x[j][2] - x[i][2];
                r_sqr = SQR(dx) + SQR(dy) + SQR(dz);

                flag = 0;
                if (r_sqr <= SQR(10.0)) { //TODO: CHECK ME !! (10.0 is actually a radius)
                    if (j < n) flag = 1;
                    else if (tag[i] < tag[j]) flag = 1;
                    else if (tag[i] == tag[j]) {
                        if (dz > SMALL) flag = 1;
                        else if (fabs(dz) < SMALL) {
                            if (dy > SMALL) flag = 1;
                            else if (fabs(dy) < SMALL && dx > SMALL)
                                flag = 1;
                        }
                    }
                }

                if (flag) {
                    A.jlist[m_fill] = j;
                    A.val[m_fill] = calculate_A( sqrt(r_sqr), sigma[type[i]], sigma[type[j]]);
                    m_fill++;
                }
            }
            A.numnbrs[i] = m_fill - A.firstnbr[i];
        }
    }

    if (m_fill >= A.m) {
        char str[128];
        sprintf(str,"A matrix size has been exceeded: m_fill=%d A.m=%d\n",
                m_fill, A.m);
        error->warning(FLERR,str);
        error->all(FLERR,"Insufficient QEq matrix size");
    }
}

/* ---------------------------------------------------------------------- */

double FixQEqGaussian::calculate_A( double r, double sigma_i, double sigma_j)
{

    double nom, denom, res;

    if (true) //non-periodic A matrix
    {
        nom =  erf(r / sqrt(2.0 * (sigma_i*sigma_i + sigma_j*sigma_j)));
        denom = r;
        res = nom / denom;
    }

    return res;
}

/* ---------------------------------------------------------------------- */

int FixQEqGaussian::CG( double *b, double *x)
{
    int  i, j, imax;
    double tmp, alpha, beta, b_norm;
    double sig_old, sig_new;

    int nn, jj;
    int *ilist;

    nn = list->inum;
    ilist = list->ilist;

    imax = 200;

    pack_flag = 1;
    sparse_matvec( &A, x, q);
    comm->reverse_comm_fix(this); //Coll_Vector( q );

    vector_sum( r , 1.,  b, -1., q, nn);

    for (jj = 0; jj < nn+1; ++jj) { //TODO:check +1 (LM)
        j = ilist[jj];
        if (atom->mask[j] & groupbit)
            d[j] = r[j] * Adia_inv[j]; //pre-condition
    }

    b_norm = parallel_norm( b, nn);
    sig_new = parallel_dot( r, d, nn);

    for (i = 1; i < imax && sqrt(sig_new) / b_norm > tolerance; ++i) {
        comm->forward_comm_fix(this); //Dist_vector( d );
        sparse_matvec( &A, d, q );
        comm->reverse_comm_fix(this); //Coll_vector( q );

        tmp = parallel_dot( d, q, nn);
        alpha = sig_new / tmp;

        vector_add( x, alpha, d, nn );
        vector_add( r, -alpha, q, nn );

        // pre-conditioning
        for (jj = 0; jj < nn; ++jj) {
            j = ilist[jj];
            if (atom->mask[j] & groupbit)
                p[j] = r[j] * Adia_inv[j];
        }

        sig_old = sig_new;
        sig_new = parallel_dot( r, p, nn);

        beta = sig_new / sig_old;
        vector_sum( d, 1., p, beta, d, nn );
    }

    if (i >= imax && comm->me == 0) {
        char str[128];
        sprintf(str,"CG convergence failed after %d iterations "
                    "at " BIGINT_FORMAT " step",i,update->ntimestep);
        error->warning(FLERR,str);
    }

    return i;
}


/* ---------------------------------------------------------------------- */

void FixQEqGaussian::sparse_matvec( sparse_matrix *A, double *x, double *b)
{
    int i, j, itr_j;
    int nn, NN, ii;
    int *ilist;

    nn = list->inum;
    NN = list->inum + list->gnum;
    ilist = list->ilist;

    for (ii = 0; ii < nn + 1; ++ii) { //TODO: +1 ???
        i = ilist[ii];
        if (atom->mask[i] & groupbit)
            b[i] = (hardness[ atom->type[i] ] + (1. / (sigma[atom->type[i]] * sqrt(M_PI)))) * x[i]; //TODO: check me
    }

    for (ii = nn; ii < NN + 1; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit)
            b[i] = 0;
    }

    for (ii = 0; ii < nn + 1; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit) {
            for (itr_j=A->firstnbr[i]; itr_j<A->firstnbr[i]+A->numnbrs[i]; itr_j++) {
                j = A->jlist[itr_j];
                b[i] += A->val[itr_j] * x[j];
                b[j] += A->val[itr_j] * x[i];
            }
        }
    }

}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::calculate_Q()
{
    int i, k;
    double u, Q_sum;
    double *q = atom->q;

    int nn, ii;
    int *ilist;

    nn = list->inum;
    ilist = list->ilist;

    Q_sum = parallel_vector_acc( Q, nn);

    for (ii = 0; ii < nn; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit) {
            q[i] = Q[i];
            /* backup Q */
            for (k = nprev-1; k > 0; --k) {
                Q_hist[i][k] = Q_hist[i][k-1];
            }
            Q_hist[i][0] = Q[i];
        }
    }

    pack_flag = 4;
    comm->forward_comm_fix(this); //Dist_vector( atom->q );
}

/* ---------------------------------------------------------------------- */

int FixQEqGaussian::pack_forward_comm(int n, int *list, double *buf,
                                  int /*pbc_flag*/, int * /*pbc*/)
{
    int m;

    //TODO: check me -> communication here
    if (pack_flag == 1)
        for(m = 0; m < n; m++) buf[m] = d[list[m]];
    else if (pack_flag == 2)
        for(m = 0; m < n; m++) buf[m] = Q[list[m]];
    else if (pack_flag == 3)
        for(m = 0; m < n; m++) buf[m] = Q[list[m]];
    else if (pack_flag == 4)
        for(m = 0; m < n; m++) buf[m] = atom->q[list[m]];
    else if (pack_flag == 5) {
        m = 0;
        for(int i = 0; i < n; i++) {
            int j = 2 * list[i];
            buf[m++] = d[j  ];
            buf[m++] = d[j+1];
        }
        return m;
    }
    return n;
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::unpack_forward_comm(int n, int first, double *buf)
{
    int i, m;

    if (pack_flag == 1)
        for(m = 0, i = first; m < n; m++, i++) d[i] = buf[m];
    else if (pack_flag == 2)
        for(m = 0, i = first; m < n; m++, i++) Q[i] = buf[m];
    else if (pack_flag == 3)
        for(m = 0, i = first; m < n; m++, i++) Q[i] = buf[m];
    else if (pack_flag == 4)
        for(m = 0, i = first; m < n; m++, i++) atom->q[i] = buf[m];
    else if (pack_flag == 5) {
        int last = first + n;
        m = 0;
        for(i = first; i < last; i++) {
            int j = 2 * i;
            d[j  ] = buf[m++];
            d[j+1] = buf[m++];
        }
    }
}

/* ---------------------------------------------------------------------- */

int FixQEqGaussian::pack_reverse_comm(int n, int first, double *buf)
{
    int i, m;
    if (pack_flag == 5) {
        m = 0;
        int last = first + n;
        for(i = first; i < last; i++) {
            int indxI = 2 * i;
            buf[m++] = q[indxI  ];
            buf[m++] = q[indxI+1];
        }
        return m;
    } else {
        for (m = 0, i = first; m < n; m++, i++) buf[m] = q[i];
        return n;
    }
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::unpack_reverse_comm(int n, int *list, double *buf)
{
    if (pack_flag == 5) {
        int m = 0;
        for(int i = 0; i < n; i++) {
            int indxI = 2 * list[i];
            q[indxI  ] += buf[m++];
            q[indxI+1] += buf[m++];
        }
    } else {
        for (int m = 0; m < n; m++) q[list[m]] += buf[m];
    }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixQEqGaussian::memory_usage()
{
    double bytes;

    bytes = atom->nmax*nprev*2 * sizeof(double); // Q_hist
    bytes += atom->nmax*11 * sizeof(double); // storage
    bytes += n_cap*2 * sizeof(int); // matrix...
    bytes += m_cap * sizeof(int);
    bytes += m_cap * sizeof(double);

    if (dual_enabled)
        bytes += atom->nmax*4 * sizeof(double); // double size for q, d, r, and p

    return bytes;
}

/* ----------------------------------------------------------------------
   allocate fictitious charge arrays
------------------------------------------------------------------------- */

void FixQEqGaussian::grow_arrays(int nmax)
{
    memory->grow(Q_hist,nmax,nprev,"qeq:Q_hist");
    //memory->grow(t_hist,nmax,nprev,"qeq:t_hist");
}

/* ----------------------------------------------------------------------
   copy values within fictitious charge arrays
------------------------------------------------------------------------- */

void FixQEqGaussian::copy_arrays(int i, int j, int /*delflag*/)
{
    for (int m = 0; m < nprev; m++) {
        Q_hist[j][m] = Q_hist[i][m];
    }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int FixQEqGaussian::pack_exchange(int i, double *buf)
{
    for (int m = 0; m < nprev; m++) buf[m] = Q_hist[i][m];
    //for (int m = 0; m < nprev; m++) buf[nprev+m] = t_hist[i][m];
    return nprev*2;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixQEqGaussian::unpack_exchange(int nlocal, double *buf)
{
    for (int m = 0; m < nprev; m++) Q_hist[nlocal][m] = buf[m];
    //for (int m = 0; m < nprev; m++) t_hist[nlocal][m] = buf[nprev+m];
    return nprev*2;
}

/* ---------------------------------------------------------------------- */

double FixQEqGaussian::parallel_norm( double *v, int n)
{
    int  i;
    double my_sum, norm_sqr;

    int ii;
    int *ilist;

    ilist = list->ilist;

    my_sum = 0.0;
    norm_sqr = 0.0;
    for (ii = 0; ii < n; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit)
            my_sum += SQR( v[i]);
    }

    MPI_Allreduce( &my_sum, &norm_sqr, 1, MPI_DOUBLE, MPI_SUM, world);

    return sqrt( norm_sqr);
}

/* ---------------------------------------------------------------------- */

double FixQEqGaussian::parallel_dot( double *v1, double *v2, int n)
{
    int  i;
    double my_dot, res;

    int ii;
    int *ilist;

    ilist = list->ilist;

    my_dot = 0.0;
    res = 0.0;
    for (ii = 0; ii < n; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit)
            my_dot += v1[i] * v2[i];
    }

    MPI_Allreduce( &my_dot, &res, 1, MPI_DOUBLE, MPI_SUM, world);

    return res;
}

/* ---------------------------------------------------------------------- */

double FixQEqGaussian::parallel_vector_acc( double *v, int n)
{
    int  i;
    double my_acc, res;

    int ii;
    int *ilist;

    ilist = list->ilist;

    my_acc = 0.0;
    res = 0.0;
    for (ii = 0; ii < n; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit)
            my_acc += v[i];
    }

    MPI_Allreduce( &my_acc, &res, 1, MPI_DOUBLE, MPI_SUM, world);

    return res;
}

/* ---------------------------------------------------------------------- */

void FixQEqGaussian::vector_sum( double* dest, double c, double* v,
                             double d, double* y, int k)
{
    int kk;
    int *ilist;

    ilist = list->ilist;

    for (--k; k>=0; --k) {
        kk = ilist[k];
        if (atom->mask[kk] & groupbit)
            dest[kk] = c * v[kk] + d * y[kk];
    }
}

void FixQEqGaussian::vector_add( double* dest, double c, double* v, int k)
{
    int kk;
    int *ilist;

    ilist = list->ilist;

    for (--k; k>=0; --k) {
        kk = ilist[k];
        if (atom->mask[kk] & groupbit)
            dest[kk] += c * v[kk];
    }
}

int FixQEqGaussian::test_dummy()
{
    return 18;
}

void FixQEqGaussian::QEq_serial(double* chi, double* hardness, double* sigma, double* Q)
{
    //manage storage and other setup & fill matrix A
    //deallocate_storage();
    allocate_storage();

    //init_storage_serial();

    //deallocate_matrix();
    //allocate_matrix();


    //exit(0);
    //fill matrix A and run CG in serial
    //runQEq_serial();

}

void FixQEqGaussian::init_storage_serial()
{
    int NN;

    NN = list->inum + list->gnum;

    for (int i = 0; i < NN; i++) {
        Adia_inv[i] = 1. / (hardness[atom->type[i]] + (1. / (sigma[atom->type[i]] * sqrt(M_PI))));
        b[i] = -chi[atom->tag[i]]; //TODO: our electronegativities are not type-dependent !!!
        b_prc[i] = 0;
        b_prm[i] = 0;
        Q[i] = 0;
    }
    Adia_inv[NN] = 0.0;
    b[NN] = 0.0; //TODO: Qref normally !!
}

void FixQEqGaussian::runQEq_serial()
{

    double t_start, t_end;

    if (update->ntimestep % nevery) return;

    n = atom->nlocal;
    N = atom->nlocal + atom->nghost;

    // grow arrays if necessary
    // need to be atom->nmax in length

    if (atom->nmax > nmax) reallocate_storage();
    if (n > n_cap*DANGER_ZONE || m_fill > m_cap*DANGER_ZONE)
        reallocate_matrix();

    init_matvec_serial();
    matvecs = CG_serial(b, Q);
    //calculate_Q_serial();

}

int FixQEqGaussian::CG_serial( double *b, double *x)
{
    int  i, j, imax;
    double tmp, alpha, beta, b_norm;
    double sig_old, sig_new;

    int nn, jj;
    int *ilist;

    nn = list->inum;
    ilist = list->ilist;

    imax = 200;

    sparse_matvec( &A, x, q);
    vector_sum( r , 1.,  b, -1., q, nn);

    for (jj = 0; jj < nn+1; ++jj) { //TODO:check +1 (LM)
        j = ilist[jj];
        if (atom->mask[j] & groupbit)
            d[j] = r[j] * Adia_inv[j]; //pre-condition
    }

    b_norm = serial_norm( b, nn);
    sig_new = serial_dot( r, d, nn);

    for (i = 1; i < imax && sqrt(sig_new) / b_norm > tolerance; ++i) {
        sparse_matvec( &A, d, q );

        tmp = serial_dot( d, q, nn);
        alpha = sig_new / tmp;

        vector_add( x, alpha, d, nn );
        vector_add( r, -alpha, q, nn );

        // pre-conditioning
        for (jj = 0; jj < nn; ++jj) {
            j = ilist[jj];
            if (atom->mask[j] & groupbit)
                p[j] = r[j] * Adia_inv[j];
        }

        sig_old = sig_new;
        sig_new = serial_dot( r, p, nn);

        beta = sig_new / sig_old;
        vector_sum( d, 1., p, beta, d, nn );
    }

    if (i >= imax && comm->me == 0) {
        char str[128];
        sprintf(str,"CG convergence failed after %d iterations "
                    "at " BIGINT_FORMAT " step",i,update->ntimestep);
        error->warning(FLERR,str);
    }

    return i;
}

void FixQEqGaussian::calculate_Q_serial()
{
    int i, k;
    double u, Q_sum;
    double *q = atom->q;

    int nn, ii;
    int *ilist;

    nn = list->inum;
    ilist = list->ilist;

    Q_sum = serial_vector_acc( Q, nn);

    for (ii = 0; ii < nn; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit) {
            q[i] = Q[i];
            /* backup Q */
            for (k = nprev-1; k > 0; --k) {
                Q_hist[i][k] = Q_hist[i][k-1];
            }
            Q_hist[i][0] = Q[i];
        }
    }

}

void FixQEqGaussian::init_matvec_serial()
{
    /* fill-in A matrix */
    compute_A();

    int nn, ii, i;
    int *ilist;

    nn = list->inum;
    ilist = list->ilist;

    for (ii = 0; ii < nn; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit) {

            //TODO: discuss this
            /* init pre-conditioner for H and init solution vectors */
            Adia_inv[i] = 1. / (hardness[atom->type[i]] + (1. / (sigma[atom->type[i]] * sqrt(M_PI))));
            b[i]      = -chi[ atom->tag[i] ]; //TODO:check this tag (it was type!)

            /* quadratic extrapolation for s & t from previous solutions */
            //t[i] = t_hist[i][2] + 3 * ( t_hist[i][0] - t_hist[i][1]);

            //TODO: discuss this
            /* cubic extrapolation for q from previous solutions */
            Q[i] = 4*(Q_hist[i][0]+Q_hist[i][2])-(6*Q_hist[i][1]+Q_hist[i][3]);
        }
    }
    // add lm here ?
    Adia_inv[nn] = 0.0;
    b[nn] = 0.0;
}

double FixQEqGaussian::serial_norm( double *v, int n)
{
    int  i;
    double my_sum, norm_sqr;

    int ii;
    int *ilist;

    ilist = list->ilist;

    my_sum = 0.0;
    norm_sqr = 0.0;
    for (ii = 0; ii < n; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit)
            my_sum += SQR( v[i]);
    }

    return sqrt( norm_sqr);
}

double FixQEqGaussian::serial_dot( double *v1, double *v2, int n)
{
    int  i;
    double my_dot, res;

    int ii;
    int *ilist;

    ilist = list->ilist;

    my_dot = 0.0;
    res = 0.0;
    for (ii = 0; ii < n; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit)
            my_dot += v1[i] * v2[i];
    }

    return res;
}

double FixQEqGaussian::serial_vector_acc( double *v, int n)
{
    int  i;
    double my_acc, res;

    int ii;
    int *ilist;

    ilist = list->ilist;

    my_acc = 0.0;
    res = 0.0;
    for (ii = 0; ii < n; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit)
            my_acc += v[i];
    }

    return res;
}