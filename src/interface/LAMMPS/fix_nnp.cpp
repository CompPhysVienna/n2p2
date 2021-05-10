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

#include "fix_nnp.h"
#include <iostream>
#include <gsl/gsl_multimin.h>
#include <mpi.h>
#include <cmath>
#include <cstring>
#include "pair_nnp.h"
#include "atom.h"
#include "comm.h"
#include "domain.h" // check for periodicity
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "force.h"
#include "group.h"
#include "pair.h"
//#include "respa.h"
#include "memory.h"
#include "citeme.h"
#include "error.h"
#include "utils.h"


using namespace LAMMPS_NS;
using namespace FixConst;

#define EV_TO_KCAL_PER_MOL 14.4
#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))
#define MIN_NBRS 100
#define SAFE_ZONE      1.2
#define DANGER_ZONE    0.90
#define MIN_CAP        50


FixNNP::FixNNP(LAMMPS *lmp, int narg, char **arg) :
        Fix(lmp, narg, arg), pertype_option(NULL)
{
    //TODO: this is designed for a normal fix that is invoked in the LAMMPS script
    if (narg<8 || narg>9) error->all(FLERR,"Illegal fix nnp command");

    nevery = utils::inumeric(FLERR,arg[3],false,lmp);
    if (nevery <= 0) error->all(FLERR,"Illegal fix nnp command");

    dum1 = utils::numeric(FLERR,arg[4],false,lmp);
    dum2 = utils::numeric(FLERR,arg[5],false,lmp);

    tolerance = utils::numeric(FLERR,arg[6],false,lmp);
    int len = strlen(arg[7]) + 1;
    pertype_option = new char[len];
    strcpy(pertype_option,arg[7]);

    // dual CG support only available for USER-OMP variant
    // check for compatibility is in Fix::post_constructor()
    //dual_enabled = 0;
    //if (narg == 9) {
    //    if (strcmp(arg[8],"dual") == 0) dual_enabled = 1;
    //    else error->all(FLERR,"Illegal fix qeq/reax command");
    //}
    //shld = NULL;

    n = n_cap = 0;
    N = nmax = 0;
    m_fill = m_cap = 0;
    pack_flag = 0;
    Q = NULL;
    nprev = 4;


    nnp = NULL;
    nnp = (PairNNP *) force->pair_match("^nnp",0);
    nnp->chi = NULL;
    nnp->hardness = NULL;
    nnp->sigma = NULL;


    comm_forward = comm_reverse = 1;

    E_elec = 0.0;
    nnp->screenInfo = NULL;

    isPeriodic();


    Q_hist = NULL;
    grow_arrays(atom->nmax);
    atom->add_callback(0);
    for (int i = 0; i < atom->nmax; i++)
        for (int j = 0; j < nprev; ++j)
            Q_hist[i][j] = 0;
}

/* ---------------------------------------------------------------------- */

FixNNP::~FixNNP()
{
    if (copymode) return;

    delete[] pertype_option;

    // unregister callbacks to this fix from Atom class

    atom->delete_callback(id,0);

    memory->destroy(Q_hist);
    memory->destroy(nnp->chi);
    memory->destroy(nnp->hardness);
    memory->destroy(nnp->sigma);

}

/* ---------------------------------------------------------------------- */

void FixNNP::post_constructor()
{
    pertype_parameters(pertype_option);

}

/* ---------------------------------------------------------------------- */

int FixNNP::setmask()
{
    int mask = 0;
    mask |= PRE_FORCE;
    mask |= PRE_FORCE_RESPA;
    mask |= MIN_PRE_FORCE;
    return mask;
}

/* ---------------------------------------------------------------------- */

void FixNNP::init()
{
    if (!atom->q_flag)
        error->all(FLERR,"Fix nnp requires atom attribute q");

    ngroup = group->count(igroup);
    if (ngroup == 0) error->all(FLERR,"Fix nnp group has no atoms");

    // need a half neighbor list w/ Newton off and ghost neighbors
    // built whenever re-neighboring occurs

    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->fix = 1;
    neighbor->requests[irequest]->newton = 2;
    neighbor->requests[irequest]->ghost = 1;

    // check for periodicity
    isPeriodic();

    //if (strstr(update->integrate_style,"respa"))
    //    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

// TODO: check this, it might be redundant in our case ?
void FixNNP::pertype_parameters(char *arg)
{
    if (strcmp(arg,"nnp") == 0) {
        nnpflag = 1;
        Pair *pair = force->pair_match("nnp",0);
        if (pair == NULL) error->all(FLERR,"No pair nnp for fix nnp");

        //int tmp;
        //TODO: we might extract these from the pair routine ?
        //nnp->chi = (double *) pair->extract("nnp->chi",tmp);
        //nnp->hardness = (double *) pair->extract("nnp->hardness",tmp);
        //nnp->sigma = (double *) pair->extract("nnp->sigma",tmp);
        //if (nnp->chi == NULL || nnp->hardness == NULL || nnp->sigma == NULL)
        //    error->all(FLERR,
        //               "Fix nnp could not extract params from pair nnp");
        return;
    }
}

/* ---------------------------------------------------------------------- */

// TODO: check these allocations and initializations
void FixNNP::allocate_QEq()
{
    int nloc = atom->nlocal;

    memory->create(nnp->chi,nloc+1,"qeq:nnp->chi");
    memory->create(nnp->hardness,nloc+1,"qeq:nnp->hardness");
    memory->create(nnp->sigma,nloc+1,"qeq:nnp->sigma");
    for (int i = 0; i < nloc; i++) {
        nnp->chi[i] = 0.0;
        nnp->hardness[i] = 0.0;
        nnp->sigma[i] = 0.0;
    }
    memory->create(nnp->screenInfo,5,"qeq:screening");


    // TODO: do we need these ?
    MPI_Bcast(&nnp->chi[1],nloc,MPI_DOUBLE,0,world);
    MPI_Bcast(&nnp->hardness[1],nloc,MPI_DOUBLE,0,world);
    MPI_Bcast(&nnp->sigma[1],nloc,MPI_DOUBLE,0,world);
}

/* ---------------------------------------------------------------------- */

void FixNNP::deallocate_QEq() {
    memory->destroy(nnp->chi);
    memory->destroy(nnp->hardness);
    memory->destroy(nnp->sigma);
    memory->destroy(nnp->screenInfo);
}

/* ---------------------------------------------------------------------- */

void FixNNP::init_list(int /*id*/, NeighList *ptr)
{
    list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixNNP::setup_pre_force(int vflag)
{

    deallocate_QEq();
    allocate_QEq();
    pre_force(vflag);

}

/* ---------------------------------------------------------------------- */

void FixNNP::calculate_electronegativities()
{
    // runs the first NN to get electronegativities
    run_network();

    // reads Qeq arrays and other required info from n2p2 into LAMMPS
    nnp->interface.getQEqParams(nnp->chi,nnp->hardness,nnp->sigma,qRef);
    nnp->interface.getScreeningInfo(nnp->screenInfo); //TODO: read function type ??

}

/* ---------------------------------------------------------------------- */

//TODO: update this
void FixNNP::run_network()
{
    if(nnp->interface.getNnpType() == 4)
    {
        // Set number of local atoms and add index and element.
        nnp->interface.setLocalAtoms(atom->nlocal,atom->tag,atom->type);

        // Transfer local neighbor list to NNP interface.
        nnp->transferNeighborList();

        // Run the first NN for electronegativities
        nnp->interface.process();
    }
    else //TODO
        error->all(FLERR,"Fix nnp cannot be used with a 2GHDNNP");
}

/* ---------------------------------------------------------------------- */

void FixNNP::min_setup_pre_force(int vflag)
{
    setup_pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixNNP::min_pre_force(int vflag)
{
    pre_force(vflag);
}

/* ---------------------------------------------------------------------- */

//TODO: main calculation routine, be careful with indices !
void FixNNP::pre_force(int /*vflag*/) {
    double t_start, t_end;
    double Qtot;

    if (update->ntimestep % nevery) return;
    if (comm->me == 0) t_start = MPI_Wtime();

    n = atom->nlocal;
    N = atom->nlocal + atom->nghost;

    // grow arrays if necessary
    // need to be atom->nmax in length

    //if (atom->nmax > nmax) reallocate_storage();
    //if (n > n_cap * DANGER_ZONE || m_fill > m_cap * DANGER_ZONE)
        //reallocate_matrix();

    //TODO: check the order of atoms
    // run the first NN here
    calculate_electronegativities();

    Qtot = 0.0;
    for (int i = 0; i < n; i++) {
        Qtot += atom->q[i];
    }
    qRef = Qtot; // total charge to be used in projection

    // Calculate atomic charges iteratively by minimizing the electrostatic energy
    calculate_QEqCharges();

    if (comm->me == 0) {
        t_end = MPI_Wtime();
        qeq_time = t_end - t_start;
    }
}

/* ---------------------------------------------------------------------- */
//TODO: parallelization + periodic option
double FixNNP::QEq_f(const gsl_vector *v)
{
    int i,j;

    int *type = atom->type;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;

    double dx, dy, dz, rij;
    double qi,qj;
    double **x = atom->x;

    double E_qeq,E_scr;
    double E_real,E_recip,E_self; // for periodic examples
    double iiterm,ijterm;

    //xall = new double[nall];
    //yall = new double[nall];
    //zall = new double[nall];
    //xall = yall = zall = NULL;
    //MPI_Allgather(&x[0],nlocal,MPI_DOUBLE,&xall,nall,MPI_DOUBLE,world);
    //MPI_Allgather(&x[1],nlocal,MPI_DOUBLE,&yall,nall,MPI_DOUBLE,world);
    //MPI_Allgather(&x[2],nlocal,MPI_DOUBLE,&zall,nall,MPI_DOUBLE,world);

    // TODO: indices & electrostatic energy
    E_qeq = 0.0;
    E_scr = 0.0;
    E_elec = 0.0;

    if (periodic)
    {
        std::cout << 31 << '\n';
        exit(0);
    }else
    {
        // first loop over local atoms
        for (i = 0; i < nlocal; i++) {
            qi = gsl_vector_get(v,i);
            // add i terms here
            iiterm = qi * qi / (2.0 * nnp->sigma[i] * sqrt(M_PI));
            E_qeq += iiterm + nnp->chi[i]*qi + 0.5*nnp->hardness[i]*qi*qi;
            E_elec += iiterm;
            E_scr -= iiterm;
            // second loop over 'all' atoms
            for (j = i + 1; j < nall; j++) {
                qj = gsl_vector_get(v, j);
                dx = x[j][0] - x[i][0];
                dy = x[j][1] - x[i][1];
                dz = x[j][2] - x[i][2];
                rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
                ijterm = (erf(rij / sqrt(2.0 * (pow(nnp->sigma[i], 2) + pow(nnp->sigma[j], 2)))) / rij);
                E_qeq += ijterm * qi * qj;
                E_elec += ijterm;
                if(rij <= nnp->screenInfo[2]) {
                    E_scr += ijterm * (nnp->screen_f(rij) - 1);
                }
            }
        }
    }





    E_elec = E_elec + E_scr; // electrostatic energy

    //MPI_Allreduce(E_elec, MPI_SUM...)

    return E_qeq;
}

/* ---------------------------------------------------------------------- */

double FixNNP::QEq_f_wrap(const gsl_vector *v, void *params)
{
    return static_cast<FixNNP*>(params)->QEq_f(v);
}

/* ---------------------------------------------------------------------- */

void FixNNP::QEq_df(const gsl_vector *v, gsl_vector *dEdQ)
{
    int i,j;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;

    double dx, dy, dz, rij;
    double qi,qj;
    double **x = atom->x;

    double val;
    double grad_sum;
    double grad_i;
    double local_sum;

    grad_sum = 0.0;
    // first loop over local atoms
    for (i = 0; i < nlocal; i++) { // TODO: indices
        qi = gsl_vector_get(v,i);
        local_sum = 0.0;
        // second loop over 'all' atoms
        for (j = 0; j < nall; j++) {
            if (j != i) {
                qj = gsl_vector_get(v, j);
                dx = x[j][0] - x[i][0];
                dy = x[j][1] - x[i][1];
                dz = x[j][2] - x[i][2];
                rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
                local_sum += qj * erf(rij / sqrt(2.0 * (pow(nnp->sigma[i], 2) + pow(nnp->sigma[j], 2)))) / rij;
            }
        }
        val = nnp->chi[i] + nnp->hardness[i]*qi + qi/(nnp->sigma[i]*sqrt(M_PI)) + local_sum;
        grad_sum = grad_sum + val;
        gsl_vector_set(dEdQ,i,val);
    }

    // Gradient projection //TODO: communication ?
    for (i = 0; i < nall; i++){
        grad_i = gsl_vector_get(dEdQ,i);
        gsl_vector_set(dEdQ,i,grad_i - (grad_sum)/nall);
    }

    //MPI_Allreduce(dEdQ, MPI_SUM...)
}

/* ---------------------------------------------------------------------- */

void FixNNP::QEq_df_wrap(const gsl_vector *v, void *params, gsl_vector *df)
{
    static_cast<FixNNP*>(params)->QEq_df(v, df);
}

/* ---------------------------------------------------------------------- */

void FixNNP::QEq_fdf(const gsl_vector *v, double *f, gsl_vector *df)
{
    *f = QEq_f(v);
    QEq_df(v, df);
}

/* ---------------------------------------------------------------------- */

void FixNNP::QEq_fdf_wrap(const gsl_vector *v, void *params, double *E, gsl_vector *df)
{
    static_cast<FixNNP*>(params)->QEq_fdf(v, E, df);
}

/* ---------------------------------------------------------------------- */

void FixNNP::calculate_QEqCharges()
{
    size_t iter = 0;
    int status;
    int i,j;
    int nsize;
    int maxit;

    double *q = atom->q;
    double qsum_it;
    double gradsum;
    double qi;

    double grad_tol,min_tol;
    double step;

    double df,alpha;


    // TODO: backward/forward communication ??

    nsize = atom->natoms;

    gsl_vector *x; // charge vector in our case

    QEq_minimizer.n = nsize; // it should be n_all in the future
    QEq_minimizer.f = &QEq_f_wrap; // function pointer f(x)
    QEq_minimizer.df = &QEq_df_wrap; // function pointer df(x)
    QEq_minimizer.fdf = &QEq_fdf_wrap;
    QEq_minimizer.params = this;

    // Starting point is the current charge vector ???
    x = gsl_vector_alloc(nsize); // +1 for LM
    for (i = 0; i < nsize; i++) {
        gsl_vector_set(x,i,q[i]);
    }

    //T = gsl_multimin_fdfminimizer_conjugate_fr; // minimization algorithm
    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T, nsize);

    // Minimizer Params TODO: user-defined ?
    grad_tol = 1e-5;
    min_tol = 1e-7;
    step = 1e-2;
    maxit = 100;

    gsl_multimin_fdfminimizer_set(s, &QEq_minimizer, x, step, min_tol); // tol = 0 might be expensive ???
    do
    {
        iter++;
        qsum_it = 0.0;
        gradsum = 0.0;

        std::cout << "iter : " << iter << '\n';
        std::cout << "E_qeq: " << s->f << '\n';
        std::cout << "-------------" << '\n';

        status = gsl_multimin_fdfminimizer_iterate(s);

        // Projection
        for(i = 0; i < nsize; i++) {
            qsum_it = qsum_it + gsl_vector_get(s->x, i);
            gradsum = gradsum + gsl_vector_get(s->gradient,i);
        }

        for(i = 0; i < nsize; i++) {
            qi = gsl_vector_get(s->x,i);
            gsl_vector_set(s->x,i, qi - (qsum_it-qRef)/nsize); // charge projection
        }

        //if (status)
        //    break;
        status = gsl_multimin_test_gradient(s->gradient, grad_tol);

        if (status == GSL_SUCCESS)
            printf ("Minimum found\n");

    }
    while (status == GSL_CONTINUE && iter < maxit);

    // read charges before deallocating x - be careful with indices !
    for (i = 0; i < nsize; i++) {
        q[i] = gsl_vector_get(s->x,i);
    }

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);
}

/* ---------------------------------------------------------------------- */

void FixNNP::isPeriodic()
{
    if (domain->nonperiodic == 0) periodic = true;
    else                          periodic = false;
}


//// BELOW ARE FIX COMMUNICATION ROUTINES
/* ---------------------------------------------------------------------- */

int FixNNP::pack_forward_comm(int n, int *list, double *buf,
                                  int /*pbc_flag*/, int * /*pbc*/)
{
    int m;

    if (pack_flag == 1)
        for(m = 0; m < n; m++) buf[m] = d[list[m]];
    else if (pack_flag == 2)
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

void FixNNP::unpack_forward_comm(int n, int first, double *buf)
{
    int i, m;

    if (pack_flag == 1)
        for(m = 0, i = first; m < n; m++, i++) d[i] = buf[m];
    else if (pack_flag == 2)
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

int FixNNP::pack_reverse_comm(int n, int first, double *buf)
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

void FixNNP::unpack_reverse_comm(int n, int *list, double *buf)
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

double FixNNP::memory_usage()
{
    double bytes;

    bytes = atom->nmax*2 * sizeof(double); // Q_hist
    bytes += atom->nmax*11 * sizeof(double); // storage
    bytes += n_cap*2 * sizeof(int); // matrix...
    bytes += m_cap * sizeof(int);
    bytes += m_cap * sizeof(double);

    return bytes;
}

/* ----------------------------------------------------------------------
   allocate fictitious charge arrays
------------------------------------------------------------------------- */

void FixNNP::grow_arrays(int nmax)
{
    memory->grow(Q_hist,nmax,nprev,"qeq:Q_hist");
}

/* ----------------------------------------------------------------------
   copy values within fictitious charge arrays
------------------------------------------------------------------------- */

void FixNNP::copy_arrays(int i, int j, int /*delflag*/)
{
    for (int m = 0; m < nprev; m++) {
        Q_hist[j][m] = Q_hist[i][m];
    }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

// TODO: be careful with this one, it returns something
int FixNNP::pack_exchange(int i, double *buf)
{
    for (int m = 0; m < nprev; m++) buf[m] = Q_hist[i][m];
    return nprev;
    //for (int m = 0; m < nprev; m++) buf[nprev+m] = t_hist[i][m];
    //return nprev*2;

}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int FixNNP::unpack_exchange(int nlocal, double *buf)
{
    for (int m = 0; m < nprev; m++) Q_hist[nlocal][m] = buf[m];
    return nprev;
    //for (int m = 0; m < nprev; m++) t_hist[nlocal][m] = buf[nprev+m];
    //return nprev*2;
}

/* ---------------------------------------------------------------------- */

double FixNNP::parallel_norm( double *v, int n)
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

double FixNNP::parallel_dot( double *v1, double *v2, int n)
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

double FixNNP::parallel_vector_acc( double *v, int n)
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

void FixNNP::vector_sum( double* dest, double c, double* v,
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

void FixNNP::vector_add( double* dest, double c, double* v, int k)
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





