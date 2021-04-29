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
#include "cg.h"  // for the CG library



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

    nevery = force->inumeric(FLERR,arg[3]);
    if (nevery <= 0) error->all(FLERR,"Illegal fix nnp command");

    dum1 = force->numeric(FLERR,arg[4]);
    dum2 = force->numeric(FLERR,arg[5]);

    tolerance = force->numeric(FLERR,arg[6]);
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

    chi = NULL;
    hardness = NULL;
    sigma = NULL;

    Adia_inv = NULL;
    b = NULL;
    b_der = NULL;
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
    A.val2d = NULL;
    A.dvalq = NULL;

    comm_forward = comm_reverse = 1;

    E_elec = 0.0;
    rscr = NULL;

    // perform initial allocation of atom-based arrays
    // register with Atom class

    nnp = NULL;
    nnp = (PairNNP *) force->pair_match("^nnp",0);

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

    deallocate_storage();
    deallocate_matrix();

    memory->destroy(chi);
    memory->destroy(hardness);
    memory->destroy(sigma);

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

// TODO: check this, it might be redundant in our case ?
void FixNNP::pertype_parameters(char *arg)
{
    if (strcmp(arg,"nnp") == 0) {
        nnpflag = 1;
        Pair *pair = force->pair_match("nnp",0);
        if (pair == NULL) error->all(FLERR,"No pair nnp for fix nnp");

        //int tmp;
        //TODO: we might extract these from the pair routine ?
        //chi = (double *) pair->extract("chi",tmp);
        //hardness = (double *) pair->extract("hardness",tmp);
        //sigma = (double *) pair->extract("sigma",tmp);
        //if (chi == NULL || hardness == NULL || sigma == NULL)
        //    error->all(FLERR,
        //               "Fix nnp could not extract params from pair nnp");
        return;
    }
}

// TODO: check these allocations and initializations
void FixNNP::allocate_qeq()
{
    int nloc = atom->nlocal;

    memory->create(chi,nloc+1,"qeq:chi");
    memory->create(hardness,nloc+1,"qeq:hardness");
    memory->create(sigma,nloc+1,"qeq:sigma");
    for (int i = 0; i < nloc; i++) {
        chi[i] = 0.0;
        hardness[i] = 0.0;
        sigma[i] = 0.0;
    }
    memory->create(rscr,10,"qeq:screening");


    // TODO: do we need these ?
    MPI_Bcast(&chi[1],nloc,MPI_DOUBLE,0,world);
    MPI_Bcast(&hardness[1],nloc,MPI_DOUBLE,0,world);
    MPI_Bcast(&sigma[1],nloc,MPI_DOUBLE,0,world);
}

void FixNNP::allocate_storage()
{
    nmax = atom->nmax;
    int n = atom->ntypes;
    int nmax = atom->nmax;

    //TODO: derivative arrays to be added ?
    //TODO: check the sizes again !

    memory->create(Q,nmax,"qeq:Q");

    memory->create(Adia_inv,nmax,"qeq:Adia_inv");
    memory->create(b,nmax,"qeq:b");
    memory->create(b_der,3*nmax,"qeq:b_der");
    memory->create(b_prc,nmax,"qeq:b_prc");
    memory->create(b_prm,nmax,"qeq:b_prm");

    memory->create(p,nmax,"qeq:p");
    memory->create(q,nmax,"qeq:q");
    memory->create(r,nmax,"qeq:r");
    memory->create(d,nmax,"qeq:d");

}

void FixNNP::deallocate_qeq()
{
    memory->destroy(chi);
    memory->destroy(hardness);
    memory->destroy(sigma);
    memory->destroy(rscr);
}

void FixNNP::deallocate_storage()
{
    memory->destroy(Q);

    memory->destroy( Adia_inv );
    memory->destroy( b );
    memory->destroy( b_der );
    memory->destroy( b_prc );
    memory->destroy( b_prm );

    memory->destroy( p );
    memory->destroy( q );
    memory->destroy( r );
    memory->destroy( d );
}

void FixNNP::reallocate_storage()
{
    deallocate_storage();
    allocate_storage();
    init_storage();
}

void FixNNP::allocate_matrix()
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

    //TODO: this way of constructing matrix A might be specific to this method, we have something else !!
    A.n = n_cap;
    A.m = m_cap;
    memory->create(A.firstnbr,n_cap,"qeq:A.firstnbr");
    memory->create(A.numnbrs,n_cap,"qeq:A.numnbrs");
    memory->create(A.jlist,m_cap,"qeq:A.jlist");
    memory->create(A.val,m_cap,"qeq:A.val");

    memory->create(A.val2d,n+1,n+1,"qeq:A.val2d");

    memory->create(A.dvalq,n+1,n+1,3,"qeq:A.dvalq");
    for (ii = 0; ii < n+1; ii++) {
        A.dvalq[ii][ii][0] = 0.0;
        A.dvalq[ii][ii][1] = 0.0;
        A.dvalq[ii][ii][2] = 0.0;
    }
}

void FixNNP::deallocate_matrix()
{
    memory->destroy( A.firstnbr );
    memory->destroy( A.numnbrs );
    memory->destroy( A.jlist );
    memory->destroy( A.val );

    memory->destroy( A.val2d);
    memory->destroy( A.dvalq );
}

void FixNNP::reallocate_matrix()
{
    deallocate_matrix();
    allocate_matrix();
}

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

void FixNNP::init_list(int /*id*/, NeighList *ptr)
{
    list = ptr;
}

void FixNNP::setup_pre_force(int vflag)
{

    deallocate_storage();
    allocate_storage();

    setup_qeq();

    init_storage();
    deallocate_matrix();
    allocate_matrix();

    pre_force(vflag);

}

void FixNNP::setup_qeq()
{
    // allocates chi,hardness and sigma arrays
    deallocate_qeq();
    allocate_qeq();

}

void FixNNP::pre_force_qeq()
{
    // runs the first NN to get electronegativities
    run_network();

    // reads Qeq arrays and other required info from n2p2 into LAMMPS
    nnp->interface.getQeqParams(chi,hardness,sigma);
    qref = nnp->interface.getTotalCharge();
    nnp->interface.getScreeningInfo(rscr); //TODO: read function type ??

}

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
    else
        error->all(FLERR,"Fix nnp cannot be used with a 2GHDNNP");
}

void FixNNP::min_setup_pre_force(int vflag)
{
    setup_pre_force(vflag);
}

void FixNNP::min_pre_force(int vflag)
{
    pre_force(vflag);
}

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

    if (atom->nmax > nmax) reallocate_storage();
    if (n > n_cap * DANGER_ZONE || m_fill > m_cap * DANGER_ZONE)
        reallocate_matrix();

    // run the first NN here
    pre_force_qeq();

    Qtot = 0.0;
    for (int i = 0; i < n; i++) {
        Qtot += atom->q[i];
        std::cout << "Q[i] : " << atom->q[i] << '\n';
    }
    qref = Qtot;

    std::cout << "Total Charge:" << Qtot << '\n';

    // calculate atomic charges by minimizing the electrostatic energy
    calculate_QEqCharges();




    if (comm->me == 0) {
        t_end = MPI_Wtime();
        qeq_time = t_end - t_start;
    }
}

double FixNNP::QEq_energy(const gsl_vector *v)
{
    int i,j;

    int *type = atom->type;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;

    double dx, dy, dz, rij;
    double qi,qj;
    double **x = atom->x;

    double E_qeq;
    double E_scr;
    double iiterm,ijterm;

    xall = new double[nall];
    yall = new double[nall];
    zall = new double[nall];

    xall = yall = zall = NULL;

    //MPI_Allgather(&x[0],nlocal,MPI_DOUBLE,&xall,nall,MPI_DOUBLE,world);
    //MPI_Allgather(&x[1],nlocal,MPI_DOUBLE,&yall,nall,MPI_DOUBLE,world);
    //MPI_Allgather(&x[2],nlocal,MPI_DOUBLE,&zall,nall,MPI_DOUBLE,world);

    // TODO: indices
    E_qeq = 0.0;
    E_scr = 0.0;
    E_elec = 0.0;
    // first loop over local atoms
    for (i = 0; i < nlocal; i++) {
        qi = gsl_vector_get(v,i);
        // add i terms here
        iiterm = qi * qi / (2.0 * sigma[i] * sqrt(M_PI));
        E_qeq += iiterm + chi[i]*qi + 0.5*hardness[i]*qi*qi;
        E_elec += iiterm;
        E_scr -= iiterm;
        // second loop over 'all' atoms
        for (j = i + 1; j < nall; j++) {
            qj = gsl_vector_get(v, j);
            dx = x[j][0] - x[i][0];
            dy = x[j][1] - x[i][1];
            dz = x[j][2] - x[i][2];
            rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
            ijterm = qi * qj * (erf(rij / sqrt(2.0 * (pow(sigma[i], 2) + pow(sigma[j], 2)))) / rij);
            E_qeq += ijterm;
            E_elec += ijterm;
            if(rij <= rscr[1]) {
                E_scr += ijterm * (screen_f(rij) - 1);
            }
        }
    }

    E_elec = E_elec + E_scr; // electrostatic energy

    //MPI_Allreduce(E_elec, MPI_SUM...)

    return E_qeq;
}

double FixNNP::QEq_energy_wrap(const gsl_vector *v, void *params)
{
    return static_cast<FixNNP*>(params)->QEq_energy(v);
}

void FixNNP::dEdQ(const gsl_vector *v, gsl_vector *dEdQ)
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
                local_sum += qj * erf(rij / sqrt(2.0 * (pow(sigma[i], 2) + pow(sigma[j], 2)))) / rij;
            }
        }
        val = chi[i] + hardness[i]*qi + qi/(sigma[i]*sqrt(M_PI)) + local_sum;
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

void FixNNP::dEdQ_wrap(const gsl_vector *v, void *params, gsl_vector *df)
{
    static_cast<FixNNP*>(params)->dEdQ(v, df);
}

void FixNNP::EdEdQ(const gsl_vector *v, double *f, gsl_vector *df)
{
    *f = QEq_energy(v);
    dEdQ(v, df);
}

void FixNNP::EdEdQ_wrap(const gsl_vector *v, void *params, double *E, gsl_vector *df)
{
    static_cast<FixNNP*>(params)->EdEdQ(v, E, df);
}


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

    QEq_fdf.n = nsize; // it should be n_all in the future
    QEq_fdf.f = &QEq_energy_wrap; // function pointer f(x)
    QEq_fdf.df = &dEdQ_wrap; // function pointer df(x)
    QEq_fdf.fdf = &EdEdQ_wrap;
    QEq_fdf.params = this;

    // Starting point is the current charge vector ???
    x = gsl_vector_alloc(nsize); // +1 for LM
    for (i = 0; i < nsize; i++) {
        gsl_vector_set(x,i,q[i]);
    }

    T = gsl_multimin_fdfminimizer_conjugate_fr; // minimization algorithm
    //T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T, nsize);

    // Minimizer Params TODO: user-defined ?
    grad_tol = 1e-5;
    min_tol = 1e-7;
    step = 1e-2;
    maxit = 100;

    gsl_multimin_fdfminimizer_set(s, &QEq_fdf, x, step, min_tol); // tol = 0 might be expensive ???
    do
    {
        iter++;
        qsum_it = 0.0;
        gradsum = 0.0;

        std::cout << "iter : " << iter << '\n';
        std::cout << "E_qeq: " << s->f << '\n';
        std::cout << "E_elec: " << E_elec << '\n';
        std::cout << "-------------" << '\n';

        status = gsl_multimin_fdfminimizer_iterate(s);

        // Projection
        for(i = 0; i < nsize; i++) {
            qsum_it = qsum_it + gsl_vector_get(s->x, i);
            gradsum = gradsum + gsl_vector_get(s->gradient,i);
        }

        for(i = 0; i < nsize; i++) {
            qi = gsl_vector_get(s->x,i);
            gsl_vector_set(s->x,i, qi - (qsum_it-qref)/nsize); // charge projection
        }

        //if (status)
        //    break;
        status = gsl_multimin_test_gradient(s->gradient, grad_tol);

        if (status == GSL_SUCCESS)
            printf ("Minimum found at:\n");

    }
    while (status == GSL_CONTINUE && iter < maxit);

    // read charges before deallocating x - be careful with indices !
    for (i = 0; i < nsize; i++) {
        q[i] = gsl_vector_get(s->x,i);
    }

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);
}


double FixNNP::fLambda_f(const gsl_vector *v)
{
    int i,j;

    int *type = atom->type;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;

    double dx, dy, dz, rij;
    double qi,qj;
    double **x = atom->x;

    double E_qeq;
    double E_scr;
    double iiterm,ijterm;

    xall = new double[nall];
    yall = new double[nall];
    zall = new double[nall];

    xall = yall = zall = NULL;

    //MPI_Allgather(&x[0],nlocal,MPI_DOUBLE,&xall,nall,MPI_DOUBLE,world);
    //MPI_Allgather(&x[1],nlocal,MPI_DOUBLE,&yall,nall,MPI_DOUBLE,world);
    //MPI_Allgather(&x[2],nlocal,MPI_DOUBLE,&zall,nall,MPI_DOUBLE,world);

    // TODO: indices
    E_qeq = 0.0;
    E_scr = 0.0;
    E_elec = 0.0;
    // first loop over local atoms
    for (i = 0; i < nlocal; i++) {
        qi = gsl_vector_get(v,i);
        // add i terms here
        iiterm = qi * qi / (2.0 * sigma[i] * sqrt(M_PI));
        E_qeq += iiterm + chi[i]*qi + 0.5*hardness[i]*qi*qi;
        E_elec += iiterm;
        E_scr -= iiterm;
        // second loop over 'all' atoms
        for (j = i + 1; j < nall; j++) {
            qj = gsl_vector_get(v, j);
            dx = x[j][0] - x[i][0];
            dy = x[j][1] - x[i][1];
            dz = x[j][2] - x[i][2];
            rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
            ijterm = qi * qj * (erf(rij / sqrt(2.0 * (pow(sigma[i], 2) + pow(sigma[j], 2)))) / rij);
            E_qeq += ijterm;
            E_elec += ijterm;
            if(rij <= rscr[1]) {
                E_scr += ijterm * (screen_f(rij) - 1);
            }
        }
    }

    E_elec = E_elec + E_scr; // electrostatic energy

    //MPI_Allreduce(E_elec, MPI_SUM...)

    return E_qeq;
}

double FixNNP::fLambda_f_wrap(const gsl_vector *v, void *params)
{
    return static_cast<FixNNP*>(params)->fLambda_f(v);
}

void FixNNP::fLambda_df(const gsl_vector *v, gsl_vector *dEdQ)
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
                local_sum += qj * erf(rij / sqrt(2.0 * (pow(sigma[i], 2) + pow(sigma[j], 2)))) / rij;
            }
        }
        val = chi[i] + hardness[i]*qi + qi/(sigma[i]*sqrt(M_PI)) + local_sum;
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

void FixNNP::fLambda_df_wrap(const gsl_vector *v, void *params, gsl_vector *df)
{
    static_cast<FixNNP*>(params)->fLambda_df(v, df);
}

void FixNNP::fLambda_fdf(const gsl_vector *v, double *f, gsl_vector *df)
{
    *f = QEq_energy(v);
    dEdQ(v, df);
}

void FixNNP::fLambda_fdf_wrap(const gsl_vector *v, void *params, double *f, gsl_vector *df)
{
    static_cast<FixNNP*>(params)->fLambda_fdf(v, f, df);
}

void FixNNP::calculate_fLambda()
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

    QEq_fdf.n = nsize; // it should be n_all in the future
    QEq_fdf.f = &QEq_energy_wrap; // function pointer f(x)
    QEq_fdf.df = &dEdQ_wrap; // function pointer df(x)
    QEq_fdf.fdf = &EdEdQ_wrap;
    QEq_fdf.params = this;

    // Starting point is the current charge vector ???
    x = gsl_vector_alloc(nsize); // +1 for LM
    for (i = 0; i < nsize; i++) {
        gsl_vector_set(x,i,q[i]);
    }

    T = gsl_multimin_fdfminimizer_conjugate_fr; // minimization algorithm
    //T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T, nsize);

    // Minimizer Params TODO: user-defined ?
    grad_tol = 1e-5;
    min_tol = 1e-7;
    step = 1e-2;
    maxit = 100;

    gsl_multimin_fdfminimizer_set(s, &QEq_fdf, x, step, min_tol); // tol = 0 might be expensive ???
    do
    {
        iter++;
        qsum_it = 0.0;
        gradsum = 0.0;

        std::cout << "iter : " << iter << '\n';
        std::cout << "E_qeq: " << s->f << '\n';
        std::cout << "E_elec: " << E_elec << '\n';
        std::cout << "-------------" << '\n';

        status = gsl_multimin_fdfminimizer_iterate(s);

        // Projection
        for(i = 0; i < nsize; i++) {
            qsum_it = qsum_it + gsl_vector_get(s->x, i);
            gradsum = gradsum + gsl_vector_get(s->gradient,i);
        }

        for(i = 0; i < nsize; i++) {
            qi = gsl_vector_get(s->x,i);
            gsl_vector_set(s->x,i, qi - (qsum_it-qref)/nsize); // charge projection
        }

        //if (status)
        //    break;
        status = gsl_multimin_test_gradient(s->gradient, grad_tol);

        if (status == GSL_SUCCESS)
            printf ("Minimum found at:\n");

    }
    while (status == GSL_CONTINUE && iter < maxit);

    // read charges before deallocating x - be careful with indices !
    for (i = 0; i < nsize; i++) {
        q[i] = gsl_vector_get(s->x,i);
    }

    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(x);
}


double FixNNP::screen_f(double r)
{
    double x;

    if (r >= rscr[1]) return 1.0;
    else if (r <= rscr[0]) return 0.0;
    else // TODO: cleanup (only cos function for now)
    {
        x = (r-rscr[0])*rscr[2];
        return 1.0 - 0.5*(cos(M_PI*x)+1);
    }

}

double FixNNP::screen_df(double r)
{

}
//// BELOW ARE THE ROUTINES FOR MATRIX APPROACH, DEPRECATED AT THIS POINT !!!!
//TODO: check indexing here, and also think about LM
void FixNNP::init_storage()
{
    int NN,nn;

    NN = list->inum + list->gnum; // ????
    nn = list->inum;

    for (int i = 0; i < nn; i++) {
        Adia_inv[i] = 1. / (hardness[atom->type[i]] + 1. / (sigma[atom->type[i]] * sqrt(M_PI)));
        //b_prc[i] = 0;
        //b_prm[i] = 0;
        Q[i] = 0;
    }
    Q[nn] = 0; // lm
    Adia_inv[nn] = 0; // lm

}

void FixNNP::init_matvec()
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

            /* init pre-conditioner for H and init solution vectors */
            //Adia_inv[i] = 1. / hardness[ atom->type[i] ];
            b[i]      = -chi[ atom->tag[i] ]; //TODO:index ??
            r[i] = 0;
            q[i] = 0;
            d[i] = 0;
            p[i] = 0;

            /* quadratic extrapolation from previous solutions */
            Q[i] = Q_hist[i][2] + 3 * ( Q_hist[i][0] - Q_hist[i][1]);
            //Q[i] = 0;
        }
    }
    b[nn] = 0.0; // LM
    Q[nn] = 0.0;
    Adia_inv[nn] = 0.0; // LM

    pack_flag = 2;
    comm->forward_comm_fix(this); //Dist_vector( Q );
}

// TODO: this is where the matrix A is FILLED, needs to be edited according to our formalism
void FixNNP::compute_A()
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
    m_fill = 0;
    r_sqr = 0;
    // TODO: my own loop (substantially changed older version is in fix_qeq_reax.cpp)
    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        if (mask[i] & groupbit) {
            A.firstnbr[i] = m_fill;
            for (jj = 0; jj < inum; jj++) {
                j = ilist[jj];
                dx = x[j][0] - x[i][0];
                dy = x[j][1] - x[i][1];
                dz = x[j][2] - x[i][2];
                r_sqr = SQR(dx) + SQR(dy) + SQR(dz);
                flag = 1; // to be removed
                if (flag) {
                    A.jlist[m_fill] = j;
                    A.val[m_fill] = calculate_A(i,j, sqrt(r_sqr));
                    A.val2d[i][j] = calculate_A(i,j, sqrt(r_sqr));
                    //std::cout << A.val[m_fill] << '\n';
                    m_fill++;
                }
            }
            A.val[m_fill] = 1.0; // LM
            A.val2d[i][j+1] = 1.0; // LM
            m_fill++;
            A.numnbrs[i] = m_fill - A.firstnbr[i];
        }
    }
    // LM
    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        A.val[m_fill] = 1.0;
        A.val2d[inum][i] = 1.0;
        m_fill++;
    }
    A.val[m_fill] = 0.0;
    A.val2d[inum][inum] = 0.0;
    m_fill++;

    if (m_fill >= A.m) {
        char str[128];
        sprintf(str,"A matrix size has been exceeded: m_fill=%d A.m=%d\n",
                m_fill, A.m);
        error->warning(FLERR,str);
        error->all(FLERR,"Fix nnp has insufficient QEq matrix size");
    }
}

// TODO: this is where elements of matrix A is CALCULATED, do the periodicity check here ??
double FixNNP::calculate_A(int i, int j,double r)
{
    double nom, denom, res;
    int *type = atom->type;
    tagint *tag = atom->tag;

    if (!periodic) //non-periodic A matrix (for now!!)
    {
        if (i == j) // diagonal elements
        {
            res = hardness[type[i]] + 1. / (sigma[type[i]] * sqrt(M_PI));
        } else  //non-diagonal elements
        {
            nom =  erf(r / sqrt(2.0 * (sigma[type[i]] * sigma[type[i]] + sigma[type[j]] * sigma[type[j]])));
            res = nom / r;
        }
    } else
    {

    }
    return res;
}

// TODO: PCG algorithm, to be adjusted according to our theory
int FixNNP::CG( double *b, double *x)
{
    int  i, j, imax;
    double tmp, alpha, beta, b_norm;
    double sig_old, sig_new;

    int nn, jj;
    int *ilist;

    nn = list->inum;
    nn = nn + 1; // LM (TODO: check)
    ilist = list->ilist;

    imax = 200;

    pack_flag = 1;
    sparse_matvec( &A, x, q);
    comm->reverse_comm_fix(this); //Coll_Vector( q );

    vector_sum( r , 1.,  b, -1., q, nn);

    // TODO: do (should) we have pre-conditioning as well ?
    for (jj = 0; jj < nn; ++jj) {
        j = ilist[jj];
        if (atom->mask[j] & groupbit)
            //d[j] = r[j] * Adia_inv[j]; //pre-condition
            d[j] = r[j] * 1.0;
    }

    b_norm = parallel_norm( b, nn);
    sig_new = parallel_dot( r, d, nn);

    for (i = 1; i < imax && sqrt(sig_new) / b_norm > tolerance; ++i) {
        comm->forward_comm_fix(this); //Dist_vector( d );
        sparse_matvec( &A, d, q );
        comm->reverse_comm_fix(this); //Coll_vector( q );

        tmp = parallel_dot( d, q, nn);
        alpha = sig_new / tmp;

        vector_add( x, alpha, d, nn);
        vector_add( r, -alpha, q, nn);

        // pre-conditioning (TODO: check me..)
        for (jj = 0; jj < nn; ++jj) {
            j = ilist[jj];
            if (atom->mask[j] & groupbit)
                //p[j] = r[j] * Adia_inv[j];
                p[j] = r[j] * 1.0;
        }

        sig_old = sig_new;
        sig_new = parallel_dot( r, p, nn);

        beta = sig_new / sig_old;
        vector_sum( d, 1., p, beta, d, nn);

        //std::cout << i << '\n';
    }

    if (i >= imax && comm->me == 0) {
        char str[128];
        sprintf(str,"Fix nnp CG convergence failed after %d iterations "
                    "at " BIGINT_FORMAT " step",i,update->ntimestep);
        error->warning(FLERR,str);
    }

    return i;
}

// TODO: this is where matrix algebra for CG is done, to be edited
void FixNNP::sparse_matvec( sparse_matrix *A, double *x, double *b)
{
    int i, j, itr_j;
    int nn, NN, ii, jj;
    int *ilist;

    nn = list->inum;
    nn = nn + 1; // LM
    NN = list->inum + list->gnum;
    ilist = list->ilist;

    // TODO: check this, why only hardness here, diagonal terms ???
    //for (ii = 0; ii < nn; ++ii) {
    //    i = ilist[ii];
    //    if (atom->mask[i] & groupbit)
    //        b[i] = hardness[ atom->type[i] ] * x[i];
    //}

    // TODO: distant neighbors ???
    //for (ii = nn; ii < NN; ++ii) {
    //    i = ilist[ii];
    //    if (atom->mask[i] & groupbit)
    //        b[i] = 0;
    //}

    // TODO: probably non-diagonal terms are handled in here
    //for (ii = 0; ii < nn; ++ii) {
    //    i = ilist[ii];
    //    if (atom->mask[i] & groupbit) {
    //        for (itr_j=A->firstnbr[i]; itr_j<A->firstnbr[i]+A->numnbrs[i]; itr_j++) {
    //            std::cout << itr_j << '\n';
    //            j = A->jlist[itr_j];
    //            b[i] += A->val[itr_j] * x[j];
    //            b[j] += A->val[itr_j] * x[i];
    //        }
    //    }
    //}

    for (ii = 0; ii < nn; ii++) {
        //i = ilist[ii];
        //if (atom->mask[i] & groupbit) {
            for (jj = 0; jj < nn; jj++) {
                //j = ilist[jj];
                b[ii] += A->val2d[ii][jj] * x[jj];
            }
        //}
    }
    //exit(0);

}

void FixNNP::calculate_Q()
{
    int i, k;
    double u, Q_sum;
    double *q = atom->q;

    int nn, ii;
    int *ilist;

    nn = list->inum;
    ilist = list->ilist;

    // TODO: be careful with indexing and LM here
    Q_sum = parallel_vector_acc( Q, nn);

    //std::cout << Q_sum << '\n';

    for (ii = 0; ii < nn; ++ii) {
        i = ilist[ii];
        if (atom->mask[i] & groupbit) {
            q[i] = Q[i];

            /* backup */
            for (k = nprev-1; k > 0; --k) {
                Q_hist[i][k] = Q_hist[i][k-1];
            }
            Q_hist[i][0] = Q[i];
        }
    }

    pack_flag = 4;
    comm->forward_comm_fix(this); //Dist_vector( atom->q );
}

void FixNNP::compute_dAdxyzQ()
{
    int inum, jnum, *ilist, *jlist, *numneigh, **firstneigh;
    int i, j, ii, jj, flag;
    double dx, dy, dz, rij;
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
    m_fill = 0;
    // TODO: my own loop (substantially changed older version is in fix_qeq_reax.cpp)
    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        if (mask[i] & groupbit) {
            for (jj = ii+1; jj < inum; jj++) {
                j = ilist[jj];
                dx = x[j][0] - x[i][0];
                dy = x[j][1] - x[i][1];
                dz = x[j][2] - x[i][2];
                rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));

                A.dvalq[i][i][0] = A.dvalq[i][i][0] + calculate_dAdxyzQ(dx,rij,i,j) * atom->q[tag[j]]; // check ind
                A.dvalq[i][i][1] = A.dvalq[i][i][1] + calculate_dAdxyzQ(dy,rij,i,j) * atom->q[tag[j]];
                A.dvalq[i][i][2] = A.dvalq[i][i][2] + calculate_dAdxyzQ(dz,rij,i,j) * atom->q[tag[j]];

                A.dvalq[j][j][0] = A.dvalq[j][j][0] - calculate_dAdxyzQ(dx,rij,i,j) * atom->q[tag[i]];
                A.dvalq[j][j][1] = A.dvalq[j][j][1] - calculate_dAdxyzQ(dy,rij,i,j) * atom->q[tag[i]];
                A.dvalq[j][j][2] = A.dvalq[j][j][2] - calculate_dAdxyzQ(dz,rij,i,j) * atom->q[tag[i]];

                A.dvalq[i][j][0] = -calculate_dAdxyzQ(dx,rij,i,j);
                A.dvalq[i][j][1] = -calculate_dAdxyzQ(dy,rij,i,j);
                A.dvalq[i][j][2] = -calculate_dAdxyzQ(dz,rij,i,j);

                A.dvalq[j][i][0] = calculate_dAdxyzQ(dx,rij,i,j);
                A.dvalq[j][i][1] = calculate_dAdxyzQ(dy,rij,i,j);
                A.dvalq[j][i][2] = calculate_dAdxyzQ(dz,rij,i,j);
            }
        }
    }
}

double FixNNP::calculate_dAdxyzQ(double dx, double r, int i, int j)

{
    double gamma, tg, fi, res;
    int *type = atom->type;

    if (!periodic)
    {
        gamma = (sigma[type[i]] * sigma[type[i]] + sigma[type[j]] * sigma[type[j]]);
        tg = 1. / (sqrt(2.0) * gamma);
        fi = (2.0 * tg * exp(-tg*tg * r*r) / (sqrt(M_PI) * r) - erf(tg*r)/(r*r));

        res = dx / r * fi ;
    } else
    {
    }
    return res;
}

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

    bytes = atom->nmax*nprev*2 * sizeof(double); // Q_hist
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

void FixNNP::isPeriodic()
{
    if (domain->nonperiodic == 0) periodic = true;
    else                          periodic = false;
}

///////////OLD////////////
void FixNNP::QEqSerial()
{
    double t_start, t_end;
    double sum;

    n = atom->nlocal;
    N = atom->nlocal + atom->nghost;

    // grow arrays if necessary
    // need to be atom->nmax in length

    //if (atom->nmax > nmax) reallocate_storage();
    if (n > n_cap*DANGER_ZONE || m_fill > m_cap*DANGER_ZONE)
        reallocate_matrix();

    init_matvec();
    //matvecs = CGSerial(b, Q);

    // external C library (serial)
    r8ge_cg(n+1,A.val,b,Q);
    r8ge_cg(n+1,A.val,b,Q);

    // TODO: be careful with indexing here
    sum = 0.0;
    for (int i=0; i<n; i++)
    {
        atom->q[i] = Q[i];
    }

}


