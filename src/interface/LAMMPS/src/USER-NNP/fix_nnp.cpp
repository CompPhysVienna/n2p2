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
#include "memory.h"
#include "error.h"
#include "utils.h"
#include <chrono> //time

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std::chrono;
using namespace nnp;

#define EV_TO_KCAL_PER_MOL 14.4
#define SQR(x) ((x)*(x))
#define CUBE(x) ((x)*(x)*(x))
#define MIN_NBRS 100
#define SAFE_ZONE      1.2
#define DANGER_ZONE    0.90
#define MIN_CAP        50


FixNNP::FixNNP(LAMMPS *lmp, int narg, char **arg) :
        Fix(lmp, narg, arg),
        matvecs       (0      ),
        qeq_time      (0.0    ),
        nnp           (nullptr),
        list          (nullptr),
        pertype_option(nullptr),
        periodic      (false  ),
        qRef          (0.0    ),
        Q             (nullptr),
        coords        (nullptr),
        xf            (nullptr),
        yf            (nullptr),
        zf            (nullptr),
        xbuf          (nullptr),
        ntotal        (0      ),
        xbufsize      (0      ),
        nevery        (0      ),
        nnpflag       (0      ),
        n             (0      ),
        N             (0      ),
        m_fill        (0      ),
        n_cap         (0      ),
        m_cap         (0      ),
        pack_flag     (0      ),
        ngroup        (0      ),
        nprev         (0      ),
        Q_hist        (nullptr),
        p             (nullptr),
        q             (nullptr),
        r             (nullptr),
        d             (nullptr)
{
    if (narg<9 || narg>10) error->all(FLERR,"Illegal fix nnp command");

    nevery = utils::inumeric(FLERR,arg[3],false,lmp);
    if (nevery <= 0) error->all(FLERR,"Illegal fix nnp command");

    int len = strlen(arg[8]) + 1;
    pertype_option = new char[len];
    strcpy(pertype_option,arg[8]);

    nnp = nullptr;
    nnp = (PairNNP *) force->pair_match("^nnp",0);
    nnp->chi = nullptr;
    nnp->hardness = nullptr;
    nnp->sigmaSqrtPi = nullptr;
    nnp->gammaSqrt2 = nullptr;
    nnp->screening_info = nullptr;

    // User-defined minimization parameters used in pair_nnp as well
    // TODO: read only on proc 0 and then Bcast ?
    nnp->grad_tol = utils::numeric(FLERR,arg[4],false,lmp); //tolerance for gradient check
    nnp->min_tol = utils::numeric(FLERR,arg[5],false,lmp); //tolerance
    nnp->step = utils::numeric(FLERR,arg[6],false,lmp); //initial nnp->step size
    nnp->maxit = utils::inumeric(FLERR,arg[7],false,lmp); //maximum number of iterations


    // TODO: check these initializations and allocations
    coords = xf = yf = zf = nullptr;
    xbuf = nullptr;
    int natoms = atom->natoms;
    int nloc = atom->nlocal;
    xbufsize = 3*natoms;
    memory->create(coords,3*natoms,"fix_nnp:coords");
    memory->create(xbuf,xbufsize,"fix_nnp:buf");
    xf = &coords[0*natoms];
    yf = &coords[1*natoms];
    zf = &coords[2*natoms];
    ntotal = 0;

    /*
     *
     * n = n_cap = 0;
    N = 0;
    m_fill = m_cap = 0;
    pack_flag = 0;
    Q = NULL;
    nprev = 4;
    comm_forward = comm_reverse = 1;
    Q_hist = NULL;
    grow_arrays(atom->nmax);
    atom->add_callback(0);
    for (int i = 0; i < atom->nmax; i++)
        for (int j = 0; j < nprev; ++j)
            Q_hist[i][j] = 0;
    */
}

/* ---------------------------------------------------------------------- */

FixNNP::~FixNNP()
{
    if (copymode) return;

    delete[] pertype_option;

    // unregister callbacks to this fix from Atom class
    atom->delete_callback(id,0);

    //memory->destroy(Q_hist);
    memory->destroy(nnp->chi);
    memory->destroy(nnp->hardness);
    memory->destroy(nnp->sigmaSqrtPi);
    memory->destroy(nnp->gammaSqrt2);

    memory->destroy(xbuf);
    memory->destroy(coords);
    memory->destroy(xf);
    memory->destroy(yf);
    memory->destroy(zf);
}

void FixNNP::post_constructor()
{
    pertype_parameters(pertype_option);

}

int FixNNP::setmask()
{
    int mask = 0;
    mask |= PRE_FORCE;
    mask |= PRE_FORCE_RESPA;
    mask |= MIN_PRE_FORCE;
    return mask;
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

    isPeriodic();
    // TODO : do we really need a full NL in periodic cases ?
    if (periodic) { // periodic : full neighborlist
        neighbor->requests[irequest]->half = 0;
        neighbor->requests[irequest]->full = 1;
        nnp->ewaldPrecision = nnp->interface.getEwaldPrecision(); // read precision from n2p2
    }

}

void FixNNP::pertype_parameters(char *arg)
{
    if (strcmp(arg,"nnp") == 0) {
        nnpflag = 1;
        Pair *pair = force->pair_match("nnp",0);
        if (pair == NULL) error->all(FLERR,"No pair nnp for fix nnp");
    }
}

// Allocate QEq arrays
void FixNNP::allocate_QEq()
{
    int ne = atom->ntypes;
    memory->create(nnp->chi,atom->natoms + 1,"qeq:nnp->chi");
    memory->create(nnp->hardness,ne+1,"qeq:nnp->hardness");
    memory->create(nnp->sigmaSqrtPi,ne+1,"qeq:nnp->sigmaSqrtPi");
    memory->create(nnp->gammaSqrt2,ne+1,ne+1,"qeq:nnp->gammaSqrt2");
    memory->create(nnp->screening_info,5,"qeq:screening");

    // TODO: check, these communications are from original LAMMPS code
    /*MPI_Bcast(&nnp->chi[1],nloc,MPI_DOUBLE,0,world);
    MPI_Bcast(&nnp->hardness[1],nloc,MPI_DOUBLE,0,world);
    MPI_Bcast(&nnp->sigma[1],nloc,MPI_DOUBLE,0,world);*/
}

// Deallocate QEq arrays
void FixNNP::deallocate_QEq() {
    memory->destroy(nnp->chi);
    memory->destroy(nnp->hardness);
    memory->destroy(nnp->sigmaSqrtPi);
    memory->destroy(nnp->gammaSqrt2);
    memory->destroy(nnp->screening_info);
}

void FixNNP::init_list(int /*id*/, NeighList *ptr)
{
    list = ptr;
}

void FixNNP::setup_pre_force(int vflag)
{
    pre_force(vflag);
}

void FixNNP::calculate_electronegativities()
{
    // Run first set of atomic NNs
    process_first_network();

    // Read QEq arrays from n2p2 into LAMMPS
    nnp->interface.getQEqParams(nnp->chi,nnp->hardness,nnp->sigmaSqrtPi,nnp->gammaSqrt2,qRef);

    // Read screening function information from n2p2 into LAMMPS
    nnp->interface.getScreeningInfo(nnp->screening_info); //TODO: read function type
}

// Runs interface.process for electronegativities
void FixNNP::process_first_network()
{
    if(nnp->interface.getNnpType() == InterfaceLammps::NNPType::HDNNP_4G)
    {
        // Set number of local atoms and add index and element.
        nnp->interface.setLocalAtoms(atom->nlocal,atom->tag,atom->type);

        // Transfer local neighbor list to NNP interface.
        nnp->transferNeighborList();

        // Run the first NN for electronegativities
        nnp->interface.process();
    }
    else{ //TODO
        error->all(FLERR,"This fix style can only be used with a 4G-HDNNP.");
    }
}

void FixNNP::min_setup_pre_force(int vflag)
{
    setup_pre_force(vflag);
}

void FixNNP::min_pre_force(int vflag)
{
    pre_force(vflag);
}

// Main calculation routine, runs before pair->compute at each timennp->step
void FixNNP::pre_force(int /*vflag*/) {

    double *q = atom->q;

    deallocate_QEq();
    allocate_QEq();

    if(periodic) nnp->kspace_setup();

    // Calculate atomic electronegativities \Chi_i
    calculate_electronegativities();

    // Calculate the current total charge
    // TODO: communication
    double Qtot = 0.0;
    int n = atom->nlocal;
    for (int i = 0; i < n; i++) {
        Qtot += q[i];
    }
    // TODO: qRef was just set before in calculate_electronegativities() to
    // structure.chargeRef, why is it reset here already?
    qRef = Qtot;

    auto start = high_resolution_clock::now();
    // Minimize QEq energy and calculate atomic charges
    calculate_QEqCharges();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "CalculateQeqCharges : " << duration.count() << '\n';

    /*Qtot = 0.0;
    for (int i=0; i < atom->nlocal; i++)
    {
        std::cout << atom->q[i] << '\n';
        Qtot += atom->q[i];
    }
    std::cout << "Total charge : " << Qtot << '\n'; //TODO: remove, for debugging
    */

    /*gather_positions(); // parallelization based on dump_dcd.cpp

    if (comm->me != 0)
    {
        for (int i = 0; i < 12; i++)
        {
            std::cout << xf[i] << '\t' << yf[i] << '\t' << zf[i] <<  '\n';
        }
        for (int i = 0; i < 36; i++)
        {
            //std::cout << xbuf[i] << '\n';
        }
    }

    exit(0);


    for (int i = 0; i < n; i++) {
        std::cout << atom->q[i] << '\n';
    }
    exit(0);*/

    /*double t_start, t_end;

    if (update->ntimennp->step % nevery) return;
    if (comm->me == 0) t_start = MPI_Wtime();

    n = atom->nlocal;
    N = atom->nlocal + atom->nghost;

    // grow arrays if necessary
    // need to be atom->nmax in length

    if (atom->nmax > nmax) reallocate_storage();
    if (n > n_cap * DANGER_ZONE || m_fill > m_cap * DANGER_ZONE)
        //reallocate_matrix();
        */
    /*if (comm->me == 0) {
        t_end = MPI_Wtime();
        qeq_time = t_end - t_start;
    }*/
}


// QEq energy function, $E_{QEq}$
// TODO: communication
double FixNNP::QEq_f(const gsl_vector *v)
{
    int i,j,jmap;
    int *tag = atom->tag;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;
    int count = 0;

    double dx, dy, dz, rij;
    double qi,qj;
    double **x = atom->x;
    double *q = atom->q;

    double E_qeq_loc,E_qeq;
    double E_elec_loc,E_scr_loc,E_scr;
    double E_real,E_recip,E_self; // for periodic examples
    double iiterm,ijterm;
    double sf_real,sf_im;


    // TODO: indices & electrostatic energy
    E_qeq = 0.0;
    E_scr = 0.0;
    nnp->E_elec = 0.0;
    if (periodic)
    {
        //TODO: add an i-j loop, j over neighbors (realcutoff)
        double sqrt2eta = (sqrt(2.0) * nnp->ewaldEta);
        double c1 = 0; // counter for real-space operations
        E_recip = 0.0;
        E_real = 0.0;
        E_self = 0.0;
        for (i = 0; i < nlocal; i++) // over local atoms
        {
            qi = gsl_vector_get(v, i);
            double qi2 = qi * qi;
            // Self term
            E_self += qi2 * (1 / (2.0 * nnp->sigmaSqrtPi[type[i]]) -
                       1 / (sqrt(2.0 * M_PI) * nnp->ewaldEta));
            E_qeq += nnp->chi[i] * qi + 0.5 * nnp->hardness[type[i]] * qi2;
            E_scr -= qi2 / (2.0 * nnp->sigmaSqrtPi[type[i]]); // self screening term
            // Real Term
            // TODO: we loop over the full neighbor list, this can be optimized
            for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                j = list->firstneigh[i][jj];
                j &= NEIGHMASK;
                jmap = atom->map(tag[j]); //TODO: check
                qj = gsl_vector_get(v, jmap);
                dx = x[i][0] - x[j][0];
                dy = x[i][1] - x[j][1];
                dz = x[i][2] - x[j][2];
                rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
                double erfcRij = (erfc(rij / sqrt2eta) - erfc(rij / nnp->gammaSqrt2[type[i]][type[jmap]])) / rij;
                double real = 0.5 * qi * qj * erfcRij;
                E_real += real;
                if (rij <= nnp->screening_info[2]) {
                    E_scr += 0.5 * qi * qj * erf(rij/nnp->gammaSqrt2[type[i]][type[jmap]])*(nnp->screening_f(rij) - 1) / rij;
                }
                //count = count + 1;
            }
        }
        //std::cout << "f : " << count << '\n';
        // Reciprocal Term
        for (int k = 0; k < nnp->kcount; k++) // over k-space
        {
            sf_real = 0.0;
            sf_im = 0.0;
            // TODO: this loop over all atoms can be replaced by a MPIallreduce ?
            for (i = 0; i < nall; i++) //TODO: discuss this additional inner loop
            {
                qi = gsl_vector_get(v,i);
                sf_real += qi * nnp->sfexp_rl[k][i];
                sf_im += qi * nnp->sfexp_im[k][i];
            }
            // TODO: sf_real->sf_real_all or MPIAllreduce for E_recip ?
            E_recip += nnp->kcoeff[k] * (pow(sf_real,2) + pow(sf_im,2));
        }
        nnp->E_elec = E_real + E_self + E_recip;
        E_qeq += nnp->E_elec;
    }else
    {
        // first loop over local atoms
        for (i = 0; i < nlocal; i++) {
            qi = gsl_vector_get(v,i);
            // add i terms here
            iiterm = qi * qi / (2.0 * nnp->sigmaSqrtPi[type[i]]);
            E_qeq += iiterm + nnp->chi[i]*qi + 0.5*nnp->hardness[type[i]]*qi*qi;
            nnp->E_elec += iiterm;
            E_scr -= iiterm;
            // second loop over 'all' atoms
            for (j = i + 1; j < nall; j++) {
                qj = gsl_vector_get(v, j);
                dx = x[j][0] - x[i][0];
                dy = x[j][1] - x[i][1];
                dz = x[j][2] - x[i][2];
                rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
                ijterm = (erf(rij / nnp->gammaSqrt2[type[i]][type[j]]) / rij);
                E_qeq += ijterm * qi * qj;
                nnp->E_elec += ijterm;
                if(rij <= nnp->screening_info[2]) {
                    E_scr += ijterm * (nnp->screening_f(rij) - 1);
                }
            }
        }
    }

    nnp->E_elec = nnp->E_elec + E_scr; // add screening energy


    //MPI_Allreduce(E_qeq_loc,E_qeq,1,MPI_DOUBLE,MPI_SUM,world);

    return E_qeq;
}

// QEq energy function - wrapper
double FixNNP::QEq_f_wrap(const gsl_vector *v, void *params)
{
    return static_cast<FixNNP*>(params)->QEq_f(v);
}

// QEq energy gradient, $\partial E_{QEq} / \partial Q_i$
// TODO: communication
void FixNNP::QEq_df(const gsl_vector *v, gsl_vector *dEdQ)
{
    int i,j,jmap;
    int nlocal = atom->nlocal;
    int nall = atom->natoms;
    int *tag = atom->tag;
    int *type = atom->type;
    int count = 0;

    double dx, dy, dz, rij;
    double qi,qj;
    double **x = atom->x;
    double *q = atom->q;

    double grad;
    double grad_sum;
    double grad_i;
    double jsum,ksum; //summation over neighbors & kspace respectively
    double recip_sum;
    double sf_real,sf_im;

    //gsl_vector *dEdQ_loc;
    //dEdQ_loc = gsl_vector_alloc(nsize);

    grad_sum = 0.0;
    if (periodic)
    {
        double sqrt2eta = (sqrt(2.0) * nnp->ewaldEta);
        for (i = 0; i < nlocal; i++) // over local atoms
        {
            qi = gsl_vector_get(v,i);
            // Reciprocal contribution
            ksum = 0.0;
            for (int k = 0; k < nnp->kcount; k++) // over k-space
            {
                sf_real = 0.0;
                sf_im = 0.0;
                //TODO: this second loop should be over all, could this be saved ?
                for (j = 0; j < nlocal; j++)
                {
                    qj = gsl_vector_get(v,j);
                    sf_real += qj * nnp->sfexp_rl[k][j];
                    sf_im += qj * nnp->sfexp_im[k][j];
                }
                ksum += 2.0 * nnp->kcoeff[k] *
                        (sf_real * nnp->sfexp_rl[k][i] + sf_im * nnp->sfexp_im[k][i]);
            }
            // Real contribution - over neighbors
            jsum = 0.0;
            for (int jj = 0; jj < list->numneigh[i]; ++jj) {
                j = list->firstneigh[i][jj];
                j &= NEIGHMASK;
                jmap = atom->map(tag[j]); //TODO: check
                qj = gsl_vector_get(v, jmap);
                dx = x[i][0] - x[j][0];
                dy = x[i][1] - x[j][1];
                dz = x[i][2] - x[j][2];
                rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
                double erfcRij = (erfc(rij / sqrt2eta) - erfc(rij / nnp->gammaSqrt2[type[i]][type[jmap]]));
                jsum += qj * erfcRij / rij;
                //count = count + 1;
            }
            grad = jsum + ksum + nnp->chi[i] + nnp->hardness[type[i]]*qi +
                   qi * (1/(nnp->sigmaSqrtPi[type[i]])- 2/(nnp->ewaldEta * sqrt(2.0 * M_PI)));
            grad_sum += grad;
            gsl_vector_set(dEdQ,i,grad);
        }
        //std::cout << "df : " << count << '\n';
    }else
    {
        // first loop over local atoms
        for (i = 0; i < nlocal; i++) { // TODO: indices
            qi = gsl_vector_get(v,i);
            // second loop over 'all' atoms
            jsum = 0.0;
            for (j = 0; j < nall; j++) {
                if (j != i) {
                    qj = gsl_vector_get(v, j);
                    dx = x[j][0] - x[i][0];
                    dy = x[j][1] - x[i][1];
                    dz = x[j][2] - x[i][2];
                    rij = sqrt(SQR(dx) + SQR(dy) + SQR(dz));
                    jsum += qj * erf(rij / nnp->gammaSqrt2[type[i]][type[j]]) / rij;
                }
            }
            grad = nnp->chi[i] + nnp->hardness[type[i]]*qi + qi/(nnp->sigmaSqrtPi[type[i]]) + jsum;
            grad_sum += grad;
            gsl_vector_set(dEdQ,i,grad);
        }
    }

    // Gradient projection //TODO: communication ?
    for (i = 0; i < nall; i++){
        grad_i = gsl_vector_get(dEdQ,i);
        gsl_vector_set(dEdQ,i,grad_i - (grad_sum)/nall);
    }

    //MPI_Allreduce(dEdQ_loc,dEdQ,atom->natoms,MPI_DOUBLE,MPI_SUM,world);
}

// QEq energy gradient - wrapper
void FixNNP::QEq_df_wrap(const gsl_vector *v, void *params, gsl_vector *df)
{
    static_cast<FixNNP*>(params)->QEq_df(v, df);
}

// QEq f*df, $E_{QEq} * (\partial E_{QEq} / \partial Q_i)$
void FixNNP::QEq_fdf(const gsl_vector *v, double *f, gsl_vector *df)
{
    *f = QEq_f(v);
    QEq_df(v, df);
}

// QEq f*df - wrapper
void FixNNP::QEq_fdf_wrap(const gsl_vector *v, void *params, double *E, gsl_vector *df)
{
    static_cast<FixNNP*>(params)->QEq_fdf(v, E, df);
}

// Main minimization routine
// TODO: communication
void FixNNP::calculate_QEqCharges()
{
    size_t iter = 0;
    int status;
    int i,j;
    int nsize;

    double *q = atom->q;
    double qsum_it;
    double gradsum;
    double qi;
    double df,alpha;

    nsize = atom->natoms; // total number of atoms

    gsl_vector *Q; // charge vector
    QEq_minimizer.n = nsize; // it should be n_all in the future
    QEq_minimizer.f = &QEq_f_wrap; // function pointer f(x)
    QEq_minimizer.df = &QEq_df_wrap; // function pointer df(x)
    QEq_minimizer.fdf = &QEq_fdf_wrap;
    QEq_minimizer.params = this;

    // Allocation : set initial guess is the current charge vector
    Q = gsl_vector_alloc(nsize);
    for (i = 0; i < nsize; i++) {
        gsl_vector_set(Q,i,q[i]);
    }


    // Numeric vs. analytic derivatives check:
    gsl_vector *dEdQ = gsl_vector_calloc(nsize);
    QEq_df(Q, dEdQ);
    double const delta = 1.0E-5;
    for (i = 0; i < nsize; ++i)
    {
        double const qi = gsl_vector_get(Q, i);
        gsl_vector_set(Q, i, qi - delta);
        double const low = QEq_f(Q);
        gsl_vector_set(Q, i, qi + delta);
        double const high = QEq_f(Q);
        gsl_vector_set(Q, i, qi);
        double const numeric = (high - low) / (2.0 * delta);
        double const analytic = gsl_vector_get(dEdQ, i);
        fprintf(stderr, "NA-Check: dEQeq/dq(%3d) = %16.8E / %16.8E (Numeric/Analytic), Diff: %16.8E\n", i, numeric, analytic, numeric - analytic);
    }

    // TODO: is bfgs2 standard ?
    //T = gsl_multimin_fdfminimizer_conjugate_pr; // minimization algorithm
    T = gsl_multimin_fdfminimizer_vector_bfgs2;
    s = gsl_multimin_fdfminimizer_alloc(T, nsize);
    gsl_multimin_fdfminimizer_set(s, &QEq_minimizer, Q, nnp->step, nnp->min_tol); // tol = 0 might be expensive ???
    do
    {
        iter++;
        qsum_it = 0.0;
        gradsum = 0.0;
        //std::cout << "iteration : " << iter << '\n';
        //std::cout << "------------------------" << '\n';
        //auto start_it = high_resolution_clock::now();
        status = gsl_multimin_fdfminimizer_iterate(s);
        //auto stop_it = high_resolution_clock::now();
        //auto duration_it = duration_cast<microseconds>(stop_it - start_it);
        //std::cout << "Iteration time : " << duration_it.count() << '\n';

        // Projection (enforcing constraints)
        // TODO: could this be done more efficiently ?
        for(i = 0; i < nsize; i++) {
            qsum_it = qsum_it + gsl_vector_get(s->x, i); // total charge after the minimization nnp->step
        }

        for(i = 0; i < nsize; i++) {
            qi = gsl_vector_get(s->x,i);
            gsl_vector_set(s->x,i, qi - (qsum_it-qRef)/nsize); // charge projection
        }

        status = gsl_multimin_test_gradient(s->gradient, nnp->grad_tol); // check for convergence

        if (status == GSL_SUCCESS)
            printf ("Minimum charge distribution is found at iteration : %zu\n", iter);

    }
    while (status == GSL_CONTINUE && iter < nnp->maxit);

    // Read charges into LAMMPS atom->q array before deallocating
    for (i = 0; i < nsize; i++) {
        q[i] = gsl_vector_get(s->x,i);
    }

    // Deallocation
    gsl_multimin_fdfminimizer_free(s);
    gsl_vector_free(Q);
}

// Check if the system is periodic
void FixNNP::isPeriodic()
{
    if (domain->nonperiodic == 0) periodic = true;
    else                          periodic = false;
}


void FixNNP::pack_positions()
{
    int m,n;
    //tagint *tag = atom->tag;
    double **x = atom->x;
    //int *mask = atom->mask;
    int nlocal = atom->nlocal;

    m = n = 0;

    for (int i = 0; i < nlocal; i++)
    {
        //if (mask[i] & groupbit) {
            xbuf[m++] = x[i][0];
            xbuf[m++] = x[i][1];
            xbuf[m++] = x[i][2];
            //ids[n++] = tag[i];
        //}
    }
}

void FixNNP::gather_positions()
{
    int tmp,nlines;
    int size_one = 1;
    int nme =  atom->nlocal; //TODO this should be fixed
    int me = comm->me;
    int nprocs = comm->nprocs;

    MPI_Status status;
    MPI_Request request;

    pack_positions();

    // TODO: check all parameters and clean
    if (me == 0) {
        for (int iproc = 0; iproc < nprocs; iproc++) {
            if (iproc) {
                //MPI_Irecv(xbuf,maxbuf*size_one,MPI_DOUBLE,me+iproc,0,world,&request);
                MPI_Irecv(xbuf,xbufsize,MPI_DOUBLE,me+iproc,0,world,&request);
                MPI_Send(&tmp,0,MPI_INT,me+iproc,0,world);
                MPI_Wait(&request,&status);
                MPI_Get_count(&status,MPI_DOUBLE,&nlines);
                nlines /= size_one; // TODO : do we need ?
            } else nlines = nme;
            std::cout << 165 << '\n';
            std::cout << nlines << '\n';
            // copy buf atom coords into 3 global arrays
            int m = 0;
            for (int i = 0; i < nlines; i++) { //TODO : check this
                //std::cout << xbuf[m] << '\n';
                xf[ntotal] = xbuf[m++];
                yf[ntotal] = xbuf[m++];
                zf[ntotal] = xbuf[m++];
                ntotal++;
            }
            std::cout << ntotal << '\n';
            // if last chunk of atoms in this snapshot, write global arrays to file
            /*if (ntotal == natoms) {
                ntotal = 0;
            }*/
        }
        //if (flush_flag && fp) fflush(fp);
    }
    else {
        MPI_Recv(&tmp,0,MPI_INT,me,0,world,MPI_STATUS_IGNORE);
        MPI_Rsend(xbuf,xbufsize,MPI_DOUBLE,me,0,world);
    }
}


/// Fix communication subroutines inherited from the parent Fix class
/// They are used in all fixes in LAMMPS, TODO: check, they might be helpful for us as well

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





