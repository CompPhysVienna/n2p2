#ifndef EXAMPLE_LAMMPS_HDNNP_H
#define EXAMPLE_LAMMPS_HDNNP_H

#include "Example.h"
#include "BoostDataContainer.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

struct Example_lammps_hdnnp : public Example
{
    std::string tool;
    std::string command;
    std::string args;
    std::string pathBin;
    std::string pathData;

    std::size_t lastTimeStep;
    double      potentialEnergy;
    double      totalEnergy;
    /// Optional: file with reference forces (columns: id fx fy fz), compared
    /// with the forces written by the example to "forces.dump".
    std::string forcesReference;
    /// Absolute tolerance for the forces.
    double      forcesTolerance;
    /// Optional: finite-difference check of the forces instead of the energy
    /// comparison. The example writes the forces of the unperturbed structure
    /// to "forces.dump" and then the potential energy (thermo_style custom
    /// step pe) for displacements of +fdStep and -fdStep of each atom in
    /// fdAtoms along x, y and z, in this order.
    std::vector<std::size_t> fdAtoms;
    /// Displacement (LAMMPS length units).
    double      fdStep;
    /// Absolute tolerance for the forces from finite differences.
    double      fdTolerance;

    Example_lammps_hdnnp(std::string name) :
        pathBin("../../../bin"),
        pathData("../../examples/interface-LAMMPS/"),
        forcesTolerance(0.0),
        fdStep(0.0),
        fdTolerance(0.0)
    {
        this->name = name;
        this->tool = "lmp_mpi";
        this->description = std::string("Test example \"")
                          + this->name
                          + "\" with tool \""
                          + this->tool + "\"";
        this->command = pathBin + "/" + this->tool;
        this->pathData += "/" + this->name;
    }
};

template<>
void BoostDataContainer<Example_lammps_hdnnp>::setup()
{
    Example_lammps_hdnnp* e = nullptr;

    examples.push_back(Example_lammps_hdnnp("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "-in md.lmp ";
    e->lastTimeStep = 5;
    e->potentialEnergy = -6000559.4;
    e->totalEnergy = -6000220.2;

    examples.push_back(Example_lammps_hdnnp("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "-in md.lmp ";
    e->lastTimeStep = 100;
    e->potentialEnergy = -574.21185;
    e->totalEnergy = -573.6561;

    // 4G-HDNNP (pair_style hdnnp/4g, kspace_style hdnnp, fix hdnnp charge
    // equilibration), single point as in the shipped md.lmp (0 steps).
    // The potential energy agrees with nnp-predict on the same structure
    // (examples/interface-LAMMPS/AlAuMgO_4G/nnp-predict, -54395.98123 eV) to
    // 4E-7 eV, charges to 3E-7 e and forces to 1E-7 eV/A.
    examples.push_back(Example_lammps_hdnnp("AlAuMgO_4G"));
    e = &(examples.back());
    e->args = "-in md.lmp ";
    e->lastTimeStep = 0;
    e->potentialEnergy = -54395.981;
    e->totalEnergy = -54391.754;

    // 4G-HDNNP with normalized units (nnp-norm) and an element mapping not in
    // the order of the atomic numbers. Energy, charges and forces agree with
    // nnp-predict (examples/nnp-predict/H2O_SPCFw_4G, -120.21747016 eV): the
    // forces of nnp-predict are in forces.nnp-predict (the remaining
    // difference, about 2E-6 eV/A, is due to the different Ewald summations).
    examples.push_back(Example_lammps_hdnnp("H2O_SPCFw_4G"));
    e = &(examples.back());
    e->args = "-in md.lmp ";
    e->lastTimeStep = 0;
    e->potentialEnergy = -120.21748;
    e->totalEnergy = -120.21748;
    e->forcesReference = "forces.nnp-predict";
    e->forcesTolerance = 1.0E-5;

    // Finite-difference check of the forces with the same model and structure
    // (md-fd.lmp): the potential energy is consistent with the forces only if
    // the charge equilibration and the force lambda are solved accurately.
    // Conjugate gradients: largest deviation 7E-6 eV/A. The former GSL BFGS
    // minimizations stopped at a relative gradient of about 1E-8: 1.9E-4 eV/A
    // here, and energy errors up to 1E-3 eV in other 630-atom structures.
    examples.push_back(Example_lammps_hdnnp("H2O_SPCFw_4G"));
    e = &(examples.back());
    e->description += " (finite-difference check of the forces)";
    e->args = "-in md-fd.lmp ";
    e->fdAtoms = {1, 2};
    e->fdStep = 1.0E-3;
    e->fdTolerance = 3.0E-5;

    // The same model and structure in eV and Angstrom without normalization
    // (four_pi_epsilon != 1, no unit conversion in LAMMPS): same results.
    examples.push_back(Example_lammps_hdnnp("H2O_SPCFw_4G_metal"));
    e = &(examples.back());
    e->args = "-in md.lmp ";
    e->lastTimeStep = 0;
    e->potentialEnergy = -120.21748;
    e->totalEnergy = -120.21748;
    e->forcesReference = "forces.nnp-predict";
    e->forcesTolerance = 1.0E-5;

    return;
}

#endif
