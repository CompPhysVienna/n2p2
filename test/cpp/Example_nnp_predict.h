#ifndef EXAMPLE_NNP_PREDICT_H
#define EXAMPLE_NNP_PREDICT_H

#include "Example_nnp.h"
#include "BoostDataContainer.h"
#include "Vec3D.h"
#include <cstddef> // std::size_t
#include <vector>

struct Example_nnp_predict : public Example_nnp
{
    double                   energy;
    std::vector<std::string> createdFiles;
    // Examples without full reference forces in ExamplePrediction.h: forces
    // and charges of a few atoms (1-based index in output.data), recorded
    // from the CPU code.
    std::vector<std::size_t> checkAtoms;
    std::vector<nnp::Vec3D>  checkForces;
    std::vector<double>      checkCharges;

    Example_nnp_predict(std::string name)
        : Example_nnp("nnp-predict", name) {};
};

template<>
void BoostDataContainer<Example_nnp_predict>::setup()
{
    Example_nnp_predict* e = nullptr;

    examples.push_back(Example_nnp_predict("H2O_RPBE-D3"));
    e = &(examples.back());
    e->args = "0 ";
    e->energy = -2.7564547347815904E+04;
    e->createdFiles.push_back("energy.out");
    e->createdFiles.push_back("nnatoms.out");
    e->createdFiles.push_back("nnforces.out");
    e->createdFiles.push_back("output.data");

    examples.push_back(Example_nnp_predict("Cu2S_PBE"));
    e = &(examples.back());
    e->args = "0 ";
    e->energy = -5.7365603183874578E+02;
    e->createdFiles.push_back("energy.out");
    e->createdFiles.push_back("nnatoms.out");
    e->createdFiles.push_back("nnforces.out");
    e->createdFiles.push_back("output.data");

    examples.push_back(Example_nnp_predict("Anisole_SCAN"));
    e = &(examples.back());
    e->args = "0 ";
    e->energy = -9.6749614724733078E+02;
    e->createdFiles.push_back("energy.out");
    e->createdFiles.push_back("nnatoms.out");
    e->createdFiles.push_back("nnforces.out");
    e->createdFiles.push_back("output.data");

    examples.push_back(Example_nnp_predict("DMABN_SCAN"));
    e = &(examples.back());
    e->args = "0 ";
    e->energy = -7.6864264790108564E+01;
    e->createdFiles.push_back("energy.out");
    e->createdFiles.push_back("nnatoms.out");
    e->createdFiles.push_back("nnforces.out");
    e->createdFiles.push_back("output.data");

    examples.push_back(Example_nnp_predict("Ethylbenzene_SCAN"));
    e = &(examples.back());
    e->args = "0 ";
    e->energy = -8.2198742206281679E+02;
    e->createdFiles.push_back("energy.out");
    e->createdFiles.push_back("nnatoms.out");
    e->createdFiles.push_back("nnforces.out");
    e->createdFiles.push_back("output.data");

    examples.push_back(Example_nnp_predict("H2O_RPBE-D3_4G"));
    e = &(examples.back());
    e->args = "0 ";
    e->energy = -9.8016327504792771E+03;
    e->checkAtoms = {1, 193, 384};
    e->checkForces = {
        nnp::Vec3D( 3.9996152193537875E-03,  1.5803086815637595E-03, -2.8555991882522006E-03),
        nnp::Vec3D( 5.0084138828683436E-05,  4.3591619812374674E-03,  3.1341279334520379E-03),
        nnp::Vec3D( 2.1748206873345314E-03, -2.0820654393644937E-03,  1.8178322147645919E-03)};
    e->checkCharges = {-2.1813800425687441E-01,
                       -2.1807130886393572E-01,
                        1.0907260351004523E-01};
    e->createdFiles.push_back("energy.out");
    e->createdFiles.push_back("nnatoms.out");
    e->createdFiles.push_back("nnforces.out");
    e->createdFiles.push_back("output.data");

    return;
}

#endif
