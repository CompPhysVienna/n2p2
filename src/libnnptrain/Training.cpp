// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "Training.h"
#include "GradientDescent.h"
#include "KalmanFilter.h"
#include "NeuralNetwork.h"
#include "utility.h"
#include "mpi-extra.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm> // std::sort, std::fill, std::find
#include <cmath>     // fabs
#include <cstdlib>   // atoi
#include <gsl/gsl_rng.h>
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error, std::range_error
#include <utility>   // std::piecewise_construct, std::forward_as_tuple

using namespace std;
using namespace nnp;

Training::Training() : Dataset(),
                       updaterType                (UT_GD          ),
                       parallelMode               (PM_TRAIN_RK0   ),
                       jacobianMode               (JM_SUM         ),
                       updateStrategy             (US_COMBINED    ),
                       hasUpdaters                (false          ),
                       hasStructures              (false          ),
                       useForces                  (false          ),
                       repeatedEnergyUpdates      (false          ),
                       freeMemory                 (false          ),
                       writeTrainingLog           (false          ),
                       stage                      (0              ),
                       numUpdaters                (0              ),
                       numEpochs                  (0              ),
                       epoch                      (0              ),
                       writeWeightsEvery          (0              ),
                       writeWeightsAlways         (0              ),
                       writeNeuronStatisticsEvery (0              ),
                       writeNeuronStatisticsAlways(0              ),
                       numWeights                 (0              ),
                       forceWeight                (0.0            ),
                       trainingLogFileName        ("train-log.out")
{
    sw["setup"].start();
}

Training::~Training()
{
    for (vector<Updater*>::iterator it = updaters.begin();
         it != updaters.end(); ++it)
    {
        if (updaterType == UT_GD)
        {
            delete dynamic_cast<GradientDescent*>(*it);
        }
        else if (updaterType == UT_KF)
        {
            delete dynamic_cast<KalmanFilter*>(*it);
        }
    }

    if (trainingLog.is_open()) trainingLog.close();
}

void Training::selectSets()
{
    log << "\n";
    log << "*** DEFINE TRAINING/TEST SETS ***********"
           "**************************************\n";
    log << "\n";

    vector<string> pCheck = {"force", "charge"};
    bool check = false;
    for (auto k : pk)
    {
        check |= (find(pCheck.begin(), pCheck.end(), k) != pCheck.end());
    }
    vector<size_t> numAtomsPerElement(numElements, 0);

    double testSetFraction = atof(settings["test_fraction"].c_str());
    log << strpr("Desired test set ratio      : %f\n", testSetFraction);
    if (structures.size() > 0) hasStructures = true;
    else hasStructures = false;

    string k;
    for (size_t i = 0; i < structures.size(); ++i)
    {
        Structure& s = structures.at(i);
        if (gsl_rng_uniform(rng) < testSetFraction)
        {
            s.sampleType = Structure::ST_TEST;
            size_t const& na = s.numAtoms;
            k = "energy"; if (p.exists(k)) p[k].numTestPatterns++;
            k = "force";  if (p.exists(k)) p[k].numTestPatterns += 3 * na;
            k = "charge"; if (p.exists(k)) p[k].numTestPatterns += na;
        }
        else
        {
            s.sampleType = Structure::ST_TRAINING;
            for (size_t j = 0; j < numElements; ++j)
            {
                numAtomsPerElement.at(j) += s.numAtomsPerElement.at(j);
            }

            k = "energy";
            if (p.exists(k))
            {
                p[k].numTrainPatterns++;
                p[k].updateCandidates.push_back(UpdateCandidate());
                p[k].updateCandidates.back().s = i;
            }
            k = "force";
            if (p.exists(k))
            {
                p[k].numTrainPatterns += 3 * s.numAtoms;
                for (vector<Atom>::const_iterator it = s.atoms.begin();
                     it != s.atoms.end(); ++it)
                {
                    for (size_t j = 0; j < 3; ++j)
                    {
                        p[k].updateCandidates.push_back(UpdateCandidate());
                        p[k].updateCandidates.back().s = i;
                        p[k].updateCandidates.back().a = it->index;
                        p[k].updateCandidates.back().c = j;
                    }
                }
            }
            k = "charge";
            if (p.exists(k))
            {
                p[k].numTrainPatterns += s.numAtoms;
                for (vector<Atom>::const_iterator it = s.atoms.begin();
                     it != s.atoms.end(); ++it)
                {
                    p[k].updateCandidates.push_back(UpdateCandidate());
                    p[k].updateCandidates.back().s = i;
                    p[k].updateCandidates.back().a = it->index;
                }
            }
        }
    }
    for (size_t i = 0; i < numElements; ++i)
    {
        if (hasStructures && (numAtomsPerElement.at(i) == 0))
        {
            log << strpr("WARNING: Process %d has no atoms of element "
                         "%d (%2s).\n",
                         myRank,
                         i,
                         elementMap[i].c_str());
        }
    }
    for (auto k : pk)
    {
        MPI_Allreduce(MPI_IN_PLACE, &(p[k].numTrainPatterns), 1, MPI_SIZE_T, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, &(p[k].numTestPatterns) , 1, MPI_SIZE_T, MPI_SUM, comm);
        double sum = p[k].numTrainPatterns + p[k].numTestPatterns;
        log << "Training/test split of data set for property \"" + k + "\":\n";
        log << strpr("- Total    patterns : %.0f\n", sum);
        log << strpr("- Training patterns : %d\n", p[k].numTrainPatterns);
        log << strpr("- Test     patterns : %d\n", p[k].numTestPatterns);
        log << strpr("- Test set fraction : %f\n", p[k].numTestPatterns / sum);
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Training::writeSetsToFiles()
{
    log << "\n";
    log << "*** WRITE TRAINING/TEST SETS ************"
           "**************************************\n";
    log << "\n";

    string fileName = strpr("train.data.%04d", myRank);
    ofstream fileTrain;
    fileTrain.open(fileName.c_str());
    if (!fileTrain.is_open())
    {
        runtime_error(strpr("ERROR: Could not open file %s\n",
                            fileName.c_str()));
    }
    fileName = strpr("test.data.%04d", myRank);
    ofstream fileTest;
    fileTest.open(fileName.c_str());
    if (!fileTest.is_open())
    {
        runtime_error(strpr("ERROR: Could not open file %s\n",
                            fileName.c_str()));
    }
    for (vector<Structure>::iterator it = structures.begin();
         it != structures.end(); ++it)
    {
        // Energy offsets are already subtracted at this point.
        // Here, we quickly add them again to provide consistent data sets.
        addEnergyOffset(*it);
        if (it->sampleType == Structure::ST_TRAINING)
        {
            it->writeToFile(&fileTrain);
        }
        else if (it->sampleType == Structure::ST_TEST)
        {
            it->writeToFile(&fileTest);
        }
        // Subtract energy offsets again.
        removeEnergyOffset(*it);
    }
    fileTrain.flush();
    fileTrain.close();
    fileTest.flush();
    fileTest.close();
    MPI_Barrier(comm);
    if (myRank == 0)
    {
        log << "Writing training/test set to files:\n";
        log << " - train.data\n";
        log << " - test.data\n";
        fileName = "train.data";
        combineFiles(fileName);
        fileName = "test.data";
        combineFiles(fileName);
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Training::initializeWeights()
{
    log << "\n";
    log << "*** WEIGHT INITIALIZATION ***************"
           "**************************************\n";
    log << "\n";

    if (settings.keywordExists("nguyen_widrow_weights_short") &&
        settings.keywordExists("precondition_weights"))
    {
        throw runtime_error("ERROR: Nguyen Widrow and preconditioning weights"
                            " initialization are incompatible\n");
    }

    // Charge NN.
    if (nnpType == NNPType::SHORT_CHARGE_NN)
    {
        log << "Setting up charge neural networks:\n";
        if ((stage == 1 && settings.keywordExists("use_old_weights_charge")) ||
            (stage == 2))
        {
            log << "Reading old weights from files.\n";
            readNeuralNetworkWeights("charge", "weightse.%03zu.data");
        }
        else randomizeNeuralNetworkWeights("charge");
    }

    // Short-range NN.
    if(nnpType == NNPType::SHORT_ONLY)
    {
        log << "Setting up short-range neural networks:\n";
        if (settings.keywordExists("use_old_weights_short"))
        {
            log << "Reading old weights from files.\n";
            readNeuralNetworkWeights("short", "weights.%03zu.data");
        }
        else randomizeNeuralNetworkWeights("short");
    }
    else if(nnpType == NNPType::SHORT_CHARGE_NN && stage == 2)
    {
        log << "Setting up short-range neural networks:\n";
        if (settings.keywordExists("use_old_weights_short"))
        {
            log << "Reading old weights from files.\n";
            readNeuralNetworkWeights("short", "weights.%03zu.data");
        }
        else randomizeNeuralNetworkWeights("short");
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Training::initializeWeightsMemory(UpdateStrategy updateStrategy)
{
    this->updateStrategy = updateStrategy;
    numWeights= 0;
    if (updateStrategy == US_COMBINED)
    {
        log << strpr("Combined updater for all elements selected: "
                     "UpdateStrategy::US_COMBINED (%d)\n", updateStrategy);
        numUpdaters = 1;
        log << strpr("Number of weight updaters    : %zu\n", numUpdaters);
        for (size_t i = 0; i < numElements; ++i)
        {
            weightsOffset.push_back(numWeights);
            numWeights += elements.at(i).neuralNetworks.at(nnId)
                          .getNumConnections();
        }
        weights.resize(numUpdaters);
        weights.at(0).resize(numWeights, 0.0);
        numWeightsPerUpdater.push_back(numWeights);
        log << strpr("Total fit parameters         : %zu\n", numWeights);
    }
    else if (updateStrategy == US_ELEMENT)
    {
        log << strpr("Separate updaters for elements selected: "
                     "UpdateStrategy::US_ELEMENT (%d)\n", updateStrategy);
        numUpdaters = numElements;
        log << strpr("Number of weight updaters    : %zu\n", numUpdaters);
        weights.resize(numUpdaters);
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            size_t n = elements.at(i).neuralNetworks.at(nnId)
                       .getNumConnections();
            weights.at(i).resize(n, 0.0);
            numWeightsPerUpdater.push_back(n);
            log << strpr("Fit parameters for element %2s: %zu\n",
                         elements.at(i).getSymbol().c_str(),
                         n);
        }
    }
    else
    {
        throw runtime_error("ERROR: Unknown update strategy.\n");
    }

    return;
}

void Training::setStage(size_t stage)
{
    this->stage = stage;

    // NNP of type SHORT_ONLY requires:
    // * "energy" (optionally: "force")
    if (nnpType == NNPType::SHORT_ONLY)
    {
        pk.push_back("energy");
        if (settings.keywordExists("use_short_forces")) pk.push_back("force");
    }
    // NNP of type SHORT_CHARGE_NN requires:
    // * stage 1: "charge"
    // * stage 2: "energy" (optionally: "force")
    else if (nnpType == NNPType::SHORT_CHARGE_NN)
    {
        if (stage == 1) pk.push_back("charge");
        else if (stage == 2)
        {
            pk.push_back("energy");
            if (settings.keywordExists("use_short_forces"))
            {
                pk.push_back("force");
            }
        }
    }

    // Initialize all training properties which will be used.
    auto initP = [this](string key) {p.emplace(piecewise_construct,
                                               forward_as_tuple(key),
                                               forward_as_tuple(key));
                 };
    for (auto k : pk) initP(k);

    return;
}

void Training::setupTraining()
{
    log << "\n";
    log << "*** SETUP: TRAINING *********************"
           "**************************************\n";
    log << "\n";

    if (nnpType == NNPType::SHORT_CHARGE_NN)
    {
        log << strpr("Running stage %zu of training:", stage);
        if      (stage == 1) log << "charge NN fitting.\n";
        else if (stage == 2) log << "short-range NN fitting.\n";
        else throw runtime_error("\nERROR: Unknown training stage.\n");
    }

    if (nnpType == NNPType::SHORT_ONLY ||
        (nnpType == NNPType::SHORT_CHARGE_NN && stage == 2))
    {
        nnId = "short";
        useForces = settings.keywordExists("use_short_forces");
        if (useForces)
        {
            log << "Forces will be used for training.\n";
            if (settings.keywordExists("force_weight"))
            {
                forceWeight = atof(settings["force_weight"].c_str());
            }
            else
            {
                log << "WARNING: Force weight not set, using default value.\n";
                forceWeight = 1.0;
            }
            log << strpr("Force update weight: %10.2E\n", forceWeight);
        }
        else
        {
            log << "Only energies will used for training.\n";
        }
    }
    else if (nnpType == NNPType::SHORT_CHARGE_NN && stage == 1)
    {
        nnId = "charge";
    }
    log << "Training will act on \"" << nnId << "\" neural networks.\n";

    if (settings.keywordExists("main_error_metric"))
    {
        string k;
        if (settings["main_error_metric"] == "RMSEpa")
        {
            k = "energy"; if (p.exists(k)) p[k].displayMetric = "RMSEpa";
            k = "force";  if (p.exists(k)) p[k].displayMetric = "RMSE";
            k = "charge"; if (p.exists(k)) p[k].displayMetric = "RMSE";
        }
        else if (settings["main_error_metric"] == "RMSE")
        {
            k = "energy"; if (p.exists(k)) p[k].displayMetric = "RMSE";
            k = "force";  if (p.exists(k)) p[k].displayMetric = "RMSE";
            k = "charge"; if (p.exists(k)) p[k].displayMetric = "RMSE";
        }
        else if (settings["main_error_metric"] == "MAEpa")
        {
            k = "energy"; if (p.exists(k)) p[k].displayMetric = "MAEpa";
            k = "force";  if (p.exists(k)) p[k].displayMetric = "MAE";
            k = "charge"; if (p.exists(k)) p[k].displayMetric = "MAE";
        }
        else if (settings["main_error_metric"] == "MAE")
        {
            k = "energy"; if (p.exists(k)) p[k].displayMetric = "MAE";
            k = "force";  if (p.exists(k)) p[k].displayMetric = "MAE";
            k = "charge"; if (p.exists(k)) p[k].displayMetric = "MAE";
        }
        else
        {
            throw runtime_error("ERROR: Unknown error metric.\n");
        }
    }
    else
    {
        string k;
        k = "energy"; if (p.exists(k)) p[k].displayMetric = "RMSEpa";
        k = "force";  if (p.exists(k)) p[k].displayMetric = "RMSE";
        k = "charge"; if (p.exists(k)) p[k].displayMetric = "RMSE";
    }

    updaterType = (UpdaterType)atoi(settings["updater_type"].c_str());
    if (updaterType == UT_GD)
    {
        log << strpr("Weight update via gradient descent selected: "
                     "updaterType::UT_GD (%d)\n",
                     updaterType);
    }
    else if (updaterType == UT_KF)
    {
        log << strpr("Weight update via Kalman filter selected: "
                     "updaterType::UT_KF (%d)\n",
                     updaterType);
    }
    else if (updaterType == UT_LM)
    {
        throw runtime_error("ERROR: LM algorithm not yet implemented.\n");
        log << strpr("Weight update via Levenberg-Marquardt algorithm "
                     "selected: updaterType::UT_LM (%d)\n",
                     updaterType);
    }
    else
    {
        throw runtime_error("ERROR: Unknown updater type.\n");
    }

    parallelMode = (ParallelMode)atoi(settings["parallel_mode"].c_str());
    //if (parallelMode == PM_DATASET)
    //{
    //    log << strpr("Serial training selected: "
    //                 "ParallelMode::PM_DATASET (%d)\n",
    //                 parallelMode);
    //}
    if (parallelMode == PM_TRAIN_RK0)
    {
        log << strpr("Parallel training (rank 0 updates) selected: "
                     "ParallelMode::PM_TRAIN_RK0 (%d)\n",
                     parallelMode);
    }
    else if (parallelMode == PM_TRAIN_ALL)
    {
        log << strpr("Parallel training (all ranks update) selected: "
                     "ParallelMode::PM_TRAIN_ALL (%d)\n",
                     parallelMode);
    }
    else
    {
        throw runtime_error("ERROR: Unknown parallelization mode.\n");
    }

    jacobianMode = (JacobianMode)atoi(settings["jacobian_mode"].c_str());
    if (jacobianMode == JM_SUM)
    {
        log << strpr("Gradient summation only selected: "
                     "JacobianMode::JM_SUM (%d)\n", jacobianMode);
        log << "No Jacobi matrix, gradients of all training candidates are "
               "summed up instead.\n";
    }
    else if (jacobianMode == JM_TASK)
    {
        log << strpr("Per-task Jacobian selected: "
                     "JacobianMode::JM_TASK (%d)\n",
                     jacobianMode);
        log << "One Jacobi matrix row per MPI task is stored, within each "
               "task gradients are summed up.\n";
    }
    else if (jacobianMode == JM_FULL)
    {
        log << strpr("Full Jacobian selected: "
                     "JacobianMode::JM_FULL (%d)\n",
                     jacobianMode);
        log << "Each update candidate generates one Jacobi matrix "
               "row entry.\n";
    }
    else
    {
        throw runtime_error("ERROR: Unknown Jacobian mode.\n");
    }

    if (updaterType == UT_GD && jacobianMode != JM_SUM)
    {
        throw runtime_error("ERROR: Gradient descent methods can only be "
                            "combined with Jacobian mode JM_SUM.\n");
    }

    updateStrategy = (UpdateStrategy)atoi(settings["update_strategy"].c_str());
    // This section is pushed into a separate function because it's needed also
    // for testing purposes.
    initializeWeightsMemory(updateStrategy);
    // Now it is possible to fill the weights arrays with weight parameters
    // from the neural network.
    getWeights();

    // Set up update candidate selection modes.
    setupSelectionMode("all");
    for (auto k : pk) setupSelectionMode(k);

    log << "-----------------------------------------"
           "--------------------------------------\n";
    repeatedEnergyUpdates = settings.keywordExists("repeated_energy_update");
    if (useForces && repeatedEnergyUpdates)
    {
        throw runtime_error("ERROR: Repeated energy updates are not correctly"
                            " implemented at the moment.\n");
        //log << "After each force update an energy update for the\n";
        //log << "corresponding structure will be performed.\n";
    }

    freeMemory = !(settings.keywordExists("memorize_symfunc_results"));
    if (freeMemory)
    {
        log << "Symmetry function memory is cleared after each calculation.\n";
    }
    else
    {
        log << "Symmetry function memory is reused (HIGH MEMORY USAGE!).\n";
    }

    numEpochs = (size_t)atoi(settings["epochs"].c_str());
    log << strpr("Training will be stopped after %zu epochs.\n", numEpochs);

    // Set up how often comparison output files should be written.
    for (auto k : pk) setupFileOutput(k);
    // Set up how often weight files should be written.
    setupFileOutput("weights_epoch");
    // Set up how often neuron statistics files should be written.
    setupFileOutput("neuronstats");

    // Prepare training log header.
    writeTrainingLog = settings.keywordExists("write_trainlog");
    if (writeTrainingLog && myRank == 0)
    {
        if (nnpType == NNPType::SHORT_CHARGE_NN)
        {
            trainingLogFileName += strpr(".stage-%zu", stage);
        }
        log << strpr("Training log with update information will be written to:"
                     " %s.\n", trainingLogFileName.c_str());
        trainingLog.open(trainingLogFileName.c_str());

        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back("Detailed information on each weight update.");
        colSize.push_back(3);
        colName.push_back("U");
        colInfo.push_back("Update type (E = energy, F = force, Q = charge).");
        colSize.push_back(5);
        colName.push_back("epoch");
        colInfo.push_back("Current training epoch.");
        colSize.push_back(10);
        colName.push_back("count");
        colInfo.push_back("Update counter (Multiple lines with identical count"
                          " for multi-streaming!).");
        colSize.push_back(5);
        colName.push_back("proc");
        colInfo.push_back("MPI process providing this update candidate.");
        colSize.push_back(3);
        colName.push_back("tl");
        colInfo.push_back("Threshold loop counter.");
        colSize.push_back(10);
        colName.push_back("rmse_frac");
        colInfo.push_back("Update candidates error divided by this "
                          "epochs RMSE.");
        colSize.push_back(10);
        colName.push_back("s_ind_g");
        colInfo.push_back("Global structure index.");
        colSize.push_back(5);
        colName.push_back("s_ind");
        colInfo.push_back("Local structure index on this MPI process.");
        colSize.push_back(5);
        colName.push_back("a_ind");
        colInfo.push_back("Atom index.");
        colSize.push_back(2);
        colName.push_back("c");
        colInfo.push_back("Force component (0 = x, 1 = y, 2 = z).");
        appendLinesToFile(trainingLog,
                          createFileHeader(title, colSize, colName, colInfo));
    }

    // Compute number of updates and properties per update.
    log << "-----------------------------------------"
           "--------------------------------------\n";
    for (auto k : pk) setupUpdatePlan(k);
    if (p.exists("energy") && p.exists("force"))
    {
        Property& pe = p["energy"];
        Property& pf = p["force"];
        log << strpr("Energy to force ratio                        : "
                     "     1 : %5.1f\n",
                     static_cast<double>(
                         pf.numUpdates * pf.patternsPerUpdateGlobal)
                         / (pe.numUpdates * pe.patternsPerUpdateGlobal));
        log << strpr("Energy to force percentages                  : "
                     "%5.1f%% : %5.1f%%\n",
                     pe.numUpdates * pe.patternsPerUpdateGlobal * 100.0 /
                     (pe.numUpdates * pe.patternsPerUpdateGlobal
                     + pf.numUpdates * pf.patternsPerUpdateGlobal),
                     pf.numUpdates * pf.patternsPerUpdateGlobal * 100.0 /
                     (pe.numUpdates * pe.patternsPerUpdateGlobal
                     + pf.numUpdates * pf.patternsPerUpdateGlobal));
    }
    double totalUpdates = 0.0;
    for (auto k : pk) totalUpdates += p[k].numUpdates;
    log << "-----------------------------------------"
           "--------------------------------------\n";

    // Allocate error and Jacobian arrays.
    for (auto k : pk) allocateArrays(k);
    log << "-----------------------------------------"
           "--------------------------------------\n";

    // Set up new C++11 random number generator (TODO: move it!).
    rngGlobalNew.seed(gsl_rng_get(rngGlobal));
    rngNew.seed(gsl_rng_get(rng));

    // Updater setup.
    GradientDescent::DescentType descentType = GradientDescent::DT_FIXED;
    if (updaterType == UT_GD)
    {
        descentType = (GradientDescent::DescentType)
                      atoi(settings["gradient_type"].c_str());
    }
    KalmanFilter::KalmanType kalmanType = KalmanFilter::KT_STANDARD;
    if (updaterType == UT_KF)
    {
        kalmanType = (KalmanFilter::KalmanType)
                     atoi(settings["kalman_type"].c_str());
    }

    for (size_t i = 0; i < numUpdaters; ++i)
    {
        if ( (myRank == 0) || (parallelMode == PM_TRAIN_ALL) )
        {
            if (updaterType == UT_GD)
            {
                updaters.push_back(
                    (Updater*)new GradientDescent(numWeightsPerUpdater.at(i),
                                                  descentType));
            }
            else if (updaterType == UT_KF)
            {
                updaters.push_back(
                    (Updater*)new KalmanFilter(numWeightsPerUpdater.at(i),
                                               kalmanType));
            }
            updaters.back()->setState(&(weights.at(i).front()));
            updaters.back()->setupTiming(strpr("wupd%zu", i));
            updaters.back()->resetTimingLoop();
        }
    }
    if (updaters.size() > 0) hasUpdaters = true;
    else hasUpdaters = false;

    if (hasUpdaters && updaterType == UT_GD)
    {
        if (descentType == GradientDescent::DT_FIXED)
        {
            double const eta = atof(settings["gradient_eta"].c_str());
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                GradientDescent* u =
                    dynamic_cast<GradientDescent*>(updaters.at(i));
                u->setParametersFixed(eta);
            }
        }
        if (descentType == GradientDescent::DT_ADAM)
        {
            double const eta = atof(settings["gradient_adam_eta"].c_str());
            double const beta1 = atof(settings["gradient_adam_beta1"].c_str());
            double const beta2 = atof(settings["gradient_adam_beta2"].c_str());
            double const eps = atof(settings["gradient_adam_epsilon"].c_str());
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                GradientDescent* u =
                    dynamic_cast<GradientDescent*>(updaters.at(i));
                u->setParametersAdam(eta, beta1, beta2, eps);
            }
        }
    }
    else if (hasUpdaters && updaterType == UT_KF)
    {
        if (kalmanType == KalmanFilter::KT_STANDARD)
        {
            double const epsilon = atof(settings["kalman_epsilon"].c_str());
            double const q0      = atof(settings["kalman_q0"     ].c_str());
            double const qtau    = atof(settings["kalman_qtau"   ].c_str())
                                 / totalUpdates;
            log << "qtau is divided by number "
                   "of projected updates per epoch.\n";
            double const qmin    = atof(settings["kalman_qmin"   ].c_str());
            double const eta0    = atof(settings["kalman_eta"    ].c_str());
            double etatau  = 1.0;
            double etamax  = eta0;
            if (settings.keywordExists("kalman_etatau") &&
                settings.keywordExists("kalman_etamax"))
            {
                etatau = atof(settings["kalman_etatau"].c_str())
                       / totalUpdates;
                log << "etatau is divided by number "
                       "of projected updates per epoch.\n";
                etamax = atof(settings["kalman_etamax"].c_str());
            }
            for (size_t i = 0; i < updaters.size(); ++i)
            {
                KalmanFilter* u = dynamic_cast<KalmanFilter*>(updaters.at(i));
                u->setParametersStandard(epsilon,
                                         q0,
                                         qtau,
                                         qmin,
                                         eta0,
                                         etatau,
                                         etamax);
            }
        }
        else if (kalmanType == KalmanFilter::KT_FADINGMEMORY)
        {
            double const epsilon = atof(settings["kalman_epsilon"].c_str());
            double const q0      = atof(settings["kalman_q0"     ].c_str());
            double const qtau    = atof(settings["kalman_qtau"   ].c_str())
                                 / totalUpdates;
            log << "qtau is divided by number "
                   "of projected updates per epoch.\n";
            double const qmin   = atof(settings["kalman_qmin"].c_str());
            double const lambda =
                                 atof(settings["kalman_lambda_short"].c_str());
            //double const nu =
            //       pow(atof(settings["kalman_nue_short"].c_str()), numProcs);
            //log << "nu is exponentiated with the number of streams.\n";
            double const nu = atof(settings["kalman_nue_short"].c_str());
            for (size_t i = 0; i < updaters.size(); ++i)
            {
                KalmanFilter* u = dynamic_cast<KalmanFilter*>(updaters.at(i));
                u->setParametersFadingMemory(epsilon,
                                             q0,
                                             qtau,
                                             qmin,
                                             lambda,
                                             nu);
            }
        }
    }

    log << "-----------------------------------------"
           "--------------------------------------\n";
    for (size_t i = 0; i < updaters.size(); ++i)
    {
            if (updateStrategy == US_COMBINED)
            {
                log << strpr("Combined weight updater:\n");
            }
            else if (updateStrategy == US_ELEMENT)
            {
                log << strpr("Weight updater for element %2s :\n",
                             elements.at(i).getSymbol().c_str());
            }
            log << "-----------------------------------------"
                   "--------------------------------------\n";
            log << updaters.at(i)->info();
            if (updaterType == UT_KF)
            {
                log << "Note: During training loop the actual observation\n";
                log << "      size corresponds to error vector size:\n";
                for (auto k : pk)
                {
                    log << strpr("sizeObservation = %zu (%s updates)\n",
                                 p[k].error.at(i).size(), k.c_str());
                }
            }
            log << "-----------------------------------------"
                   "--------------------------------------\n";
    }

    log << strpr("TIMING Finished setup: %.2f seconds.\n",
                 sw["setup"].stop());
    log << "*****************************************"
           "**************************************\n";

    return;
}

void Training::calculateNeighborLists()
{
    sw["nl"].start();
    log << "\n";
    log << "*** CALCULATE NEIGHBOR LISTS ************"
           "**************************************\n";
    log << "\n";

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    log << strpr("Temporarily disabling OpenMP parallelization: %d threads.\n",
                 omp_get_max_threads());
#endif
    log << "Calculating neighbor lists for all structures.\n";
    double maxCutoffRadiusPhys = maxCutoffRadius;
    if (normalize) maxCutoffRadiusPhys = maxCutoffRadius / convLength;
    log << strpr("Cutoff radius for neighbor lists: %f\n",
                 maxCutoffRadiusPhys);
    for (vector<Structure>::iterator it = structures.begin();
         it != structures.end(); ++it)
    {
        it->calculateNeighborList(maxCutoffRadius);
    }
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    log << strpr("Restoring OpenMP parallelization: max. %d threads.\n",
                 omp_get_max_threads());
#endif

    log << "-----------------------------------------"
           "--------------------------------------\n";
    log << strpr("TIMING Finished neighbor lists: %.2f seconds.\n",
                 sw["nl"].stop());
    log << "*****************************************"
           "**************************************\n";

    return;
}

void Training::calculateError(
                             map<string, pair<string, string>> const fileNames)
{
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    vector<string> write;
    for (auto i : fileNames)
    {
        if (i.second.first.size() == 0 || i.second.second.size() == 0)
        {
            throw runtime_error("ERROR: No filename provided for comparison "
                                "files.\n");
        }
        write.push_back(i.first);
    }
    auto doWrite = [&write](string key){
                       return find(write.begin(),
                                   write.end(),
                                   key) != write.end();
                   };


    map<string, size_t> countTrain;
    map<string, size_t> countTest;
    for (auto k : pk) countTrain[k] = 0;
    for (auto k : pk) countTest[k]  = 0;

    map<string, ofstream> filesTrain;
    map<string, ofstream> filesTest;

    // Reset current error metrics.
    for (auto k : pk)
    {
        for (auto& m : p[k].errorTrain) m.second = 0.0;
        for (auto& m : p[k].errorTest) m.second = 0.0;
    }

    for (auto k : write)
    {
        filesTrain[k].open(strpr("%s.%04d",
                                 fileNames.at(k).first.c_str(),
                                 myRank).c_str());
        filesTest[k].open(strpr("%s.%04d",
                                fileNames.at(k).second.c_str(),
                                myRank).c_str());
        // File header.
        vector<string> header;
        if (myRank == 0)
        {
            vector<string> title;
            vector<string> colName;
            vector<string> colInfo;
            vector<size_t> colSize;
            if (k == "energy")
            {
                title.push_back("Energy comparison.");
                colSize.push_back(10);
                colName.push_back("index");
                colInfo.push_back("Structure index.");
                colSize.push_back(16);
                colName.push_back("Eref");
                colInfo.push_back("Reference potential energy per atom "
                                  "(training units).");
                colSize.push_back(16);
                colName.push_back("Ennp");
                colInfo.push_back("NNP potential energy per atom "
                                  "(training units).");
            }
            else if (k == "force")
            {
                title.push_back("Force comparison.");
                colSize.push_back(10);
                colName.push_back("index_s");
                colInfo.push_back("Structure index.");
                colSize.push_back(10);
                colName.push_back("index_a");
                colInfo.push_back("Atom index (x, y, z components in "
                                  "consecutive lines).");
                colSize.push_back(16);
                colName.push_back("Fref");
                colInfo.push_back("Reference force (training units).");
                colSize.push_back(16);
                colName.push_back("Fnnp");
                colInfo.push_back("NNP force (training units).");
            }
            else if (k == "charge")
            {
                title.push_back("Charge comparison.");
                colSize.push_back(10);
                colName.push_back("index_s");
                colInfo.push_back("Structure index.");
                colSize.push_back(10);
                colName.push_back("index_a");
                colInfo.push_back("Atom index.");
                colSize.push_back(16);
                colName.push_back("Qref");
                colInfo.push_back("Reference charge.");
                colSize.push_back(16);
                colName.push_back("Qnnp");
                colInfo.push_back("NNP charge.");
            }
            header = createFileHeader(title, colSize, colName, colInfo);
            appendLinesToFile(filesTrain.at(k), header);
            appendLinesToFile(filesTest.at(k), header);
        }
    }

    for (vector<Structure>::iterator it = structures.begin();
         it != structures.end(); ++it)
    {
#ifdef NNP_NO_SF_GROUPS
        calculateSymmetryFunctions((*it), useForces);
#else
        calculateSymmetryFunctionGroups((*it), useForces);
#endif
        calculateAtomicNeuralNetworks((*it), useForces);
        calculateEnergy((*it));
        if (useForces) calculateForces((*it));
        for (auto k : pk)
        {
            map<string, double>* error = nullptr;
            size_t* count = nullptr;
            ofstream* file = nullptr;
            if (it->sampleType == Structure::ST_TRAINING)
            {
                error = &(p[k].errorTrain);
                count = &(countTrain.at(k));
                if (doWrite(k)) file = &(filesTrain.at(k));
            }
            else if (it->sampleType == Structure::ST_TEST)
            {
                error = &(p[k].errorTest);
                count = &(countTest.at(k));
                if (doWrite(k)) file = &(filesTest.at(k));
            }

            it->updateError(k, *error, *count);
            if (doWrite(k))
            {
                if      (k == "energy") (*file) << it->getEnergyLine();
                else if (k == "force")
                {
                    for (auto l : it->getForcesLines()) (*file) << l;
                }
                else if (k == "charge")
                {
                    for (auto l : it->getChargesLines()) (*file) << l;
                }
            }
        }
        if (freeMemory) it->freeAtoms(true);
    }

    for (auto k : pk)
    {
        collectError(k, p[k].errorTrain, countTrain.at(k));
        collectError(k, p[k].errorTest, countTest.at(k));
        if (doWrite(k))
        {
            filesTrain.at(k).close();
            filesTest.at(k).close();
            MPI_Barrier(comm);
            if (myRank == 0)
            {
                combineFiles(fileNames.at(k).first);
                combineFiles(fileNames.at(k).second);
            }
        }
    }

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

    return;
}

void Training::calculateErrorEpoch()
{
    // Check whether property comparison files should be written for
    // this epoch.
    map<string, pair<string, string>> fileNames;

    for (auto const& ip : p)
    {
        string const& k = ip.first; // key
        Property const& d = ip.second; // data
        if (d.writeCompEvery > 0 &&
            (epoch % d.writeCompEvery == 0 || epoch <= d.writeCompAlways))
        {
            string middle;
            if      (k == "energy") middle = "points";
            else if (k == "force" ) middle = "forces";
            else if (k == "charge") middle = "charges";
            fileNames[k] = make_pair(strpr("train%s.%06zu.out",
                                           middle.c_str(), epoch),
                                     strpr("test%s.%06zu.out",
                                           middle.c_str(), epoch));
        }
    }

    // Calculate errors and write comparison files.
    calculateError(fileNames);

    return;
}

void Training::printHeader()
{
    string metric = "?";
    string peratom = "";

    log << "The training loop output covers different errors, update and\n";
    log << "timing information. The following quantities are organized\n";
    log << "according to the matrix scheme below:\n";
    log << "-------------------------------------------------------------------\n";
    log << "ep ........ Epoch.\n";
    for (auto k : pk)
    {
        string const& pmetric = p[k].displayMetric;
        if      (pmetric.find("RMSE") != pmetric.npos) metric = "RMSE";
        else if (pmetric.find("MAE")  != pmetric.npos) metric = "MAE";
        if      (pmetric.find("pa") != pmetric.npos) peratom = " per atom";
        else peratom = "";
        log << p[k].tiny << "_count ... Number of " << k << " updates.\n";
        log << p[k].tiny << "_train ... " << metric << " of training "
            << p[k].plural << peratom << ".\n";
        log << p[k].tiny << "_test .... " << metric << " of test     "
            << p[k].plural << peratom << ".\n";
        //log << p[k].tiny << "_time ........ Time for " << k << " updates "
        //                 << "(seconds).\n";
        log << p[k].tiny << "_pt ...... Percentage of time for " << k <<
                            " updates w.r.t. to t_train.\n";
    }
    log << "count ..... Total number of updates.\n";
    log << "train ..... Percentage of time for training.\n";
    log << "error ..... Percentage of time for error calculation.\n";
    log << "other ..... Percentage of time for other purposes.\n";
    log << "epoch ..... Total time for this epoch (seconds).\n";
    log << "total ..... Total time for all epochs (seconds).\n";
    log << "-------------------------------------------------------------------\n";
    for (auto k : pk)
    {
        log << strpr("%-6s", k.c_str())
            << strpr("  %5s", "ep")
            << strpr("  %7s", (p[k].tiny + "_count").c_str())
            << strpr("   %11s", (p[k].tiny + "_train").c_str())
            << strpr("   %11s", (p[k].tiny + "_test").c_str())
            << strpr("   %5s", (p[k].tiny + "_pt").c_str())
            << "\n";
    }
    log << strpr("%-6s", "timing")
        << strpr("  %5s", "ep")
        << strpr("  %7s", "count")
        << strpr("  %5s", "train")
        << strpr("  %5s", "error")
        << strpr("  %5s", "other")
        << strpr("  %9s", "epoch")
        << strpr("  %9s", "total")
        << "\n";
    log << "-------------------------------------------------------------------\n";

    return;
}

void Training::printEpoch()
{
    double timeLoop = sw["loop"].getLoop();
    double timeTrain = sw["train"].getLoop();
    size_t totalUpdates = 0;
    for (auto k : pk)
    {
        totalUpdates += p[k].countUpdates;
        double timeProp = sw[k].getLoop();
        string caps = k;
        for (auto& c : caps) c = toupper(c);
        log << strpr("%-6s", caps.c_str());
        log << strpr("  %5zu", epoch);
        log << strpr("  %7zu", p[k].countUpdates);
        if (normalize && (k != "charge"))
        {
            log << strpr("   %11.5E   %11.5E",
                         physical(k, p[k].errorTrain.at(p[k].displayMetric)),
                         physical(k, p[k].errorTest.at(p[k].displayMetric)));
        }
        else
        {
            log << strpr("   %11.5E   %11.5E",
                         p[k].errorTrain.at(p[k].displayMetric),
                         p[k].errorTest.at(p[k].displayMetric));
        }
        if (epoch == 0) log << strpr("   %5.1f", 0.0);
        else log << strpr("   %5.1f", timeProp / timeTrain * 100.0);
        log << "\n";
    }
    double timeOther = timeLoop;
    timeOther -= sw["error"].getLoop();
    timeOther -= sw["train"].getLoop();
    log << strpr("%-6s", "TIMING");
    log << strpr("  %5zu", epoch);
    log << strpr("  %7zu", totalUpdates);
    log << strpr("  %5.1f", sw["train"].getLoop() / timeLoop * 100.0);
    log << strpr("  %5.1f", sw["error"].getLoop() / timeLoop * 100.0);
    log << strpr("  %5.1f", timeOther / timeLoop * 100.0);
    log << strpr("  %9.2f", sw["loop"].getLoop());
    log << strpr("  %9.2f", sw["loop"].getTotal());
    log << "\n";

    return;
}

void Training::writeWeights(string const& nnName,
                            string const& fileNameFormat) const
{
    ofstream file;

    for (size_t i = 0; i < numElements; ++i)
    {
        string fileName = strpr(fileNameFormat.c_str(),
                                elements.at(i).getAtomicNumber());
        file.open(fileName.c_str());
        elements.at(i).neuralNetworks.at(nnName).writeConnections(file);
        file.close();
    }

    return;
}

void Training::writeWeightsEpoch() const
{
    if (writeWeightsEvery > 0 &&
        (epoch % writeWeightsEvery == 0 || epoch <= writeWeightsAlways))
    {
        if (nnpType == NNPType::SHORT_ONLY ||
            (nnpType == NNPType::SHORT_CHARGE_NN && stage == 2))
        {
            writeWeights("short", strpr("weights.%%03zu.%06d.out", epoch));
        }
        else if (nnpType == NNPType::SHORT_CHARGE_NN && stage == 1)
        {
            writeWeights("charge", strpr("weightse.%%03zu.%06d.out", epoch));
        }
    }

    return;
}

void Training::writeLearningCurve(bool append, string const fileName) const
{
    ofstream file;
    string fileNameActual = fileName;
    if (nnpType == NNPType::SHORT_CHARGE_NN)
    {
        fileNameActual += strpr(".stage-%zu", stage);
    }

    if (append) file.open(fileNameActual.c_str(), ofstream::app);
    else
    {
        file.open(fileNameActual.c_str());

        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        if (nnpType == NNPType::SHORT_ONLY ||
            (nnpType == NNPType::SHORT_CHARGE_NN && stage == 2))
        {
            title.push_back("Learning curves for energies and forces.");
        }
        else if (nnpType == NNPType::SHORT_CHARGE_NN && stage == 1)
        {
            title.push_back("Learning curves for charges.");
        }
        colSize.push_back(10);
        colName.push_back("epoch");
        colInfo.push_back("Current epoch.");

        map<string, string> text;
        text["RMSEpa"] = "RMSE of %s %s per atom";
        text["RMSE"]   = "RMSE of %s %s";
        text["MAEpa"]  = "MAE of %s %s per atom";
        text["MAE"]    = "MAE of %s %s";

        for (auto k : pk)
        {
            for (auto m : p[k].errorMetrics)
            {
                colSize.push_back(16);
                colName.push_back(m + "_" + p[k].tiny + "train_pu");
                colInfo.push_back(strpr(
                                       (text[m] + " (physical units)").c_str(),
                                       "training",
                                       p[k].plural.c_str()));
                colSize.push_back(16);
                colName.push_back(m + "_" + p[k].tiny + "test_pu");
                colInfo.push_back(strpr(
                                       (text[m] + " (physical units)").c_str(),
                                       "test",
                                       p[k].plural.c_str()));
            }
        }
        if (normalize)
        {
            for (auto k : pk)
            {
                // Internal units only for energies and forces.
                if (!(k == "energy" || k == "force")) continue;
                for (auto m : p[k].errorMetrics)
                {
                    colSize.push_back(16);
                    colName.push_back(m + "_" + p[k].tiny + "train_iu");
                    colInfo.push_back(strpr(
                                       (text[m] + " (internal units)").c_str(),
                                       "training",
                                       p[k].plural.c_str()));
                    colSize.push_back(16);
                    colName.push_back(m + "_" + p[k].tiny + "test_iu");
                    colInfo.push_back(strpr(
                                       (text[m] + " (internal units)").c_str(),
                                       "test",
                                       p[k].plural.c_str()));
                }
            }
        }
        appendLinesToFile(file,
                          createFileHeader(title, colSize, colName, colInfo));
    }

    file << strpr("%10zu", epoch);
    if (normalize)
    {
        for (auto k : pk)
        {
            if (!(k == "energy" || k == "force")) continue;
            for (auto m : p[k].errorMetrics)
            {
                file << strpr(" %16.8E %16.8E",
                              physical(k, p[k].errorTrain.at(m)),
                              physical(k, p[k].errorTest.at(m)));
            }
        }
    }
    for (auto k : pk)
    {
        for (auto m : p[k].errorMetrics)
        {
            file << strpr(" %16.8E %16.8E",
                          p[k].errorTrain.at(m),
                          p[k].errorTest.at(m));
        }
    }
    file << "\n";
    file.flush();
    file.close();

    return;
}

void Training::writeNeuronStatistics(string const& nnName,
                                     string const& fileName) const
{
    ofstream file;
    if (myRank == 0)
    {
        file.open(fileName.c_str());

        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back("Statistics for individual neurons of network \""
                        + nnName + "\" gathered during RMSE calculation.");
        colSize.push_back(10);
        colName.push_back("element");
        colInfo.push_back("Element index.");
        colSize.push_back(10);
        colName.push_back("neuron");
        colInfo.push_back("Neuron number.");
        colSize.push_back(10);
        colName.push_back("count");
        colInfo.push_back("Number of neuron value computations.");
        colSize.push_back(16);
        colName.push_back("min");
        colInfo.push_back("Minimum neuron value encounterd.");
        colSize.push_back(16);
        colName.push_back("max");
        colInfo.push_back("Maximum neuron value encounterd.");
        colSize.push_back(16);
        colName.push_back("mean");
        colInfo.push_back("Mean neuron value.");
        colSize.push_back(16);
        colName.push_back("sigma");
        colInfo.push_back("Standard deviation of neuron value.");
        appendLinesToFile(file,
                          createFileHeader(title, colSize, colName, colInfo));
    }

    for (size_t i = 0; i < numElements; ++i)
    {
        size_t n = elements.at(i).neuralNetworks.at(nnName).getNumNeurons();
        vector<long>   count(n, 0);
        vector<double> min(n, 0.0);
        vector<double> max(n, 0.0);
        vector<double> mean(n, 0.0);
        vector<double> sigma(n, 0.0);
        elements.at(i).neuralNetworks.at(nnName).
            getNeuronStatistics(&(count.front()),
                                &(min.front()),
                                &(max.front()),
                                &(mean.front()),
                                &(sigma.front()));
        // Collect statistics from all processors on proc 0.
        if (myRank == 0)
        {
            MPI_Reduce(MPI_IN_PLACE, &(count.front()), n, MPI_LONG  , MPI_SUM, 0, comm);
            MPI_Reduce(MPI_IN_PLACE, &(min.front())  , n, MPI_DOUBLE, MPI_MIN, 0, comm);
            MPI_Reduce(MPI_IN_PLACE, &(max.front())  , n, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(MPI_IN_PLACE, &(mean.front()) , n, MPI_DOUBLE, MPI_SUM, 0, comm);
            MPI_Reduce(MPI_IN_PLACE, &(sigma.front()), n, MPI_DOUBLE, MPI_SUM, 0, comm);
        }
        else
        {
            MPI_Reduce(&(count.front()), &(count.front()), n, MPI_LONG  , MPI_SUM, 0, comm);
            MPI_Reduce(&(min.front())  , &(min.front())  , n, MPI_DOUBLE, MPI_MIN, 0, comm);
            MPI_Reduce(&(max.front())  , &(max.front())  , n, MPI_DOUBLE, MPI_MAX, 0, comm);
            MPI_Reduce(&(mean.front()) , &(mean.front()) , n, MPI_DOUBLE, MPI_SUM, 0, comm);
            MPI_Reduce(&(sigma.front()), &(sigma.front()), n, MPI_DOUBLE, MPI_SUM, 0, comm);
        }
        if (myRank == 0)
        {
            for (size_t j = 0; j < n; ++j)
            {
                size_t m = count.at(j);
                sigma.at(j) = sqrt((m * sigma.at(j) - mean.at(j) * mean.at(j))
                            / (m * (m - 1)));
                mean.at(j) /= m;
                file << strpr("%10d %10d %10d %16.8E %16.8E %16.8E %16.8E\n",
                              i + 1,
                              j + 1,
                              count[j],
                              min[j],
                              max[j],
                              mean[j],
                              sigma[j]);
            }
        }
    }

    if (myRank == 0)
    {
        file.close();
    }

    return;
}

void Training::writeNeuronStatisticsEpoch() const
{
    vector<pair<string, string>> nnInfo; // NN name and file name.

    if (writeNeuronStatisticsEvery > 0 &&
        (epoch % writeNeuronStatisticsEvery == 0
        || epoch <= writeNeuronStatisticsAlways))
    {
        if (nnpType == NNPType::SHORT_ONLY)
        {
            nnInfo.push_back(
                make_pair("short", strpr("neuron-stats.%06zu.out", epoch)));
        }
        else if (nnpType == NNPType::SHORT_CHARGE_NN)
        {
            if (stage == 1)
            {
                nnInfo.push_back(
                    make_pair("charge",
                              strpr("neuron-stats.%s.%06zu.out.stage-%zu",
                                    "charge", epoch, stage)));
            }
            else if (stage == 2)
            {
                nnInfo.push_back(
                    make_pair("charge",
                              strpr("neuron-stats.%s.%06zu.out.stage-%zu",
                                    "charge", epoch, stage)));
                nnInfo.push_back(
                    make_pair("short",
                              strpr("neuron-stats.%s.%06zu.out.stage-%zu",
                                    "short", epoch, stage)));
            }
        }
        for (auto const& i : nnInfo) writeNeuronStatistics(i.first, i.second);
    }

    return;
}

void Training::resetNeuronStatistics()
{
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        for (auto& nn : it->neuralNetworks) nn.second.resetNeuronStatistics();
    }
    return;
}

void Training::writeUpdaterStatus(bool         append,
                                  string const fileNameFormat) const
{
    ofstream file;
    string fileNameFormatActual = fileNameFormat;
    if (nnpType == NNPType::SHORT_CHARGE_NN)
    {
        fileNameFormatActual += strpr(".stage-%zu", stage);
    }

    for (size_t i = 0; i < numUpdaters; ++i)
    {
        string fileName;
        if (updateStrategy == US_COMBINED)
        {
            fileName = strpr(fileNameFormatActual.c_str(), 0);
        }
        else if (updateStrategy == US_ELEMENT)
        {
            fileName = strpr(fileNameFormatActual.c_str(),
                             elementMap.atomicNumber(i));
        }
        if (append) file.open(fileName.c_str(), ofstream::app);
        else
        {
            file.open(fileName.c_str());
            appendLinesToFile(file, updaters.at(i)->statusHeader());
        }
        file << updaters.at(i)->status(epoch);
        file.close();
    }

    return;
}

void Training::sortUpdateCandidates(string const& property)
{
    // Update error for all structures.
    for (auto& uc : p[property].updateCandidates)
    {
        if (property == "energy")
        {
            Structure const& s = structures.at(uc.s);
            uc.error = fabs((s.energyRef - s.energy) / s.numAtoms);
        }
        else if (property == "force")
        {
            Atom const& a = structures.at(uc.s).atoms.at(uc.a);
            uc.error = fabs(a.fRef[uc.c] - a.f[uc.c]);
        }
        else if (property == "charge")
        {
            Atom const& a = structures.at(uc.s).atoms.at(uc.a);
            uc.error = fabs(a.chargeRef - a.charge);
        }
    }
    // Sort update candidates list.
    sort(p[property].updateCandidates.begin(),
         p[property].updateCandidates.end());
    // Reset current position.
    p[property].posUpdateCandidates = 0;

    return;
}

void Training::shuffleUpdateCandidates(string const& property)
{
    shuffle(p[property].updateCandidates.begin(),
            p[property].updateCandidates.end(),
            rngNew);
    // Reset current position.
    p[property].posUpdateCandidates = 0;

    return;
}

void Training::setEpochSchedule()
{
    // Clear epoch schedule.
    epochSchedule.clear();
    vector<int>(epochSchedule).swap(epochSchedule);

    // Grow schedule vector by each property's number of desired updates.
    // Fill this array looping in reverse direction for backward compatibility.
    //for (size_t i = 0; i < pk.size(); ++i)
    for (int i = pk.size() - 1; i >= 0; --i)
    {
        epochSchedule.insert(epochSchedule.end(), p[pk.at(i)].numUpdates, i);
    }

    // Return if there is only a single training property.
    if (pk.size() == 1) return;

    // Now shuffle the schedule to get a random sequence.
    shuffle(epochSchedule.begin(), epochSchedule.end(), rngGlobalNew);

    //for (size_t i = 0; i < epochSchedule.size(); ++i)
    //{
    //    log << strpr("%zu %zu\n", i, epochSchedule.at(i));
    //}

    return;
}

void Training::checkSelectionMode()
{
    for (auto k : pk)
    {
        if (p[k].selectionModeSchedule.find(epoch)
            != p[k].selectionModeSchedule.end())
        {
            p[k].selectionMode = p[k].selectionModeSchedule[epoch];
            if (epoch != 0)
            {
                string message = "INFO   Switching selection mode for "
                                 "property \"" + k + "\" to ";
                if (p[k].selectionMode == SM_RANDOM)
                {
                    message += strpr("SM_RANDOM (%d).\n", p[k].selectionMode);
                }
                else if (p[k].selectionMode == SM_SORT)
                {
                    message += strpr("SM_SORT (%d).\n", p[k].selectionMode);
                }
                else if (p[k].selectionMode == SM_THRESHOLD)
                {
                    message += strpr("SM_THRESHOLD (%d).\n",
                                     p[k].selectionMode);
                }
                log << message;
            }
        }
    }

    return;
}

void Training::loop()
{
    sw["loop"].start();
    log << "\n";
    log << "*** TRAINING LOOP ***********************"
           "**************************************\n";
    log << "\n";
    printHeader();

    // Calculate initial RMSE and write comparison files.
    sw["error"].start();
    calculateErrorEpoch();
    sw["error"].stop();

    // Write initial weights to files.
    if (myRank == 0) writeWeightsEpoch();

    // Write learning curve.
    if (myRank == 0) writeLearningCurve(false);

    // Write updater status to file.
    if (myRank == 0) writeUpdaterStatus(false);

    // Write neuron statistics.
    writeNeuronStatisticsEpoch();

    // Print timing information.
    sw["loop"].stop();
    printEpoch();

    // Check if training should be continued.
    while (advance())
    {
        sw["loop"].start();

        // Increment epoch counter.
        epoch++;
        log << "------\n";

        // Reset update counters.
        for (auto k : pk) p[k].countUpdates = 0;

        // Check if selection mode should be changed in this epoch.
        checkSelectionMode();

        // Sort or shuffle update candidates.
        for (auto k : pk)
        {
            if (p[k].selectionMode == SM_SORT) sortUpdateCandidates(k);
            else shuffleUpdateCandidates(k);
        }

        // Determine epoch update schedule.
        setEpochSchedule();

        // Perform property updates according to schedule.
        sw["train"].start();
        for (auto i : epochSchedule)
        {
            string property = pk.at(i);
            update(property);
            p[property].countUpdates++;
        }
        sw["train"].stop();

        // Reset neuron statistics.
        resetNeuronStatistics();

        // Calculate errors and write comparison files.
        sw["error"].start();
        calculateErrorEpoch();
        sw["error"].stop();

        // Write weights to files.
        if (myRank == 0) writeWeightsEpoch();

        // Append to learning curve.
        if (myRank == 0) writeLearningCurve(true);

        // Write updater status to file.
        if (myRank == 0) writeUpdaterStatus(true);

        // Write neuron statistics.
        writeNeuronStatisticsEpoch();

        // Print error overview and timing information.
        sw["loop"].stop();
        printEpoch();

        if (myRank == 0) writeTimingData(epoch != 1);
    }

    log << "-----------------------------------------"
           "--------------------------------------\n";
    log << strpr("TIMING Training loop finished: %.2f seconds.\n",
                 sw["loop"].getTotal());
    log << "*****************************************"
           "**************************************\n";

    return;
}

void Training::update(string const& property)
{
    // Shortcuts.
    string const& k = property; // Property key.
    Property& pu = p[k]; // Update property.
    // Start watch for error and jacobian computation, reset loop timer if
    // first update in this epoch.
    bool newLoop = pu.countUpdates == 0;
    sw[k].start(newLoop);
    sw[k + "_err"].start(newLoop);

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif

    ///////////////////////////////////////////////////////////////////////////
    // PART 1: Find update candidate, compute error fractions and derivatives
    ///////////////////////////////////////////////////////////////////////////

    size_t batchSize = pu.taskBatchSize;
    if (batchSize == 0) batchSize = pu.patternsPerUpdate;
    bool derivatives = false;
    if (k == "force") derivatives = true;

    vector<size_t> thresholdLoopCount(batchSize, 0);
    vector<double> currentRmseFraction(batchSize, 0.0);
    vector<UpdateCandidate*> currentUpdateCandidates(batchSize, NULL);
    for (size_t i = 0; i < numUpdaters; ++i)
    {
        fill(pu.error.at(i).begin(), pu.error.at(i).end(), 0.0);
        fill(pu.jacobian.at(i).begin(), pu.jacobian.at(i).end(), 0.0);
    }

    // Loop over (mini-)batch size.
    for (size_t b = 0; b < batchSize; ++b)
    {
        UpdateCandidate* c = NULL; // Actual current update candidate.
        size_t indexBest = 0; // Index of best update candidate so far.
        double rmseFractionBest = 0.0; // RMSE of best update candidate so far.

        // For SM_THRESHOLD need to loop until candidate's RMSE is above
        // threshold. Other modes don't loop here.
        size_t trials = 1;
        if (pu.selectionMode == SM_THRESHOLD) trials = pu.rmseThresholdTrials;
        size_t il = 0;
        for (il = 0; il < trials; ++il)
        {
            // Restart position index if necessary.
            if (pu.posUpdateCandidates >= pu.updateCandidates.size())
            {
                pu.posUpdateCandidates = 0;
            }

            //log << strpr("pos %zu b %zu size %zu\n", pu.posUpdateCandidates, b, currentUpdateCandidates.size());
            // Set current update candidate.
            c = &(pu.updateCandidates.at(pu.posUpdateCandidates));
            // Keep update candidates (for logging later).
            currentUpdateCandidates.at(b) = c;
            // Shortcut for current structure.
            Structure& s = structures.at(c->s);
            // Calculate symmetry functions (if results are already stored
            // these functions will return immediately).
#ifdef NOSFGROUPS
            calculateSymmetryFunctions(s, derivatives);
#else
            calculateSymmetryFunctionGroups(s, derivatives);
#endif
            // For SM_THRESHOLD calculate RMSE of update candidate.
            if (pu.selectionMode == SM_THRESHOLD)
            {
                if (k == "energy")
                {
                    if (nnpType == NNPType::SHORT_ONLY)
                    {
                        calculateAtomicNeuralNetworks(s, derivatives);
                        calculateEnergy(s);
                        currentRmseFraction.at(b) =
                            fabs(s.energyRef - s.energy)
                            / (s.numAtoms * pu.errorTrain.at("RMSEpa"));
                    }
                    // Assume stage 2.
                    else if (nnpType == NNPType::SHORT_CHARGE_NN)
                    {
                        // TODO: Reuse already present charge-NN data and
                        // compute only short-NN energy contributions.
                        throw runtime_error("ERROR: Not implemented.\n");
                    }
                }
                else if (k == "force")
                {
                    if (nnpType == NNPType::SHORT_ONLY)
                    {
                        calculateAtomicNeuralNetworks(s, derivatives);
                        calculateForces(s);
                        Atom const& a = s.atoms.at(c->a);
                        currentRmseFraction.at(b) =
                            fabs(a.fRef[c->c] - a.f[c->c])
                            / pu.errorTrain.at("RMSE");
                    }
                    // Assume stage 2.
                    else if (nnpType == NNPType::SHORT_CHARGE_NN)
                    {
                        // TODO: Reuse already present charge-NN data and
                        // compute only short-NN force contributions.
                        throw runtime_error("ERROR: Not implemented.\n");
                    }
                }
                else if (k == "charge")
                {
                    // Assume NNPType::SHORT_CHARGE_NN stage 1.
                    // Compute only charge-NN
                    Atom& a = s.atoms.at(c->a);
                    NeuralNetwork& nn =
                        elements.at(a.element).neuralNetworks.at("charge");
                    nn.setInput(&(a.G.front()));
                    nn.propagate();
                    nn.getOutput(&(a.charge));
                    currentRmseFraction.at(b) = fabs(a.chargeRef - a.charge)
                                              / pu.errorTrain.at("RMSE");
                }
                // If force RMSE is above threshold stop loop immediately.
                if (currentRmseFraction.at(b) > pu.rmseThreshold)
                {
                    // Increment position in update candidate list.
                    pu.posUpdateCandidates++;
                    break;
                }
                // If loop continues, free memory and remember best candidate
                // so far.
                if (freeMemory)
                {
                    s.freeAtoms(true);
                }
                if (currentRmseFraction.at(b) > rmseFractionBest)
                {
                    rmseFractionBest = currentRmseFraction.at(b);
                    indexBest = pu.posUpdateCandidates;
                }
                // Increment position in update candidate list.
                pu.posUpdateCandidates++;
            }
            // Break loop for all selection modes but SM_THRESHOLD.
            else if (pu.selectionMode == SM_RANDOM ||
                     pu.selectionMode == SM_SORT)
            {
                // Increment position in update candidate list.
                pu.posUpdateCandidates++;
                break;
            }
        }
        thresholdLoopCount.at(b) = il;

        // If loop was not stopped because of a proper update candidate found
        // (RMSE above threshold) use best candidate during iteration.
        if (pu.selectionMode == SM_THRESHOLD && il == trials)
        {
            // Set best candidate.
            currentUpdateCandidates.at(b) =
                &(pu.updateCandidates.at(indexBest));
            currentRmseFraction.at(b) = rmseFractionBest;
            // Need to calculate the symmetry functions again, maybe results
            // were not stored.
            Structure& s = structures.at(c->s);
#ifdef NNP_NO_SF_GROUPS
            calculateSymmetryFunctions(s, derivatives);
#else
            calculateSymmetryFunctionGroups(s, derivatives);
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        // PART 2: Compute error vector and Jacobian
        ///////////////////////////////////////////////////////////////////////

        Structure& s = structures.at(c->s);
        // Temporary storage for derivative contributions of atoms (dXdc stores
        // dEdc, dFdc or dQdc for energy, force or charge update, respectively.
        vector<vector<double>> dXdc;
        dXdc.resize(numElements);
        for (size_t i = 0; i < numElements; ++i)
        {
            size_t n = elements.at(i).neuralNetworks.at(nnId)
                       .getNumConnections();
            dXdc.at(i).resize(n, 0.0);
        }
        // Precalculate offset in Jacobian array.
        size_t iu = 0;
        vector<size_t> offset(numElements, 0);
        for (size_t i = 0; i < numElements; ++i)
        {
            if (updateStrategy == US_ELEMENT) iu = i;
            else iu = 0;
            if (parallelMode == PM_TRAIN_ALL && jacobianMode != JM_SUM)
            {
                offset.at(i) += pu.offsetPerTask.at(myRank)
                              * numWeightsPerUpdater.at(iu);
                //log << strpr("%zu os 1: %zu ", i, offset.at(i));
            }
            if (jacobianMode == JM_FULL)
            {
                offset.at(i) += b * numWeightsPerUpdater.at(iu);
                //log << strpr("%zu os 2: %zu ", i, offset.at(i));
            }
            if (updateStrategy == US_COMBINED)
            {
                offset.at(i) += weightsOffset.at(i);
                //log << strpr("%zu os 3: %zu", i, offset.at(i));
            }
            //log << strpr(" %zu final os: %zu\n", i, offset.at(i));
        }
        // Now compute Jacobian.
        if (k == "energy")
        {
            if (nnpType == NNPType::SHORT_ONLY)
            {
                // Loop over atoms and calculate atomic energy contributions.
                for (vector<Atom>::iterator it = s.atoms.begin();
                     it != s.atoms.end(); ++it)
                {
                    size_t i = it->element;
                    NeuralNetwork& nn = elements.at(i).neuralNetworks.at(nnId);
                    nn.setInput(&((it->G).front()));
                    nn.propagate();
                    nn.getOutput(&(it->energy));
                    // Compute derivative of output node with respect to all
                    // neural network connections (weights + biases).
                    nn.calculateDEdc(&(dXdc.at(i).front()));
                    // Finally sum up Jacobian.
                    if (updateStrategy == US_ELEMENT) iu = i;
                    else iu = 0;
                    for (size_t j = 0; j < dXdc.at(i).size(); ++j)
                    {
                        pu.jacobian.at(iu).at(offset.at(i) + j) +=
                            dXdc.at(i).at(j);
                    }
                }
            }
            else if (nnpType == NNPType::SHORT_CHARGE_NN)
            {
                throw runtime_error("ERROR: Not implemented.\n");
            }
        }
        else if (k == "force")
        {
            if (nnpType == NNPType::SHORT_ONLY)
            {
                // Loop over atoms and calculate atomic energy contributions.
                for (vector<Atom>::iterator it = s.atoms.begin();
                     it != s.atoms.end(); ++it)
                {
                    // For force update save derivative of symmetry function
                    // with respect to coordinate.
#ifndef NNP_FULL_SFD_MEMORY
                    collectDGdxia((*it), c->a, c->c);
#else
                    it->collectDGdxia(c->a, c->c);
#endif
                    size_t i = it->element;
                    NeuralNetwork& nn = elements.at(i).neuralNetworks.at(nnId);
                    nn.setInput(&((it->G).front()));
                    nn.propagate();
                    if (derivatives) nn.calculateDEdG(&((it->dEdG).front()));
                    nn.getOutput(&(it->energy));
                    // Compute derivative of output node with respect to all
                    // neural network connections (weights + biases).
#ifndef NNP_FULL_SFD_MEMORY
                    nn.calculateDFdc(&(dXdc.at(i).front()),
                                     &(dGdxia.front()));
#else
                    nn.calculateDFdc(&(dXdc.at(i).front()),
                                     &(it->dGdxia.front()));
#endif
                    // Finally sum up Jacobian.
                    if (updateStrategy == US_ELEMENT) iu = i;
                    else iu = 0;
                    for (size_t j = 0; j < dXdc.at(i).size(); ++j)
                    {
                        pu.jacobian.at(iu).at(offset.at(i) + j) +=
                            dXdc.at(i).at(j);
                    }
                }

            }
            else if (nnpType == NNPType::SHORT_CHARGE_NN)
            {
                throw runtime_error("ERROR: Not implemented.\n");
            }
        }
        else if (k == "charge")
        {
            // Assume NNPType::SHORT_CHARGE_NN stage 1.
            // Shortcut to selected atom.
            Atom& a = s.atoms.at(c->a);
            size_t i = a.element;
            NeuralNetwork& nn = elements.at(i).neuralNetworks.at(nnId);
            nn.setInput(&(a.G.front()));
            nn.propagate();
            nn.getOutput(&(a.charge));
            // Compute derivative of output node with respect to all
            // neural network connections (weights + biases).
            nn.calculateDEdc(&(dXdc.at(i).front()));
            // Finally sum up Jacobian.
            if (updateStrategy == US_ELEMENT) iu = i;
            else iu = 0;
            for (size_t j = 0; j < dXdc.at(i).size(); ++j)
            {
                pu.jacobian.at(iu).at(offset.at(i) + j) +=
                    dXdc.at(i).at(j);
            }
        }

        // Sum up total potential energy or calculate force.
        if (k == "energy")
        {
            calculateEnergy(s);
            currentRmseFraction.at(b) = fabs(s.energyRef - s.energy)
                                      / (s.numAtoms
                                         * pu.errorTrain.at("RMSEpa"));
        }
        else if (k == "force")
        {
            calculateForces(s);
            Atom const& a = s.atoms.at(c->a);
            currentRmseFraction.at(b) = fabs(a.fRef[c->c] - a.f[c->c])
                                      / pu.errorTrain.at("RMSE");
        }
        else if (k == "charge")
        {
            Atom const& a = s.atoms.at(c->a);
            currentRmseFraction.at(b) = fabs(a.chargeRef - a.charge)
                                      / pu.errorTrain.at("RMSE");
        }

        // Now symmetry function memory is not required any more for this
        // update.
        if (freeMemory) s.freeAtoms(true);

        // Precalculate offset in error array.
        size_t offset2 = 0;
        if (parallelMode == PM_TRAIN_ALL && jacobianMode != JM_SUM)
        {
            offset2 += pu.offsetPerTask.at(myRank);
            //log << strpr("os 4: %zu ", offset2);
        }
        if (jacobianMode == JM_FULL)
        {
            offset2 += b;
            //log << strpr("os 5: %zu ", offset2);
        }
        //log << strpr(" final os: %zu\n", offset2);


        // Compute error vector (depends on update strategy).
        if (updateStrategy == US_COMBINED)
        {
            if (k == "energy")
            {
                pu.error.at(0).at(offset2) += s.energyRef - s.energy;
            }
            else if (k == "force")
            {
                Atom const& a = s.atoms.at(c->a);
                pu.error.at(0).at(offset2) +=  a.fRef[c->c] - a.f[c->c];
            }
            else if (k == "charge")
            {
                Atom const& a = s.atoms.at(c->a);
                pu.error.at(0).at(offset2) +=  a.chargeRef - a.charge;
            }
        }
        else if (updateStrategy == US_ELEMENT)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                if (k == "energy")
                {
                    pu.error.at(i).at(offset2) += (s.energyRef - s.energy)
                                               * s.numAtomsPerElement.at(i)
                                               / s.numAtoms;
                }
                else if (k == "force")
                {
                    Atom const& a = s.atoms.at(c->a);
                    pu.error.at(i).at(offset2) += (a.fRef[c->c] - a.f[c->c])
                                               * a.numNeighborsPerElement.at(i)
                                               / a.numNeighbors;
                }
                else if (k == "charge")
                {
                    Atom const& a = s.atoms.at(c->a);
                    pu.error.at(i).at(offset2) += a.chargeRef - a.charge;
                }
            }
        }
    }

    // Apply force update weight to error and Jacobian.
    if (k == "force")
    {
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            for (size_t j = 0; j < pu.error.at(i).size(); ++j)
            {
                pu.error.at(i).at(j) *= forceWeight;
            }
            for (size_t j = 0; j < pu.jacobian.at(i).size(); ++j)
            {
                pu.jacobian.at(i).at(j) *= forceWeight;
            }
        }
    }
    sw[k + "_err"].stop();

    ///////////////////////////////////////////////////////////////////////
    // PART 3: Communicate error and Jacobian.
    ///////////////////////////////////////////////////////////////////////

    sw[k + "_com"].start(newLoop);
    if (jacobianMode == JM_SUM)
    {
        if (parallelMode == PM_TRAIN_RK0)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                if (myRank == 0) MPI_Reduce(MPI_IN_PLACE             , &(pu.error.at(i).front()), 1, MPI_DOUBLE, MPI_SUM, 0, comm);
                else             MPI_Reduce(&(pu.error.at(i).front()), &(pu.error.at(i).front()), 1, MPI_DOUBLE, MPI_SUM, 0, comm);
                if (myRank == 0) MPI_Reduce(MPI_IN_PLACE                , &(pu.jacobian.at(i).front()), numWeightsPerUpdater.at(i), MPI_DOUBLE, MPI_SUM, 0, comm);
                else             MPI_Reduce(&(pu.jacobian.at(i).front()), &(pu.jacobian.at(i).front()), numWeightsPerUpdater.at(i), MPI_DOUBLE, MPI_SUM, 0, comm);
            }
        }
        else if (parallelMode == PM_TRAIN_ALL)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                MPI_Allreduce(MPI_IN_PLACE, &(pu.error.at(i).front()), 1, MPI_DOUBLE, MPI_SUM, comm);
                MPI_Allreduce(MPI_IN_PLACE, &(pu.jacobian.at(i).front()), numWeightsPerUpdater.at(i), MPI_DOUBLE, MPI_SUM, comm);
            }
        }
    }
    else if (jacobianMode == JM_TASK)
    {
        if (parallelMode == PM_TRAIN_RK0)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                if (myRank == 0) MPI_Gather(MPI_IN_PLACE             , 1, MPI_DOUBLE, &(pu.error.at(i).front()),  1, MPI_DOUBLE, 0, comm);
                else             MPI_Gather(&(pu.error.at(i).front()), 1, MPI_DOUBLE, NULL                     ,  1, MPI_DOUBLE, 0, comm);
                if (myRank == 0) MPI_Gather(MPI_IN_PLACE                , numWeightsPerUpdater.at(i), MPI_DOUBLE, &(pu.jacobian.at(i).front()), numWeightsPerUpdater.at(i), MPI_DOUBLE, 0, comm);
                else             MPI_Gather(&(pu.jacobian.at(i).front()), numWeightsPerUpdater.at(i), MPI_DOUBLE, NULL                        , numWeightsPerUpdater.at(i), MPI_DOUBLE, 0, comm);
            }
        }
        else if (parallelMode == PM_TRAIN_ALL)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                MPI_Allgather(MPI_IN_PLACE, 1, MPI_DOUBLE, &(pu.error.at(i).front()),  1, MPI_DOUBLE, comm);
                MPI_Allgather(MPI_IN_PLACE, numWeightsPerUpdater.at(i), MPI_DOUBLE, &(pu.jacobian.at(i).front()), numWeightsPerUpdater.at(i), MPI_DOUBLE, comm);
            }
        }
    }
    else if (jacobianMode == JM_FULL)
    {
        if (parallelMode == PM_TRAIN_RK0)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                if (myRank == 0) MPI_Gatherv(MPI_IN_PLACE             , 0                          , MPI_DOUBLE, &(pu.error.at(i).front()), &(pu.errorsPerTask.front()), &(pu.offsetPerTask.front()), MPI_DOUBLE, 0, comm);
                else             MPI_Gatherv(&(pu.error.at(i).front()), pu.errorsPerTask.at(myRank), MPI_DOUBLE, NULL                     , NULL                       , NULL                       , MPI_DOUBLE, 0, comm);
                if (myRank == 0) MPI_Gatherv(MPI_IN_PLACE                , 0                                 , MPI_DOUBLE, &(pu.jacobian.at(i).front()), &(pu.weightsPerTask.at(i).front()), &(pu.offsetJacobian.at(i).front()), MPI_DOUBLE, 0, comm);
                else             MPI_Gatherv(&(pu.jacobian.at(i).front()), pu.weightsPerTask.at(i).at(myRank), MPI_DOUBLE, NULL                        , NULL                              , NULL                              , MPI_DOUBLE, 0, comm);
            }
        }
        else if (parallelMode == PM_TRAIN_ALL)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, &(pu.error.at(i).front()), &(pu.errorsPerTask.front()), &(pu.offsetPerTask.front()), MPI_DOUBLE, comm);
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, &(pu.jacobian.at(i).front()), &(pu.weightsPerTask.at(i).front()), &(pu.offsetJacobian.at(i).front()), MPI_DOUBLE, comm);
            }
        }
    }
    sw[k + "_com"].stop();

    ///////////////////////////////////////////////////////////////////////
    // PART 4: Perform weight update and apply new weights.
    ///////////////////////////////////////////////////////////////////////

    sw[k + "_upd"].start(newLoop);
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif
    // Loop over all updaters.
    for (size_t i = 0; i < updaters.size(); ++i)
    {
        updaters.at(i)->setError(&(pu.error.at(i).front()),
                                 pu.error.at(i).size());
        updaters.at(i)->setJacobian(&(pu.jacobian.at(i).front()),
                                    pu.error.at(i).size());
        if (updaterType == UT_KF)
        {
            KalmanFilter* kf = dynamic_cast<KalmanFilter*>(updaters.at(i));
            kf->setSizeObservation(pu.error.at(i).size());
        }
        updaters.at(i)->update();
    }
    countUpdates++;

    // Redistribute weights to all MPI tasks.
    if (parallelMode == PM_TRAIN_RK0)
    {
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            MPI_Bcast(&(weights.at(i).front()), weights.at(i).size(), MPI_DOUBLE, 0, comm);
        }
    }

    // Set new weights in neural networks.
    setWeights();
    sw[k + "_upd"].stop();

    ///////////////////////////////////////////////////////////////////////
    // PART 5: Communicate candidates and RMSE fractions and write log.
    ///////////////////////////////////////////////////////////////////////

    sw[k + "_log"].start(newLoop);
    if (writeTrainingLog)
    {
        vector<int>    procUpdateCandidate;
        vector<size_t> indexStructure;
        vector<size_t> indexStructureGlobal;
        vector<size_t> indexAtom;
        vector<size_t> indexCoordinate;

        vector<int> currentUpdateCandidatesPerTask;
        vector<int> currentUpdateCandidatesOffset;
        int myCurrentUpdateCandidates = currentUpdateCandidates.size();

        if (myRank == 0)
        {
            currentUpdateCandidatesPerTask.resize(numProcs, 0);
            currentUpdateCandidatesPerTask.at(0) = myCurrentUpdateCandidates;
        }
        if (myRank == 0) MPI_Gather(MPI_IN_PLACE                , 1, MPI_INT, &(currentUpdateCandidatesPerTask.front()),  1, MPI_INT, 0, comm);
        else             MPI_Gather(&(myCurrentUpdateCandidates), 1, MPI_INT, NULL                                     ,  1, MPI_INT, 0, comm);

        if (myRank == 0)
        {
            int totalUpdateCandidates = 0;
            for (size_t i = 0; i < currentUpdateCandidatesPerTask.size(); ++i)
            {
                currentUpdateCandidatesOffset.push_back(totalUpdateCandidates);
                totalUpdateCandidates += currentUpdateCandidatesPerTask.at(i);
            }
            procUpdateCandidate.resize(totalUpdateCandidates, 0);
            indexStructure.resize(totalUpdateCandidates, 0);
            indexStructureGlobal.resize(totalUpdateCandidates, 0);
            indexAtom.resize(totalUpdateCandidates, 0);
            indexCoordinate.resize(totalUpdateCandidates, 0);
            // Increase size of this vectors (only rank 0).
            currentRmseFraction.resize(totalUpdateCandidates, 0.0);
            thresholdLoopCount.resize(totalUpdateCandidates, 0.0);
        }
        else
        {
            procUpdateCandidate.resize(myCurrentUpdateCandidates, 0);
            indexStructure.resize(myCurrentUpdateCandidates, 0);
            indexStructureGlobal.resize(myCurrentUpdateCandidates, 0);
            indexAtom.resize(myCurrentUpdateCandidates, 0);
            indexCoordinate.resize(myCurrentUpdateCandidates, 0);
        }
        for (int i = 0; i < myCurrentUpdateCandidates; ++i)
        {
            procUpdateCandidate.at(i) = myRank;
            UpdateCandidate& c = *(currentUpdateCandidates.at(i));
            indexStructure.at(i) = c.s;
            indexStructureGlobal.at(i) = structures.at(c.s).index;
            indexAtom.at(i) = c.a;
            indexCoordinate.at(i) = c.c;
        }
        if (myRank == 0)
        {
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, &(currentRmseFraction.front()) , &(currentUpdateCandidatesPerTask.front()), &(currentUpdateCandidatesOffset.front()), MPI_DOUBLE, 0, comm);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_SIZE_T, &(thresholdLoopCount.front())  , &(currentUpdateCandidatesPerTask.front()), &(currentUpdateCandidatesOffset.front()), MPI_SIZE_T, 0, comm);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_INT   , &(procUpdateCandidate.front()) , &(currentUpdateCandidatesPerTask.front()), &(currentUpdateCandidatesOffset.front()), MPI_INT   , 0, comm);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_SIZE_T, &(indexStructure.front())      , &(currentUpdateCandidatesPerTask.front()), &(currentUpdateCandidatesOffset.front()), MPI_SIZE_T, 0, comm);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_SIZE_T, &(indexStructureGlobal.front()), &(currentUpdateCandidatesPerTask.front()), &(currentUpdateCandidatesOffset.front()), MPI_SIZE_T, 0, comm);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_SIZE_T, &(indexAtom.front())           , &(currentUpdateCandidatesPerTask.front()), &(currentUpdateCandidatesOffset.front()), MPI_SIZE_T, 0, comm);
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_SIZE_T, &(indexCoordinate.front())     , &(currentUpdateCandidatesPerTask.front()), &(currentUpdateCandidatesOffset.front()), MPI_SIZE_T, 0, comm);
        }
        else
        {
            MPI_Gatherv(&(currentRmseFraction.front()) , myCurrentUpdateCandidates, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, comm);
            MPI_Gatherv(&(thresholdLoopCount.front())  , myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
            MPI_Gatherv(&(procUpdateCandidate.front()) , myCurrentUpdateCandidates, MPI_INT   , NULL, NULL, NULL, MPI_INT   , 0, comm);
            MPI_Gatherv(&(indexStructure.front())      , myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
            MPI_Gatherv(&(indexStructureGlobal.front()), myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
            MPI_Gatherv(&(indexAtom.front())           , myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
            MPI_Gatherv(&(indexCoordinate.front())     , myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
        }

        if (myRank == 0)
        {
            for (size_t i = 0; i < procUpdateCandidate.size(); ++i)
            {
                if (k == "energy")
                {
                    addTrainingLogEntry(procUpdateCandidate.at(i),
                                        thresholdLoopCount.at(i),
                                        currentRmseFraction.at(i),
                                        indexStructureGlobal.at(i),
                                        indexStructure.at(i));
                }
                else if (k == "force")
                {
                    addTrainingLogEntry(procUpdateCandidate.at(i),
                                        thresholdLoopCount.at(i),
                                        currentRmseFraction.at(i),
                                        indexStructureGlobal.at(i),
                                        indexStructure.at(i),
                                        indexAtom.at(i),
                                        indexCoordinate.at(i));
                }
                else if (k == "charge")
                {
                    addTrainingLogEntry(procUpdateCandidate.at(i),
                                        thresholdLoopCount.at(i),
                                        currentRmseFraction.at(i),
                                        indexStructureGlobal.at(i),
                                        indexStructure.at(i),
                                        indexAtom.at(i));
                }
            }
        }
    }
    sw[k + "_log"].stop();
    sw[k].stop();

    return;
}

double Training::getSingleWeight(size_t element, size_t index)
{
    getWeights();

    return weights.at(element).at(index);
}

void Training::setSingleWeight(size_t element, size_t index, double value)
{
    weights.at(element).at(index) = value;
    setWeights();

    return;
}

vector<
vector<double>> Training::calculateWeightDerivatives(Structure* structure)
{
    Structure& s = *structure;
#ifdef NNP_NO_SF_GROUPS
    calculateSymmetryFunctions(s, false);
#else
    calculateSymmetryFunctionGroups(s, false);
#endif

    vector<vector<double> > dEdc;
    vector<vector<double> > dedc;
    dEdc.resize(numElements);
    dedc.resize(numElements);
    for (size_t i = 0; i < numElements; ++i)
    {
        size_t n = elements.at(i).neuralNetworks.at("short")
                   .getNumConnections();
        dEdc.at(i).resize(n, 0.0);
        dedc.at(i).resize(n, 0.0);
    }
    for (vector<Atom>::iterator it = s.atoms.begin();
         it != s.atoms.end(); ++it)
    {
        size_t i = it->element;
        NeuralNetwork& nn = elements.at(i).neuralNetworks.at("short");
        nn.setInput(&((it->G).front()));
        nn.propagate();
        nn.getOutput(&(it->energy));
        nn.calculateDEdc(&(dedc.at(i).front()));
        for (size_t j = 0; j < dedc.at(i).size(); ++j)
        {
            dEdc.at(i).at(j) += dedc.at(i).at(j);
        }
    }

    return dEdc;
}

// Doxygen requires namespace prefix for arguments...
vector<
vector<double>> Training::calculateWeightDerivatives(Structure*  structure,
                                                      std::size_t atom,
                                                      std::size_t component)
{
    Structure& s = *structure;
#ifdef NNP_NO_SF_GROUPS
    calculateSymmetryFunctions(s, true);
#else
    calculateSymmetryFunctionGroups(s, true);
#endif

    vector<vector<double> > dFdc;
    vector<vector<double> > dfdc;
    dFdc.resize(numElements);
    dfdc.resize(numElements);
    for (size_t i = 0; i < numElements; ++i)
    {
        size_t n = elements.at(i).neuralNetworks.at("short")
                   .getNumConnections();
        dFdc.at(i).resize(n, 0.0);
        dfdc.at(i).resize(n, 0.0);
    }
    for (vector<Atom>::iterator it = s.atoms.begin();
         it != s.atoms.end(); ++it)
    {
#ifndef NNP_FULL_SFD_MEMORY
        collectDGdxia((*it), atom, component);
#else
        it->collectDGdxia(atom, component);
#endif
        size_t i = it->element;
        NeuralNetwork& nn = elements.at(i).neuralNetworks.at("short");
        nn.setInput(&((it->G).front()));
        nn.propagate();
        nn.getOutput(&(it->energy));
#ifndef NNP_FULL_SFD_MEMORY
        nn.calculateDFdc(&(dfdc.at(i).front()), &(dGdxia.front()));
#else
        nn.calculateDFdc(&(dfdc.at(i).front()), &(it->dGdxia.front()));
#endif
        for (size_t j = 0; j < dfdc.at(i).size(); ++j)
        {
            dFdc.at(i).at(j) += dfdc.at(i).at(j);
        }
    }

    return dFdc;
}

void Training::setTrainingLogFileName(string fileName)
{
    trainingLogFileName = fileName;

    return;
}

bool Training::advance() const
{
    if (epoch < numEpochs) return true;
    else return false;
}

void Training::getWeights()
{
    if (updateStrategy == US_COMBINED)
    {
        size_t pos = 0;
        for (size_t i = 0; i < numElements; ++i)
        {
            NeuralNetwork const& nn = elements.at(i).neuralNetworks.at(nnId);
            nn.getConnections(&(weights.at(0).at(pos)));
            pos += nn.getNumConnections();
        }
    }
    else if (updateStrategy == US_ELEMENT)
    {
        for (size_t i = 0; i < numElements; ++i)
        {
            NeuralNetwork const& nn = elements.at(i).neuralNetworks.at(nnId);
            nn.getConnections(&(weights.at(i).front()));
        }
    }

    return;
}

void Training::setWeights()
{
    if (updateStrategy == US_COMBINED)
    {
        size_t pos = 0;
        for (size_t i = 0; i < numElements; ++i)
        {
            NeuralNetwork& nn = elements.at(i).neuralNetworks.at(nnId);
            nn.setConnections(&(weights.at(0).at(pos)));
            pos += nn.getNumConnections();
        }
    }
    else if (updateStrategy == US_ELEMENT)
    {
        for (size_t i = 0; i < numElements; ++i)
        {
            NeuralNetwork& nn = elements.at(i).neuralNetworks.at(nnId);
            nn.setConnections(&(weights.at(i).front()));
        }
    }

    return;
}

// Doxygen requires namespace prefix for arguments...
void Training::addTrainingLogEntry(int                 proc,
                                   std::size_t         il,
                                   double              f,
                                   std::size_t         isg,
                                   std::size_t         is)
{
    string s = strpr("  E %5zu %10zu %5d %3zu %10.2E %10zu %5zu\n",
                     epoch, countUpdates, proc, il + 1, f, isg, is);
    trainingLog << s;

    return;
}

// Doxygen requires namespace prefix for arguments...
void Training::addTrainingLogEntry(int                 proc,
                                   std::size_t         il,
                                   double              f,
                                   std::size_t         isg,
                                   std::size_t         is,
                                   std::size_t         ia,
                                   std::size_t         ic)
{
    string s = strpr("  F %5zu %10zu %5d %3zu %10.2E %10zu %5zu %5zu %2zu\n",
                     epoch, countUpdates, proc, il + 1, f, isg, is, ia, ic);
    trainingLog << s;

    return;
}

// Doxygen requires namespace prefix for arguments...
void Training::addTrainingLogEntry(int                 proc,
                                   std::size_t         il,
                                   double              f,
                                   std::size_t         isg,
                                   std::size_t         is,
                                   std::size_t         ia)
{
    string s = strpr("  Q %5zu %10zu %5d %3zu %10.2E %10zu %5zu %5zu\n",
                     epoch, countUpdates, proc, il + 1, f, isg, is, ia);
    trainingLog << s;

    return;
}

#ifndef NNP_FULL_SFD_MEMORY
void Training::collectDGdxia(Atom const& atom,
                             size_t      indexAtom,
                             size_t      indexComponent)
{
    size_t const nsf = atom.numSymmetryFunctions;

    // Reset dGdxia array.
    dGdxia.clear();
    vector<double>(dGdxia).swap(dGdxia);
    dGdxia.resize(nsf, 0.0);

    vector<vector<size_t> > const& tableFull
        = elements.at(atom.element).getSymmetryFunctionTable();

    for (size_t i = 0; i < atom.numNeighbors; i++)
    {
        if (atom.neighbors[i].index == indexAtom)
        {
            Atom::Neighbor const& n = atom.neighbors[i];
            vector<size_t> const& table = tableFull.at(n.element);
            for (size_t j = 0; j < n.dGdr.size(); ++j)
            {
                dGdxia[table.at(j)] += n.dGdr[j][indexComponent];
            }
        }
    }
    if (atom.index == indexAtom)
    {
        for (size_t i = 0; i < nsf; ++i)
        {
            dGdxia[i] += atom.dGdr[i][indexComponent];
        }
    }

    return;
}
#endif

void Training::randomizeNeuralNetworkWeights(string const& type)
{
    string keywordNW = "";
    if      (type == "short" ) keywordNW = "nguyen_widrow_weights_short";
    else if (type == "charge") keywordNW = "nguyen_widrow_weights_charge";
    else
    {
        throw runtime_error("ERROR: Unknown neural network type.\n");
    }

    double minWeights = atof(settings["weights_min"].c_str());
    double maxWeights = atof(settings["weights_max"].c_str());
    log << strpr("Initial weights selected randomly in interval "
                 "[%f, %f).\n", minWeights, maxWeights);
    vector<double> w;
    for (size_t i = 0; i < numElements; ++i)
    {
        NeuralNetwork& nn = elements.at(i).neuralNetworks.at(type);
        w.resize(nn.getNumConnections(), 0);
        for (size_t j = 0; j < w.size(); ++j)
        {
            w.at(j) = minWeights + gsl_rng_uniform(rngGlobal)
                    * (maxWeights - minWeights);
        }
        nn.setConnections(&(w.front()));
    }
    if (settings.keywordExists(keywordNW))
    {
        log << "Weights modified according to Nguyen Widrow scheme.\n";
        for (vector<Element>::iterator it = elements.begin();
             it != elements.end(); ++it)
        {
            NeuralNetwork& nn = it->neuralNetworks.at(type);
            nn.modifyConnections(NeuralNetwork::MS_NGUYENWIDROW);
        }
    }
    else if (settings.keywordExists("precondition_weights"))
    {
        throw runtime_error("ERROR: Preconditioning of weights not yet"
                            " implemented.\n");
    }
    else
    {
        log << "Weights modified accoring to Glorot Bengio scheme.\n";
        //log << "Weights connected to output layer node set to zero.\n";
        log << "Biases set to zero.\n";
        for (vector<Element>::iterator it = elements.begin();
             it != elements.end(); ++it)
        {
            NeuralNetwork& nn = it->neuralNetworks.at(type);
            nn.modifyConnections(NeuralNetwork::MS_GLOROTBENGIO);
            //nn->modifyConnections(NeuralNetwork::MS_ZEROOUTPUTWEIGHTS);
            nn.modifyConnections(NeuralNetwork::MS_ZEROBIAS);
        }
    }

    return;
}

void Training::setupSelectionMode(string const& property)
{
    bool all = (property == "all");
    bool isProperty = (find(pk.begin(), pk.end(), property) != pk.end());
    if (!(all || isProperty))
    {
        throw runtime_error("ERROR: Unknown property for selection mode"
                            " setup.\n");
    }

    if (all)
    {
        if (!(settings.keywordExists("selection_mode") ||
              settings.keywordExists("rmse_threshold") ||
              settings.keywordExists("rmse_threshold_trials"))) return;
        log << "Global selection mode settings:\n";
    }
    else
    {
        if (!(settings.keywordExists("selection_mode_" + property) ||
              settings.keywordExists("rmse_threshold_" + property) ||
              settings.keywordExists("rmse_threshold_trials_"
                                     + property))) return;
        log << "Selection mode settings specific to property \""
            << property << "\":\n";
    }
    string keyword;
    if (all) keyword = "selection_mode";
    else keyword = "selection_mode_" + property;

    if (settings.keywordExists(keyword))
    {
        map<size_t, SelectionMode> schedule;
        vector<string> args = split(settings[keyword]);
        if (args.size() % 2 != 1)
        {
            throw runtime_error("ERROR: Incorrect selection mode format.\n");
        }
        schedule[0] = (SelectionMode)atoi(args.at(0).c_str());
        for (size_t i = 1; i < args.size(); i = i + 2)
        {
            schedule[(size_t)atoi(args.at(i).c_str())] =
                (SelectionMode)atoi(args.at(i + 1).c_str());
        }
        for (map<size_t, SelectionMode>::const_iterator it = schedule.begin();
             it != schedule.end(); ++it)
        {
            log << strpr("- Selection mode starting with epoch %zu:\n",
                         it->first);
            if (it->second == SM_RANDOM)
            {
                log << strpr("  Random selection of update candidates: "
                             "SelectionMode::SM_RANDOM (%d)\n", it->second);
            }
            else if (it->second == SM_SORT)
            {
                log << strpr("  Update candidates selected according to error: "
                             "SelectionMode::SM_SORT (%d)\n", it->second);
            }
            else if (it->second == SM_THRESHOLD)
            {
                log << strpr("  Update candidates chosen randomly above RMSE "
                             "threshold: SelectionMode::SM_THRESHOLD (%d)\n",
                             it->second);
            }
            else
            {
                throw runtime_error("ERROR: Unknown selection mode.\n");
            }
        }
        if (all)
        {
            for (auto& i : p)
            {
                i.second.selectionModeSchedule = schedule;
                i.second.selectionMode = schedule[0];
            }
        }
        else
        {
            p[property].selectionModeSchedule = schedule;
            p[property].selectionMode = schedule[0];
        }
    }

    if (all) keyword = "rmse_threshold";
    else keyword = "rmse_threshold_" + property;
    if (settings.keywordExists(keyword))
    {
        double t = atof(settings[keyword].c_str());
        log << strpr("- RMSE selection threshold: %.2f * RMSE\n", t);
        if (all) for (auto& i : p) i.second.rmseThreshold = t;
        else p[property].rmseThreshold = t;
    }

    if (all) keyword = "rmse_threshold_trials";
    else keyword = "rmse_threshold_trials_" + property;
    if (settings.keywordExists(keyword))
    {
        size_t t = atoi(settings[keyword].c_str());
        log << strpr("- RMSE selection trials   : %zu\n", t);
        if (all) for (auto& i : p) i.second.rmseThresholdTrials = t;
        else p[property].rmseThresholdTrials = t;
    }

    return;
}

void Training::setupFileOutput(string const& type)
{
    string keyword = "write_";
    bool isProperty = (find(pk.begin(), pk.end(), type) != pk.end());
    if      (type == "energy"       ) keyword += "trainpoints";
    else if (type == "force"        ) keyword += "trainforces";
    else if (type == "charge"       ) keyword += "traincharges";
    else if (type == "weights_epoch") keyword += type;
    else if (type == "neuronstats"  ) keyword += type;
    else
    {
        throw runtime_error("ERROR: Invalid type for file output setup.\n");
    }

    // Check how often energy comparison files should be written.
    if (settings.keywordExists(keyword))
    {
        size_t* writeEvery = nullptr;
        size_t* writeAlways = nullptr;
        string message;
        if (isProperty)
        {
            writeEvery = &(p[type].writeCompEvery);
            writeAlways = &(p[type].writeCompAlways);
            message = "Property \"" + type + "\" comparison";
            message.at(0) = toupper(message.at(0));
        }
        else if (type == "weights_epoch")
        {
            writeEvery = &writeWeightsEvery;
            writeAlways = &writeWeightsAlways;
            message = "Weight";
        }
        else if (type == "neuronstats")
        {
            writeEvery = &writeNeuronStatisticsEvery;
            writeAlways = &writeNeuronStatisticsAlways;
            message = "Neuron statistics";
        }

        *writeEvery = 1;
        vector<string> v = split(reduce(settings[keyword]));
        if (v.size() == 1) *writeEvery = (size_t)atoi(v.at(0).c_str());
        else if (v.size() == 2)
        {
            *writeEvery = (size_t)atoi(v.at(0).c_str());
            *writeAlways = (size_t)atoi(v.at(1).c_str());
        }
        log << strpr((message
                      + " files will be written every %zu epochs.\n").c_str(),
                     *writeEvery);
        if (*writeAlways > 0)
        {
            log << strpr((message
                          + " files will always be written up to epoch "
                            "%zu.\n").c_str(), *writeAlways);
        }
    }

    return;
}

void Training::setupUpdatePlan(string const& property)
{
    bool isProperty = (find(pk.begin(), pk.end(), property) != pk.end());
    if (!isProperty)
    {
        throw runtime_error("ERROR: Unknown property for update plan"
                            " setup.\n");
    }

    // Actual property modified here.
    Property& pa = p[property];
    string keyword = property + "_fraction";
    pa.epochFraction = atof(settings[keyword].c_str());
    keyword = "task_batch_size_" + property;
    pa.taskBatchSize = (size_t)atoi(settings[keyword].c_str());
    if (pa.taskBatchSize == 0)
    {
        pa.patternsPerUpdate =
            static_cast<size_t>(pa.updateCandidates.size() * pa.epochFraction);
        pa.numUpdates = 1;
    }
    else
    {
        pa.patternsPerUpdate = pa.taskBatchSize;
        pa.numUpdates =
            static_cast<size_t>((pa.numTrainPatterns * pa.epochFraction)
                                / pa.taskBatchSize / numProcs);
    }
    pa.patternsPerUpdateGlobal = pa.patternsPerUpdate;
    MPI_Allreduce(MPI_IN_PLACE, &(pa.patternsPerUpdateGlobal), 1, MPI_SIZE_T, MPI_SUM, comm);
    pa.errorsPerTask.resize(numProcs, 0);
    if (jacobianMode == JM_FULL)
    {
        pa.errorsPerTask.at(myRank) = static_cast<int>(pa.patternsPerUpdate);
    }
    else
    {
        pa.errorsPerTask.at(myRank) = 1;
    }
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, &(pa.errorsPerTask.front()), 1, MPI_INT, comm);
    if (jacobianMode == JM_FULL)
    {
        pa.weightsPerTask.resize(numUpdaters);
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            pa.weightsPerTask.at(i).resize(numProcs, 0);
            for (int j = 0; j < numProcs; ++j)
            {
                pa.weightsPerTask.at(i).at(j) = pa.errorsPerTask.at(j)
                                              * numWeightsPerUpdater.at(i);
            }
        }
    }
    pa.numErrorsGlobal = 0;
    for (size_t i = 0; i < pa.errorsPerTask.size(); ++i)
    {
        pa.offsetPerTask.push_back(pa.numErrorsGlobal);
        pa.numErrorsGlobal += pa.errorsPerTask.at(i);
    }
    pa.offsetJacobian.resize(numUpdaters);
    for (size_t i = 0; i < numUpdaters; ++i)
    {
        for (size_t j = 0; j < pa.offsetPerTask.size(); ++j)
        {
            pa.offsetJacobian.at(i).push_back(pa.offsetPerTask.at(j) *
                                              numWeightsPerUpdater.at(i));
        }
    }
    log << "Update plan for property \"" + property + "\":\n";
    log << strpr("- Per-task batch size                          : %zu\n",
                 pa.taskBatchSize);
    log << strpr("- Fraction of patterns used per epoch          : %.4f\n",
                 pa.epochFraction);
    log << strpr("- Updates per epoch                            : %zu\n",
                 pa.numUpdates);
    log << strpr("- Patterns used per update (rank %3d / global) : "
                 "%10zu / %zu\n",
                 myRank, pa.patternsPerUpdate, pa.patternsPerUpdateGlobal);
    log << "----------------------------------------------\n";

    return;
}

void Training::allocateArrays(string const& property)
{
    bool isProperty = (find(pk.begin(), pk.end(), property) != pk.end());
    if (!isProperty)
    {
        throw runtime_error("ERROR: Unknown property for array allocation.\n");
    }

    log << "Allocating memory for " + property +
           " error vector and Jacobian.\n";
    Property& pa = p[property];
    pa.error.resize(numUpdaters);
    pa.jacobian.resize(numUpdaters);
    for (size_t i = 0; i < numUpdaters; ++i)
    {
        size_t size = 1;
        if (( parallelMode == PM_TRAIN_ALL ||
              (parallelMode == PM_TRAIN_RK0 && myRank == 0)) &&
            jacobianMode != JM_SUM)
        {
            size *= pa.numErrorsGlobal;
        }
        else if ((parallelMode == PM_TRAIN_RK0 && myRank != 0) &&
                 jacobianMode != JM_SUM)
        {
            size *= pa.errorsPerTask.at(myRank);
        }
        pa.error.at(i).resize(size, 0.0);
        pa.jacobian.at(i).resize(size * numWeightsPerUpdater.at(i), 0.0);
        log << strpr("Updater %3zu:\n", i);
        log << strpr(" - Error    size: %zu\n", pa.error.at(i).size());
        log << strpr(" - Jacobian size: %zu\n", pa.jacobian.at(i).size());
    }
    log << "----------------------------------------------\n";

    return;
}

void Training::writeTimingData(bool append, string const fileName)
{
    ofstream file;
    string fileNameActual = fileName;
    if (nnpType == NNPType::SHORT_CHARGE_NN)
    {
        fileNameActual += strpr(".stage-%zu", stage);
    }

    vector<string> sub = {"_err", "_com", "_upd", "_log"};
    if (append) file.open(fileNameActual.c_str(), ofstream::app);
    else
    {
        file.open(fileNameActual.c_str());

        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back("Timing data for training loop.");
        colSize.push_back(10);
        colName.push_back("epoch");
        colInfo.push_back("Current epoch.");
        colSize.push_back(11);
        colName.push_back("train");
        colInfo.push_back("Time for training.");
        colSize.push_back(7);
        colName.push_back("ptrain");
        colInfo.push_back("Time for training (percentage of loop).");
        colSize.push_back(11);
        colName.push_back("error");
        colInfo.push_back("Time for error calculation.");
        colSize.push_back(7);
        colName.push_back("perror");
        colInfo.push_back("Time for error calculation (percentage of loop).");
        colSize.push_back(11);
        colName.push_back("epoch");
        colInfo.push_back("Time for this epoch.");
        colSize.push_back(11);
        colName.push_back("total");
        colInfo.push_back("Total time for all epochs.");
        for (auto k : pk)
        {
            colSize.push_back(11);
            colName.push_back(p[k].tiny + "train");
            colInfo.push_back("");
            colSize.push_back(7);
            colName.push_back(p[k].tiny + "ptrain");
            colInfo.push_back("");
        }
        for (auto s : sub)
        {
            for (auto k : pk)
            {
                colSize.push_back(11);
                colName.push_back(p[k].tiny + s);
                colInfo.push_back("");
                colSize.push_back(7);
                colName.push_back(p[k].tiny + "p" + s);
                colInfo.push_back("");
            }
        }
        appendLinesToFile(file,
                          createFileHeader(title, colSize, colName, colInfo));
    }

    double timeLoop = sw["loop"].getLoop();
    file << strpr("%10zu", epoch);
    file << strpr(" %11.3E", sw["train"].getLoop());
    file << strpr(" %7.3f", sw["train"].getLoop() / timeLoop);
    file << strpr(" %11.3E", sw["error"].getLoop());
    file << strpr(" %7.3f", sw["error"].getLoop() / timeLoop);
    file << strpr(" %11.3E", timeLoop);
    file << strpr(" %11.3E", sw["loop"].getTotal());

    for (auto k : pk)
    {
        file << strpr(" %11.3E", sw[k].getLoop());
        file << strpr(" %7.3f", sw[k].getLoop() / sw["train"].getLoop());
    }
    for (auto s : sub)
    {
        for (auto k : pk)
        {
            file << strpr(" %11.3E", sw[k + s].getLoop());
            file << strpr(" %7.3f", sw[k + s].getLoop() / sw[k].getLoop());
        }
    }
    file << "\n";

    file.flush();
    file.close();

    return;
}

Training::Property::Property(string const& property) :
    property               (property ),
    displayMetric          (""       ),
    tiny                   (""       ),
    plural                 (""       ),
    selectionMode          (SM_RANDOM),
    numTrainPatterns       (0        ),
    numTestPatterns        (0        ),
    taskBatchSize          (0        ),
    writeCompEvery         (0        ),
    writeCompAlways        (0        ),
    posUpdateCandidates    (0        ),
    rmseThresholdTrials    (0        ),
    countUpdates           (0        ),
    numUpdates             (0        ),
    patternsPerUpdate      (0        ),
    patternsPerUpdateGlobal(0        ),
    numErrorsGlobal        (0        ),
    epochFraction          (0.0      ),
    rmseThreshold          (0.0      )
{
    if (property == "energy")
    {
        tiny = "E";
        plural = "energies";
        errorMetrics = {"RMSEpa", "RMSE", "MAEpa", "MAE"};
    }
    else if (property == "force")
    {
        tiny = "F";
        plural = "forces";
        errorMetrics = {"RMSE", "MAE"};
    }
    else if (property == "charge")
    {
        tiny = "Q";
        plural = "charges";
        errorMetrics = {"RMSE", "MAE"};
    }
    else
    {
        throw runtime_error("ERROR: Unknown training property.\n");
    }

    // Set up error metrics
    for (auto m : errorMetrics)
    {
        errorTrain[m] = 0.0;
        errorTest[m] = 0.0;
    }
    displayMetric = errorMetrics.at(0);
}
