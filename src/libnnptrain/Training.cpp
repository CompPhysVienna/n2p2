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
#include "Stopwatch.h"
#include "utility.h"
#include "mpi-extra.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm> // std::sort, std::fill
#include <cmath>     // fabs
#include <cstdlib>   // atoi
#include <gsl/gsl_rng.h>
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error, std::range_error

using namespace std;
using namespace nnp;

Training::Training() : Dataset(),
                       updaterType                (UT_GD          ),
                       parallelMode               (PM_TRAIN_RK0   ),
                       jacobianMode               (JM_SUM         ),
                       updateStrategy             (US_COMBINED    ),
                       selectionMode              (SM_RANDOM      ),
                       hasUpdaters                (false          ),
                       hasStructures              (false          ),
                       useForces                  (false          ),
                       reapeatedEnergyUpdates     (false          ),
                       freeMemory                 (false          ),
                       writeTrainingLog           (false          ),
                       numUpdaters                (0              ),
                       numEnergiesTrain           (0              ),
                       numForcesTrain             (0              ),
                       numEpochs                  (0              ),
                       taskBatchSizeEnergy        (0              ),
                       taskBatchSizeForce         (0              ),
                       epoch                      (0              ),
                       writeEnergiesEvery         (0              ),
                       writeForcesEvery           (0              ),
                       writeWeightsEvery          (0              ),
                       writeNeuronStatisticsEvery (0              ),
                       writeEnergiesAlways        (0              ),
                       writeForcesAlways          (0              ),
                       writeWeightsAlways         (0              ),
                       writeNeuronStatisticsAlways(0              ),
                       posUpdateCandidatesEnergy  (0              ),
                       posUpdateCandidatesForce   (0              ),
                       rmseThresholdTrials        (0              ),
                       countUpdates               (0              ),
                       energyUpdates              (0              ),
                       forceUpdates               (0              ),
                       energiesPerUpdate          (0              ),
                       energiesPerUpdateGlobal    (0              ),
                       errorsGlobalEnergy         (0              ),
                       forcesPerUpdate            (0              ),
                       forcesPerUpdateGlobal      (0              ),
                       errorsGlobalForce          (0              ),
                       numWeights                 (0              ),
                       epochFractionEnergies      (0.0            ),
                       epochFractionForces        (0.0            ),
                       rmseEnergiesTrain          (0.0            ),
                       rmseEnergiesTest           (0.0            ),
                       rmseForcesTrain            (0.0            ),
                       rmseForcesTest             (0.0            ),
                       rmseThresholdEnergy        (0.0            ),
                       rmseThresholdForce         (0.0            ),
                       forceWeight                (0.0            ),
                       trainingLogFileName        ("train-log.out")
{
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

    size_t numMyEnergiesTrain = 0;
    size_t numMyEnergiesTest  = 0;
    size_t numMyForcesTrain   = 0;
    size_t numMyForcesTest    = 0;
    vector<size_t> numMyForcesTrainPerElement;
    numMyForcesTrainPerElement.resize(numElements, 0);
    double testSetFraction = atof(settings["test_fraction"].c_str());
    log << strpr("Desired test set ratio      : %f\n", testSetFraction);
    //for (vector<Structure>::iterator it = structures.begin();
    //     it != structures.end(); ++it)
    if (structures.size() > 0) hasStructures = true;
    else hasStructures = false;
    for (size_t i = 0; i < structures.size(); ++i)
    {
        Structure& s = structures.at(i);
        if (gsl_rng_uniform(rng) < testSetFraction)
        {
            s.sampleType = Structure::ST_TEST;
            numMyEnergiesTest++;
            numMyForcesTest += 3 * s.numAtoms;
        }
        else
        {
            s.sampleType = Structure::ST_TRAINING;
            numMyEnergiesTrain++;
            numMyForcesTrain += 3 * s.numAtoms;
            for (size_t j = 0; j < numElements; ++j)
            {
                numMyForcesTrainPerElement.at(j) +=
                                                3 * s.numAtomsPerElement.at(j);
            }
            updateCandidatesEnergy.push_back(UpdateCandidate());
            updateCandidatesEnergy.back().s = i;
            for (vector<Atom>::const_iterator it = s.atoms.begin();
                 it != s.atoms.end(); ++it)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    updateCandidatesForce.push_back(UpdateCandidate());
                    updateCandidatesForce.back().s = i;
                    updateCandidatesForce.back().a = it->index;
                    updateCandidatesForce.back().c = j;
                }
            }
        }
    }
    for (size_t i = 0; i < numElements; ++i)
    {
        if (hasStructures && (numMyForcesTrainPerElement.at(i) == 0))
        {
            throw runtime_error(strpr("ERROR: Process %d has no atoms of "
                                      "element %d (%2s).\n",
                                      myRank,
                                      i,
                                      elementMap[i].c_str()));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &numMyEnergiesTrain, 1, MPI_SIZE_T, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &numMyEnergiesTest , 1, MPI_SIZE_T, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &numMyForcesTrain  , 1, MPI_SIZE_T, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &numMyForcesTest   , 1, MPI_SIZE_T, MPI_SUM, comm);
    numEnergiesTrain = numMyEnergiesTrain;
    numForcesTrain = numMyForcesTrain;
    log << strpr("Total number of energies    : %d\n", numStructures);
    log << strpr("Number of training energies : %d\n", numMyEnergiesTrain);
    log << strpr("Number of test     energies : %d\n", numMyEnergiesTest);
    log << strpr("Number of training forces   : %d\n", numMyForcesTrain);
    log << strpr("Number of test     forces   : %d\n", numMyForcesTest);
    log << strpr("Actual test set fraction    : %f\n",
                 numMyEnergiesTest / double(numStructures));

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
    // Local weights array (actual weights array depends on update strategy).
    vector<vector<double> > w;

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

    // Create per-element connections vectors.
    for (size_t i = 0; i < numElements; ++i)
    {
        w.push_back(vector<double>());
        w.at(i).resize(elements.at(i).neuralNetwork->getNumConnections(), 0.0);
    }

    if (settings.keywordExists("use_old_weights_short"))
    {
        log << "Reading old weights from files.\n";
        log << "Calling standard weight initialization routine...\n";
        log << "*****************************************"
               "**************************************\n";
        setupNeuralNetworkWeights("weights.%03zu.data");
        return;
    }
    else
    {
        double minWeights = atof(settings["weights_min"].c_str());
        double maxWeights = atof(settings["weights_max"].c_str());
        log << strpr("Initial weights selected randomly in interval "
                     "[%f, %f).\n", minWeights, maxWeights);
        for (size_t i = 0; i < numElements; ++i)
        {
            for (size_t j = 0; j < w.at(i).size(); ++j)
            {
                w.at(i).at(j) = minWeights + gsl_rng_uniform(rngGlobal)
                              * (maxWeights - minWeights);
            }
            elements.at(i).neuralNetwork->setConnections(&(w.at(i).front()));
        }
        if (settings.keywordExists("nguyen_widrow_weights_short"))
        {
            log << "Weights modified according to Nguyen Widrow scheme.\n";
            for (vector<Element>::iterator it = elements.begin();
                 it != elements.end(); ++it)
            {
                it->neuralNetwork->
                    modifyConnections(NeuralNetwork::MS_NGUYENWIDROW);
            }
        }
        else if (settings.keywordExists("precondition_weights"))
        {
            throw runtime_error("ERROR: Preconditioning of weights not yet"
                                " implemented.\n");
            //it->neuralNetwork->
            //    modifyConnections(NeuralNetwork::MS_PRECONDITIONOUTPUT,
            //                      mean,
            //                      sigma);
        }
        else
        {
            log << "Weights modified accoring to Glorot Bengio scheme.\n";
            //log << "Weights connected to output layer node set to zero.\n";
            log << "Biases set to zero.\n";
            for (vector<Element>::iterator it = elements.begin();
                 it != elements.end(); ++it)
            {
                it->neuralNetwork->
                    modifyConnections(NeuralNetwork::MS_GLOROTBENGIO);
                //it->neuralNetwork->
                //    modifyConnections(NeuralNetwork::MS_ZEROOUTPUTWEIGHTS);
                it->neuralNetwork->
                    modifyConnections(NeuralNetwork::MS_ZEROBIAS);
            }
        }
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
            numWeights += elements.at(i).neuralNetwork->getNumConnections();
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
            size_t n = elements.at(i).neuralNetwork->getNumConnections();
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

void Training::setupTraining()
{
    log << "\n";
    log << "*** SETUP: TRAINING *********************"
           "**************************************\n";
    log << "\n";

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

    vector<string> selectionModeArgs = split(settings["selection_mode"]);
    if (selectionModeArgs.size() % 2 != 1)
    {
        throw runtime_error("ERROR: Incorrect selection mode format.\n");
    }
    selectionModeSchedule[0] =
                          (SelectionMode)atoi(selectionModeArgs.at(0).c_str());
    for (size_t i = 1; i < selectionModeArgs.size(); i = i + 2)
    {
        selectionModeSchedule[(size_t)atoi(selectionModeArgs.at(i).c_str())] =
                      (SelectionMode)atoi(selectionModeArgs.at(i + 1).c_str());
    }
    for (map<size_t, SelectionMode>::const_iterator it =
         selectionModeSchedule.begin();
         it != selectionModeSchedule.end(); ++it)
    {
        log << strpr("Selection mode starting with epoch %zu:\n", it->first);
        if (it->second == SM_RANDOM)
        {
            log << strpr("Random selection of update candidates: "
                         "SelectionMode::SM_RANDOM (%d)\n", it->second);
        }
        else if (it->second == SM_SORT)
        {
            log << strpr("Update candidates selected according to error: "
                         "SelectionMode::SM_SORT (%d)\n", it->second);
        }
        else if (it->second == SM_THRESHOLD)
        {
            log << strpr("Update candidates chosen randomly above RMSE "
                         "threshold: SelectionMode::SM_THRESHOLD (%d)\n",
                         it->second);
            rmseThresholdEnergy
                      = atof(settings["short_energy_error_threshold"].c_str());
            rmseThresholdForce
                       = atof(settings["short_force_error_threshold"].c_str());
            rmseThresholdTrials
                             = atof(settings["rmse_threshold_trials"].c_str());
            log << strpr("Energy threshold: %.2f * RMSE(Energy)\n",
                         rmseThresholdEnergy);
            if (useForces)
            {
                log << strpr("Force  threshold: %.2f * RMSE(Force)\n",
                             rmseThresholdForce);
            }
            log << strpr("Maximum number of update candidate trials: %zu\n",
                         rmseThresholdTrials);
        }
        else
        {
            throw runtime_error("ERROR: Unknown selection mode.\n");
        }
    }
    selectionMode = selectionModeSchedule[0];

    log << "-----------------------------------------"
           "--------------------------------------\n";
    reapeatedEnergyUpdates = settings.keywordExists("repeated_energy_update");
    if (useForces && reapeatedEnergyUpdates)
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

    // Check how often energy comparison files should be written.
    if (settings.keywordExists("write_trainpoints"))
    {
        writeEnergiesEvery = 1;
        vector<string> v = split(reduce(settings["write_trainpoints"]));
        if (v.size() == 1)
        {
            writeEnergiesEvery = (size_t)atoi(v.at(0).c_str());
        }
        else if (v.size() == 2)
        {
            writeEnergiesEvery = (size_t)atoi(v.at(0).c_str());
            writeEnergiesAlways = (size_t)atoi(v.at(1).c_str());
        }
        log << strpr("Energy comparison will be written every %d epochs.\n",
                     writeEnergiesEvery);
        if (writeEnergiesAlways > 0)
        {
            log << strpr("Up to epoch %d energy comparison will be written"
                         " every epoch.\n", writeEnergiesAlways);
        }
    }

    // Check how often force comparison files should be written.
    if (settings.keywordExists("write_trainforces"))
    {
        writeForcesEvery = 1;
        vector<string> v = split(reduce(settings["write_trainforces"]));
        if (v.size() == 1)
        {
            writeForcesEvery = (size_t)atoi(v.at(0).c_str());
        }
        else if (v.size() == 2)
        {
            writeForcesEvery = (size_t)atoi(v.at(0).c_str());
            writeForcesAlways = (size_t)atoi(v.at(1).c_str());
        }
        if (useForces)
        {
            log << strpr("Force comparison will be written every %d epochs.\n",
                         writeForcesEvery);
        }
        if (writeForcesAlways > 0 && useForces)
        {
            log << strpr("Up to epoch %d force comparison will be written"
                         " every epoch.\n", writeForcesAlways);
        }
    }

    // Check how often energy comparison files should be written.
    if (settings.keywordExists("write_weights_epoch"))
    {
        writeWeightsEvery = 1;
        vector<string> v = split(reduce(settings["write_weights_epoch"]));
        if (v.size() == 1)
        {
            writeWeightsEvery = (size_t)atoi(v.at(0).c_str());
        }
        else if (v.size() == 2)
        {
            writeWeightsEvery = (size_t)atoi(v.at(0).c_str());
            writeWeightsAlways = (size_t)atoi(v.at(1).c_str());
        }
        log << strpr("Weights will be written every %d epochs.\n",
                     writeWeightsEvery);
        if (writeWeightsAlways > 0)
        {
            log << strpr("Up to epoch %d weights will be written"
                         " every epoch.\n", writeWeightsAlways);
        }
    }

    // Check how often neuron statistics should be written.
    if (settings.keywordExists("write_neuronstats"))
    {
        writeNeuronStatisticsEvery = 1;
        vector<string> v = split(reduce(settings["write_neuronstats"]));
        if (v.size() == 1)
        {
            writeNeuronStatisticsEvery = (size_t)atoi(v.at(0).c_str());
        }
        else if (v.size() == 2)
        {
            writeNeuronStatisticsEvery = (size_t)atoi(v.at(0).c_str());
            writeNeuronStatisticsAlways = (size_t)atoi(v.at(1).c_str());
        }
        log << strpr("Neuron statistics will be written every %d epochs.\n",
                     writeNeuronStatisticsEvery);
        if (writeNeuronStatisticsAlways > 0)
        {
            log << strpr("Up to epoch %d neuron statistics will be written"
                         " every epoch.\n", writeNeuronStatisticsAlways);
        }
    }

    // Prepare training log header.
    writeTrainingLog = settings.keywordExists("write_trainlog");
    if (writeTrainingLog && myRank == 0)
    {
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
        colInfo.push_back("Update type (E = energy, F = force).");
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

    // Compute number of updates and energies/forces per update.
    log << "-----------------------------------------"
           "--------------------------------------\n";
    epochFractionEnergies = atof(settings["short_energy_fraction"].c_str());
    taskBatchSizeEnergy
        = (size_t)atoi(settings["task_batch_size_energy"].c_str());
    if (taskBatchSizeEnergy == 0)
    {
        energiesPerUpdate = static_cast<size_t>(updateCandidatesEnergy.size()
                                                * epochFractionEnergies);
        energyUpdates = 1;
    }
    else
    {
        energiesPerUpdate = taskBatchSizeEnergy;
        energyUpdates = static_cast<size_t>((numEnergiesTrain
                                            * epochFractionEnergies)
                                            / taskBatchSizeEnergy / numProcs);
    }
    energiesPerUpdateGlobal = energiesPerUpdate;
    MPI_Allreduce(MPI_IN_PLACE, &energiesPerUpdateGlobal, 1, MPI_SIZE_T, MPI_SUM, comm);
    errorsPerTaskEnergy.resize(numProcs, 0);
    if (jacobianMode == JM_FULL)
    {
        errorsPerTaskEnergy.at(myRank) = static_cast<int>(energiesPerUpdate);
    }
    else
    {
        errorsPerTaskEnergy.at(myRank) = 1;
    }
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, &(errorsPerTaskEnergy.front()), 1, MPI_INT, comm);
    if (jacobianMode == JM_FULL)
    {
        weightsPerTaskEnergy.resize(numUpdaters);
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            weightsPerTaskEnergy.at(i).resize(numProcs, 0);
            for (int j = 0; j < numProcs; ++j)
            {
                weightsPerTaskEnergy.at(i).at(j) = errorsPerTaskEnergy.at(j)
                                                 * numWeightsPerUpdater.at(i);
            }
        }
    }
    errorsGlobalEnergy = 0;
    for (size_t i = 0; i < errorsPerTaskEnergy.size(); ++i)
    {
        offsetPerTaskEnergy.push_back(errorsGlobalEnergy);
        errorsGlobalEnergy += errorsPerTaskEnergy.at(i);
    }
    offsetJacobianEnergy.resize(numUpdaters);
    for (size_t i = 0; i < numUpdaters; ++i)
    {
        for (size_t j = 0; j < offsetPerTaskEnergy.size(); ++j)
        {
            offsetJacobianEnergy.at(i).push_back(offsetPerTaskEnergy.at(j) *
                                                 numWeightsPerUpdater.at(i));
        }
    }
    log << strpr("Per-task batch size for energies             : %zu\n",
                 taskBatchSizeEnergy);
    log << strpr("Fraction of energies used per epoch          : %.4f\n",
                 epochFractionEnergies);
    log << strpr("Energy updates per epoch                     : %zu\n",
                 energyUpdates);
    log << strpr("Energies used per update (rank %3d / global) : %10zu / %zu\n",
                 myRank, energiesPerUpdate, energiesPerUpdateGlobal);

    if (useForces)
    {
        log << "----------------------------------------------\n";
        epochFractionForces = atof(settings["short_force_fraction"].c_str());
        taskBatchSizeForce
            = (size_t)atoi(settings["task_batch_size_force"].c_str());
        if (taskBatchSizeForce == 0)
        {
            forcesPerUpdate = static_cast<size_t>(updateCandidatesForce.size()
                                                  * epochFractionForces);
            forceUpdates = 1;
        }
        else
        {
            forcesPerUpdate = taskBatchSizeForce;
            forceUpdates = static_cast<size_t>((numForcesTrain
                                                * epochFractionForces)
                                                / taskBatchSizeForce
                                                / numProcs);
        }
        forcesPerUpdateGlobal = forcesPerUpdate;
        MPI_Allreduce(MPI_IN_PLACE, &forcesPerUpdateGlobal, 1, MPI_SIZE_T, MPI_SUM, comm);
        errorsPerTaskForce.resize(numProcs, 0);
        if (jacobianMode == JM_FULL)
        {
            errorsPerTaskForce.at(myRank) = static_cast<int>(forcesPerUpdate);
        }
        else
        {
            errorsPerTaskForce.at(myRank) = 1;
        }
        MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, &(errorsPerTaskForce.front()), 1, MPI_INT, comm);
        if (jacobianMode == JM_FULL)
        {
            weightsPerTaskForce.resize(numUpdaters);
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                weightsPerTaskForce.at(i).resize(numProcs, 0);
                for (int j = 0; j < numProcs; ++j)
                {
                    weightsPerTaskForce.at(i).at(j) =
                        errorsPerTaskForce.at(j) * numWeightsPerUpdater.at(i);
                }
            }
        }
        errorsGlobalForce = 0;
        for (size_t i = 0; i < errorsPerTaskForce.size(); ++i)
        {
            offsetPerTaskForce.push_back(errorsGlobalForce);
            errorsGlobalForce += errorsPerTaskForce.at(i);
        }
        offsetJacobianForce.resize(numUpdaters);
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            for (size_t j = 0; j < offsetPerTaskForce.size(); ++j)
            {
                offsetJacobianForce.at(i).push_back(
                                                   offsetPerTaskForce.at(j) *
                                                   numWeightsPerUpdater.at(i));
            }
        }
        log << strpr("Per-task batch size for forces               : %zu\n",
                     taskBatchSizeForce);
        log << strpr("Fraction of forces used per epoch            : %.4f\n",
                     epochFractionForces);
        log << strpr("Force updates per epoch                      : %zu\n",
                     forceUpdates);
        log << strpr("Forces used per update (rank %3d / global)   : %10zu / "
                     "%zu\n", myRank, forcesPerUpdate, forcesPerUpdateGlobal);
        log << "----------------------------------------------\n";
        log << strpr("Energy to force ratio                        : "
                     "     1 : %5.1f\n",
                     static_cast<double>(forceUpdates * forcesPerUpdateGlobal)
                     / (energyUpdates * energiesPerUpdateGlobal));
        log << strpr("Energy to force percentages                  : "
                     "%5.1f%% : %5.1f%%\n",
                     energyUpdates * energiesPerUpdateGlobal * 100.0 /
                     (energyUpdates * energiesPerUpdateGlobal
                     + forceUpdates * forcesPerUpdateGlobal),
                     forceUpdates * forcesPerUpdateGlobal * 100.0 /
                     (energyUpdates * energiesPerUpdateGlobal
                     + forceUpdates * forcesPerUpdateGlobal));
    }
    double totalUpdates = energyUpdates + forceUpdates;
    log << "-----------------------------------------"
           "--------------------------------------\n";

    // Allocate error and Jacobian arrays.
    log << "Allocating memory for energy error vector and Jacobian.\n";
    errorE.resize(numUpdaters);
    jacobianE.resize(numUpdaters);
    for (size_t i = 0; i < numUpdaters; ++i)
    {
        size_t size = 1;
        if (( parallelMode == PM_TRAIN_ALL ||
              (parallelMode == PM_TRAIN_RK0 && myRank == 0)) &&
            jacobianMode != JM_SUM)
        {
            size *= errorsGlobalEnergy;            
        }
        else if ((parallelMode == PM_TRAIN_RK0 && myRank != 0) &&
                 jacobianMode != JM_SUM)
        {
            size *= errorsPerTaskEnergy.at(myRank);            
        }
        errorE.at(i).resize(size, 0.0);
        jacobianE.at(i).resize(size * numWeightsPerUpdater.at(i), 0.0);
        log << strpr("Updater %3zu:\n", i);
        log << strpr(" - Error    size: %zu\n", errorE.at(i).size());
        log << strpr(" - Jacobian size: %zu\n", jacobianE.at(i).size());
    }
    if (useForces)
    {
        log << "------------------------------------------------------\n";
        log << "Allocating memory for force error vector and Jacobian.\n";
        errorF.resize(numUpdaters);
        jacobianF.resize(numUpdaters);
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            size_t size = 1;
            if ((parallelMode == PM_TRAIN_ALL ||
                 (parallelMode == PM_TRAIN_RK0 && myRank == 0)) &&
                jacobianMode != JM_SUM)
            {
                size *= errorsGlobalForce;            
            }
            else if ((parallelMode == PM_TRAIN_RK0 && myRank != 0) &&
                     jacobianMode != JM_SUM)
            {
                size *= errorsPerTaskForce.at(myRank);            
            }
            errorF.at(i).resize(size, 0.0);
            jacobianF.at(i).resize(size * numWeightsPerUpdater.at(i), 0.0);
            log << strpr("Updater %3zu:\n", i);
            log << strpr(" - Error    size: %zu\n", errorF.at(i).size());
            log << strpr(" - Jacobian size: %zu\n", jacobianF.at(i).size());
        }
    }
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
            log << "-----------------------------------------"
                   "--------------------------------------\n";
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Training::calculateNeighborLists()
{
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

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Training::calculateRmse(bool const   writeCompFiles,
                             string const identifier,
                             string const fileNameEnergiesTrain,
                             string const fileNameEnergiesTest,
                             string const fileNameForcesTrain,
                             string const fileNameForcesTest)
{
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    bool     energiesTrain = true;
    if (fileNameEnergiesTrain == "") energiesTrain = false;
    bool     energiesTest = true;
    if (fileNameEnergiesTest == "") energiesTest = false;
    bool     forcesTrain = true;
    if (fileNameForcesTrain == "") forcesTrain = false;
    bool     forcesTest = true;
    if (fileNameForcesTest == "") forcesTest = false;
    size_t   countEnergiesTrain = 0;
    size_t   countEnergiesTest  = 0;
    size_t   countForcesTrain   = 0;
    size_t   countForcesTest    = 0;
    ofstream fileEnergiesTrain;
    ofstream fileEnergiesTest;
    ofstream fileForcesTrain;
    ofstream fileForcesTest;

    // Reset current RMSEs.
    rmseEnergiesTrain = 0.0;
    rmseEnergiesTest  = 0.0;
    rmseForcesTrain   = 0.0;
    rmseForcesTest    = 0.0;

    if (writeCompFiles)
    {
        // File header.
        vector<string> header;
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        if (myRank == 0 && (energiesTrain || energiesTest))
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
            header = createFileHeader(title, colSize, colName, colInfo);
        }

        if (energiesTrain)
        {
            fileEnergiesTrain.open(strpr("%s.%04d",
                                         fileNameEnergiesTrain.c_str(),
                                         myRank).c_str());
            if (myRank == 0) appendLinesToFile(fileEnergiesTrain, header);
        }
        if (energiesTest)
        {
            fileEnergiesTest.open(strpr("%s.%04d",
                                        fileNameEnergiesTest.c_str(),
                                        myRank).c_str());
            if (myRank == 0) appendLinesToFile(fileEnergiesTest, header);
        }
        if (useForces)
        {
            header.clear();
            title.clear();
            colName.clear();
            colInfo.clear();
            colSize.clear();
            if (myRank == 0 && (forcesTrain || forcesTest))
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
                header = createFileHeader(title, colSize, colName, colInfo);
            }
            if (forcesTrain)
            {
                fileForcesTrain.open(strpr("%s.%04d",
                                           fileNameForcesTrain.c_str(),
                                           myRank).c_str());
                if (myRank == 0) appendLinesToFile(fileForcesTrain, header);
            }
            if (forcesTest)
            {
                fileForcesTest.open(strpr("%s.%04d",
                                          fileNameForcesTest.c_str(),
                                          myRank).c_str());
                if (myRank == 0) appendLinesToFile(fileForcesTest, header);
            }
        }
    }

    for (vector<Structure>::iterator it = structures.begin();
         it != structures.end(); ++it)
    {
#ifdef NOSFGROUPS
        calculateSymmetryFunctions((*it), useForces);
#else
        calculateSymmetryFunctionGroups((*it), useForces);
#endif
        calculateAtomicNeuralNetworks((*it), useForces);
        calculateEnergy((*it));
        if (it->sampleType == Structure::ST_TRAINING)
        {
            it->updateRmseEnergy(rmseEnergiesTrain, countEnergiesTrain);
            if (writeCompFiles && energiesTrain)
            {
                fileEnergiesTrain << it->getEnergyLine();
            }
            
        }
        else if (it->sampleType == Structure::ST_TEST)
        {
            it->updateRmseEnergy(rmseEnergiesTest, countEnergiesTest);
            if (writeCompFiles && energiesTest)
            {
                fileEnergiesTest << it->getEnergyLine();
            }
        }
        if (useForces)
        {
            calculateForces((*it));
            if (it->sampleType == Structure::ST_TRAINING)
            {
                it->updateRmseForces(rmseForcesTrain, countForcesTrain);
                if (writeCompFiles && forcesTrain)
                {
                    vector<string> v = it->getForcesLines();
                    for (vector<string>::const_iterator it2 = v.begin();
                         it2 != v.end(); ++it2)
                    {
                        fileForcesTrain << (*it2);
                    }
                }
            }
            else if (it->sampleType == Structure::ST_TEST)
            {
                it->updateRmseForces(rmseForcesTest, countForcesTest);
                if (writeCompFiles && forcesTest)
                {
                    vector<string> v = it->getForcesLines();
                    for (vector<string>::const_iterator it2 = v.begin();
                         it2 != v.end(); ++it2)
                    {
                        fileForcesTest << (*it2);
                    }
                }
            }
        }
        if (freeMemory) it->freeAtoms(true);
    }

    averageRmse(rmseEnergiesTrain, countEnergiesTrain);
    averageRmse(rmseEnergiesTest , countEnergiesTest );
    log << strpr("ENERGY %4s", identifier.c_str());
    if (normalize)
    {
        log << strpr(" %13.5E %13.5E",
                     physicalEnergy(rmseEnergiesTrain),
                     physicalEnergy(rmseEnergiesTest));
    }
    log << strpr(" %13.5E %13.5E\n", rmseEnergiesTrain, rmseEnergiesTest);
    if (useForces)
    {
        averageRmse(rmseForcesTrain, countForcesTrain);
        averageRmse(rmseForcesTest , countForcesTest );
        log << strpr("FORCES %4s", identifier.c_str());
        if (normalize)
        {
            log << strpr(" %13.5E %13.5E",
                         physicalForce(rmseForcesTrain),
                         physicalForce(rmseForcesTest));
        }
        log << strpr(" %13.5E %13.5E\n", rmseForcesTrain, rmseForcesTest);
    }

    if (writeCompFiles)
    {
        if (energiesTrain) fileEnergiesTrain.close();
        if (energiesTest) fileEnergiesTest.close();
        if (useForces)
        {
            if (forcesTrain) fileForcesTrain.close();
            if (forcesTest) fileForcesTest.close();
        }
        MPI_Barrier(comm);
        if (myRank == 0)
        {
            if (energiesTrain) combineFiles(fileNameEnergiesTrain);
            if (energiesTest) combineFiles(fileNameEnergiesTest);
            if (useForces)
            {
                if (forcesTrain) combineFiles(fileNameForcesTrain);
                if (forcesTest) combineFiles(fileNameForcesTest);
            }
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

    return;
}

void Training::calculateRmseEpoch()
{
    // Check whether energy/force comparison files should be written for
    // this epoch.
    string identifier = strpr("%d", epoch);
    string fileNameEnergiesTrain = "";
    string fileNameEnergiesTest = "";
    string fileNameForcesTrain = "";
    string fileNameForcesTest = "";
    if (writeEnergiesEvery > 0 &&
        (epoch % writeEnergiesEvery == 0 || epoch <= writeEnergiesAlways))
    {
        fileNameEnergiesTrain = strpr("trainpoints.%06zu.out", epoch);
        fileNameEnergiesTest = strpr("testpoints.%06zu.out", epoch);
    }
    if (useForces &&
        writeForcesEvery > 0 &&
        (epoch % writeForcesEvery == 0 || epoch <= writeForcesAlways))
    {
        fileNameForcesTrain = strpr("trainforces.%06zu.out", epoch);
        fileNameForcesTest = strpr("testforces.%06zu.out", epoch);
    }

    // Calculate RMSE and write comparison files.
    calculateRmse(true,
                  identifier,
                  fileNameEnergiesTrain,
                  fileNameEnergiesTest,
                  fileNameForcesTrain,
                  fileNameForcesTest);

    return;
}

void Training::writeWeights(string const fileNameFormat) const
{
    ofstream file;

    for (size_t i = 0; i < numElements; ++i)
    {
        string fileName = strpr(fileNameFormat.c_str(),
                                elements.at(i).getAtomicNumber());
        file.open(fileName.c_str());
        elements.at(i).neuralNetwork->writeConnections(file);
        file.close();
    }

    return;
}

void Training::writeWeightsEpoch() const
{
    string fileNameFormat = strpr("weights.%%03zu.%06d.out", epoch);

    if (writeWeightsEvery > 0 &&
        (epoch % writeWeightsEvery == 0 || epoch <= writeWeightsAlways))
    {
        writeWeights(fileNameFormat);
    }

    return;
}

void Training::writeLearningCurve(bool append, string const fileName) const
{
    ofstream file;

    if (append) file.open(fileName.c_str(), ofstream::app);
    else 
    {
        file.open(fileName.c_str());

        // File header.
        vector<string> title;
        vector<string> colName;
        vector<string> colInfo;
        vector<size_t> colSize;
        title.push_back("Learning curves for energies and forces.");
        colSize.push_back(10);
        colName.push_back("epoch");
        colInfo.push_back("Current epoch.");
        colSize.push_back(16);
        colName.push_back("rmse_Etrain_phys");
        colInfo.push_back("RMSE of training energies per atom (physical "
                          "units).");
        colSize.push_back(16);
        colName.push_back("rmse_Etest_phys");
        colInfo.push_back("RMSE of test energies per atom (physical units).");
        colSize.push_back(16);
        colName.push_back("rmse_Ftrain_phys");
        colInfo.push_back("RMSE of training forces (physical units).");
        colSize.push_back(16);
        colName.push_back("rmse_Ftest_phys");
        colInfo.push_back("RMSE of test forces (physical units).");
        if (normalize)
        {
            colSize.push_back(16);
            colName.push_back("rmse_Etrain_int");
            colInfo.push_back("RMSE of training energies per atom (internal "
                              "units).");
            colSize.push_back(16);
            colName.push_back("rmse_Etest_int");
            colInfo.push_back("RMSE of test energies per atom (internal units).");
            colSize.push_back(16);
            colName.push_back("rmse_Ftrain_int");
            colInfo.push_back("RMSE of training forces (internal units).");
            colSize.push_back(16);
            colName.push_back("rmse_Ftest_int");
            colInfo.push_back("RMSE of test forces (internal units).");
        }
        appendLinesToFile(file,
                          createFileHeader(title, colSize, colName, colInfo));
    }

    file << strpr("%10zu", epoch);
    if (normalize)
    {
        file << strpr(" %16.8E %16.8E %16.8E %16.8E",
                      physicalEnergy(rmseEnergiesTrain),
                      physicalEnergy(rmseEnergiesTest),
                      physicalForce(rmseForcesTrain),
                      physicalForce(rmseForcesTest));
    }
    file << strpr(" %16.8E %16.8E %16.8E %16.8E\n",
                  rmseEnergiesTrain,
                  rmseEnergiesTest,
                  rmseForcesTrain,
                  rmseForcesTest);
    file.close();

    return;
}

void Training::writeNeuronStatistics(string const fileName) const
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
        title.push_back("Statistics for individual neurons gathered during "
                        "RMSE calculation.");
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
        size_t n = elements.at(i).neuralNetwork->getNumNeurons();
        vector<long>   count(n, 0);
        vector<double> min(n, 0.0);
        vector<double> max(n, 0.0);
        vector<double> mean(n, 0.0);
        vector<double> sigma(n, 0.0);
        elements.at(i).neuralNetwork->getNeuronStatistics(&(count.front()),
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
    if (writeNeuronStatisticsEvery > 0 &&
        (epoch % writeNeuronStatisticsEvery == 0
        || epoch <= writeNeuronStatisticsAlways))
    {
        string fileName = strpr("neuron-stats.%06zu.out", epoch);
        writeNeuronStatistics(fileName);
    }

    return;
}

void Training::resetNeuronStatistics() const
{
    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->neuralNetwork->resetNeuronStatistics();
    }
    return;
}

void Training::writeUpdaterStatus(bool         append,
                                  string const fileNameFormat) const
{
    ofstream file;

    for (size_t i = 0; i < numUpdaters; ++i)
    {
        string fileName;
        if (updateStrategy == US_COMBINED)
        {
            fileName = strpr(fileNameFormat.c_str(), 0);
        }
        else if (updateStrategy == US_ELEMENT)
        {
            fileName = strpr(fileNameFormat.c_str(),
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

void Training::sortUpdateCandidates()
{
    // Update error for all structures.
    for (vector<UpdateCandidate>::iterator it = updateCandidatesEnergy.begin();
         it != updateCandidatesEnergy.end(); ++it)
    {
        Structure const& s = structures.at(it->s);
        it->error = fabs((s.energyRef - s.energy) / s.numAtoms);
    }
    // Sort energy update candidates list.
    sort(updateCandidatesEnergy.begin(), updateCandidatesEnergy.end());
    // Reset current position.
    posUpdateCandidatesEnergy = 0;

    if (useForces)
    {
        // Update error for all forces.
        for (vector<UpdateCandidate>::iterator it =
             updateCandidatesForce.begin();
             it != updateCandidatesForce.end(); ++it)
        {
            Atom const& a = structures.at(it->s).atoms.at(it->a);
            it->error = fabs(a.fRef[it->c] - a.f[it->c]);
        }
        // Sort force update candidate list.
        sort(updateCandidatesForce.begin(), updateCandidatesForce.end());
        // Reset current position.
        posUpdateCandidatesForce = 0;
    }

    return;
}

void Training::shuffleUpdateCandidates()
{
    shuffle(updateCandidatesEnergy.begin(),
            updateCandidatesEnergy.end(),
            rngNew);

    // Reset current position.
    posUpdateCandidatesEnergy = 0;

    if (useForces)
    {
        shuffle(updateCandidatesForce.begin(),
                updateCandidatesForce.end(),
                rngNew);

        // Reset current position.
        posUpdateCandidatesForce = 0;
    }

    return;
}

void Training::setEpochSchedule()
{
    // Clear epoch schedule.
    epochSchedule.clear();
    vector<int>(epochSchedule).swap(epochSchedule);

    // Set size of schedule and initialize to 0 (i.e. all energy updates).
    epochSchedule.resize(energyUpdates + forceUpdates, 0);

    // Fill with as many "1"s as there are force updates.
    fill(epochSchedule.begin(), epochSchedule.begin() + forceUpdates, 1);

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
    if (selectionModeSchedule.find(epoch) != selectionModeSchedule.end())
    {
        selectionMode = selectionModeSchedule[epoch];
        if (epoch != 0)
        {
            string message = "INFO   Switching selection mode to ";
            if (selectionMode == SM_RANDOM)
            {
                message += strpr("SM_RANDOM (%d).\n", selectionMode);
            }
            else if (selectionMode == SM_SORT)
            {
                message += strpr("SM_SORT (%d).\n", selectionMode);
            }
            else if (selectionMode == SM_THRESHOLD)
            {
                message += strpr("SM_THRESHOLD (%d).\n", selectionMode);
            }
            log << message;
        }
    }

    return;
}

void Training::loop()
{
    log << "\n";
    log << "*** TRAINING LOOP ***********************"
           "**************************************\n";
    log << "\n";

    log << "The training loop output covers different RMSEs, update and\n";
    log << "timing information. The following quantities are organized\n";
    log << "according to the matrix scheme below:\n";
    log << "-------------------------------------------------------------------\n";
    log << "ep ............ Epoch.\n";
    log << "Etrain_phys ... RMSE of training energies per atom (p. u.).\n";
    log << "Etest_phys .... RMSE of test     energies per atom (p. u.).\n";
    log << "Etrain_int .... RMSE of training energies per atom (i. u.).\n";
    log << "Etest_int ..... RMSE of test     energies per atom (i. u.).\n";
    log << "Ftrain_phys ... RMSE of training forces (p. u.).\n";
    log << "Ftest_phys .... RMSE of test     forces (p. u.).\n";
    log << "Ftrain_int .... RMSE of training forces (i. u.).\n";
    log << "Ftest_int ..... RMSE of test     forces (i. u.).\n";
    log << "E_count ....... Number of energy updates.\n";
    log << "F_count ....... Number of force  updates.\n";
    log << "count ......... Total number of updates.\n";
    log << "t_train ....... Time for training (seconds).\n";
    log << "t_rmse ........ Time for RMSE calculation (seconds).\n";
    log << "t_epoch ....... Total time for this epoch (seconds).\n";
    log << "t_tot ......... Total time for all epochs (seconds).\n";
    log << "Abbreviations:\n";
    log << "  p. u. = physical units.\n";
    log << "  i. u. = internal units.\n";
    log << "Note: RMSEs in internal units (columns 5 + 6) are only present \n";
    log << "      if data set normalization is used.\n";
    log << "-------------------------------------------------------------------\n";
    log << "     1    2             3             4             5             6\n";
    log << "energy   ep   Etrain_phys    Etest_phys    Etrain_int     Etest_int\n";
    log << "forces   ep   Ftrain_phys    Ftest_phys    Ftrain_int     Ftest_int\n";
    log << "update   ep       E_count       F_count         count\n";
    log << "timing   ep       t_train        t_rmse       t_epoch         t_tot\n";
    log << "-------------------------------------------------------------------\n";

    // Set up stopwatch.
    double timeSplit;
    double timeTotal;
    Stopwatch swTotal;
    Stopwatch swTrain;
    Stopwatch swRmse;
    swTotal.start();

    // Calculate initial RMSE and write comparison files.
    swRmse.start();
    calculateRmseEpoch();
    swRmse.stop();

    // Write initial weights to files.
    if (myRank == 0) writeWeightsEpoch();

    // Write learning curve.
    if (myRank == 0) writeLearningCurve(false);

    // Write updater status to file.
    //if (myRank == 0) writeUpdaterStatus(false);

    // Write neuron statistics.
    writeNeuronStatisticsEpoch();

    // Print timing information.
    timeTotal = swTotal.split(&timeSplit);
    log << strpr("TIMING %4zu %13.2f %13.2f %13.2f %13.2f\n",
                 epoch,
                 swTrain.getTimeElapsed(),
                 swRmse.getTimeElapsed(),
                 timeSplit,
                 timeTotal);

    // Check if training should be continued.
    while (advance())
    {
        // Increment epoch counter.
        epoch++;
        log << "------\n";

        // Reset timers.
        swTrain.reset();
        swRmse.reset();

        // Reset update counters.
        size_t numUpdatesEnergy = 0;
        size_t numUpdatesForce = 0;

        // Check if selection mode should be changed in this epoch.
        checkSelectionMode();

        // Sort or shuffle update candidates.
        if (selectionMode == SM_SORT) sortUpdateCandidates();
        else shuffleUpdateCandidates();

        // Determine epoch update schedule.
        setEpochSchedule();

        // Perform energy/force updates according to schedule.
        swTrain.start();
        for (size_t i = 0; i < epochSchedule.size(); ++i)
        {
            bool force = static_cast<bool>(epochSchedule.at(i));
            update(force);
            if (force) numUpdatesForce++;
            else       numUpdatesEnergy++;
        }
        swTrain.stop();

        // Reset neuron statistics.
        resetNeuronStatistics();

        // Calculate RMSE and write comparison files.
        swRmse.start();
        calculateRmseEpoch();
        swRmse.stop();

        // Print update information.
        log << strpr("UPDATE %4zu %13zu %13zu %13zu\n",
                     epoch,
                     numUpdatesEnergy,
                     numUpdatesForce,
                     numUpdatesEnergy + numUpdatesForce);

        // Write weights to files.
        if (myRank == 0) writeWeightsEpoch();

        // Append to learning curve.
        if (myRank == 0) writeLearningCurve(true);

        // Write updater status to file.
        //if (myRank == 0) writeUpdaterStatus(true);

        // Write neuron statistics.
        writeNeuronStatisticsEpoch();

        // Print timing information.
        timeTotal = swTotal.split(&timeSplit);
        log << strpr("TIMING %4zu %13.2f %13.2f %13.2f %13.2f\n",
                     epoch,
                     swTrain.getTimeElapsed(),
                     swRmse.getTimeElapsed(),
                     timeSplit,
                     timeTotal);
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Training::update(bool force)
{
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    ///////////////////////////////////////////////////////////////////////
    // PART 1: Calculate errors and derivatives
    ///////////////////////////////////////////////////////////////////////

    // Set local variables depending on energy/force update.
    size_t batchSize = 0;
    size_t* posUpdateCandidates = NULL;
    vector<int>* errorsPerTask = NULL;
    vector<int>* offsetPerTask = NULL;
    vector<vector<int> >* weightsPerTask = NULL;
    vector<vector<int> >* offsetJacobian = NULL;
    vector<vector<double> >* error = NULL;
    vector<vector<double> >* jacobian = NULL;
    vector<UpdateCandidate>* updateCandidates = NULL;
    if (force)
    {
        batchSize = taskBatchSizeForce;
        if (batchSize == 0) batchSize = forcesPerUpdate;
        posUpdateCandidates = &posUpdateCandidatesForce;
        errorsPerTask = &errorsPerTaskForce;
        offsetPerTask = &offsetPerTaskForce;
        weightsPerTask = &weightsPerTaskForce;
        offsetJacobian = &offsetJacobianForce;
        error = &errorF;
        jacobian = &jacobianF;
        updateCandidates = &updateCandidatesForce;
    }
    else
    {
        batchSize = taskBatchSizeEnergy;
        if (batchSize == 0) batchSize = energiesPerUpdate;
        posUpdateCandidates = &posUpdateCandidatesEnergy;
        errorsPerTask = &errorsPerTaskEnergy;
        offsetPerTask = &offsetPerTaskEnergy;
        weightsPerTask = &weightsPerTaskEnergy;
        offsetJacobian = &offsetJacobianEnergy;
        error = &errorE;
        jacobian = &jacobianE;
        updateCandidates = &updateCandidatesEnergy;
    }
    vector<size_t> thresholdLoopCount(batchSize, 0);
    vector<double> currentRmseFraction(batchSize, 0.0);
    vector<UpdateCandidate*> currentUpdateCandidates(batchSize, NULL);
    for (size_t i = 0; i < numUpdaters; ++i)
    {
        fill(error->at(i).begin(), error->at(i).end(), 0.0);
        fill(jacobian->at(i).begin(), jacobian->at(i).end(), 0.0);
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
        if (selectionMode == SM_THRESHOLD) trials = rmseThresholdTrials;
        size_t il = 0;
        for (il = 0; il < trials; ++il)
        {
            // Restart position index if necessary.
            if (*posUpdateCandidates >= updateCandidates->size())
            {
                *posUpdateCandidates = 0;
            }

            //log << strpr("pos %zu b %zu size %zu\n", *posUpdateCandidates, b, currentUpdateCandidates.size());
            // Set current update candidate.
            c = &(updateCandidates->at(*posUpdateCandidates));
            // Keep update candidates (for logging later).
            currentUpdateCandidates.at(b) = c;
            // Shortcut for current structure.
            Structure& s = structures.at(c->s);
            // Calculate symmetry functions (if results are already stored
            // these functions will return immediately).
#ifdef NOSFGROUPS
            calculateSymmetryFunctions(s, force);
#else
            calculateSymmetryFunctionGroups(s, force);
#endif
            // For SM_THRESHOLD calculate RMSE of update candidate.
            if (selectionMode == SM_THRESHOLD)
            {
                calculateAtomicNeuralNetworks(s, force);
                if (force)
                {
                    calculateForces(s);
                    Atom const& a = s.atoms.at(c->a);
                    currentRmseFraction.at(b)
                        = fabs(a.fRef[c->c] - a.f[c->c]) / rmseForcesTrain;
                    // If force RMSE is above threshold stop loop immediately.
                    if (currentRmseFraction.at(b) > rmseThresholdForce)
                    {
                        // Increment position in update candidate list.
                        (*posUpdateCandidates)++;
                        break;
                    }
                }
                else
                {
                    calculateEnergy(s);
                    currentRmseFraction.at(b)
                        = fabs(s.energyRef - s.energy)
                        / (s.numAtoms * rmseEnergiesTrain);
                    // If energy RMSE is above threshold stop loop immediately.
                    if (currentRmseFraction.at(b) > rmseThresholdEnergy)
                    {
                        // Increment position in update candidate list.
                        (*posUpdateCandidates)++;
                        break;
                    }
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
                    indexBest = *posUpdateCandidates;
                }
                // Increment position in update candidate list.
                (*posUpdateCandidates)++;
            }
            // Break loop for all selection modes but SM_THRESHOLD.
            else if (selectionMode == SM_RANDOM || selectionMode == SM_SORT)
            {
                // Increment position in update candidate list.
                (*posUpdateCandidates)++;
                break;
            }
        }
        thresholdLoopCount.at(b) = il;

        // If loop was not stopped because of a proper update candidate found
        // (RMSE above threshold) use best candidate during iteration.
        if (selectionMode == SM_THRESHOLD && il == trials)
        {
            // Set best candidate.
            currentUpdateCandidates.at(b) = &(updateCandidates->at(indexBest));
            currentRmseFraction.at(b) = rmseFractionBest;
            // Need to calculate the symmetry functions again, maybe results
            // were not stored.
            Structure& s = structures.at(c->s);
#ifdef NOSFGROUPS
            calculateSymmetryFunctions(s, force);
#else
            calculateSymmetryFunctionGroups(s, force);
#endif
        }

        ///////////////////////////////////////////////////////////////////////
        // PART 2: Compute error vector and Jacobian
        ///////////////////////////////////////////////////////////////////////
 
        //log << strpr("%zu %s", b, force ? "F" : "E");
        //if (force)
        //{
        //    log << strpr(" %zu %zu %zu %zu %f\n",
        //                 posUpdateCandidatesForce,
        //                 c->s, c->a, c->c, rmseFractionBest);
        //}
        //else {
        //    log << strpr(" %zu %zu %f\n",
        //                 posUpdateCandidatesEnergy,
        //                 c->s, rmseFractionBest);
        //}
 
        Structure& s = structures.at(c->s);
        // Temporary storage for derivative contributions of atoms (dXdc stores
        // dEdc or dFdc for energy or force update, respectively.
        vector<vector<double> > dXdc;
        dXdc.resize(numElements);
        for (size_t i = 0; i < numElements; ++i)
        {
            size_t n = elements.at(i).neuralNetwork->getNumConnections();
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
                offset.at(i) += offsetPerTask->at(myRank)
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
        // Loop over atoms and calculate atomic energy contributions.
        for (vector<Atom>::iterator it = s.atoms.begin();
             it != s.atoms.end(); ++it)
        {
            // For force update save derivative of symmetry function with
            // respect to coordinate.
            if (force) it->collectDGdxia(c->a, c->c);
            size_t i = it->element;
            NeuralNetwork* const& nn = elements.at(i).neuralNetwork;
            nn->setInput(&((it->G).front()));
            nn->propagate();
            if (force) nn->calculateDEdG(&((it->dEdG).front()));
            nn->getOutput(&(it->energy));
            // Compute derivative of output node with respect to all neural
            // network connections (weights + biases).
            if (force)
            {
                nn->calculateDFdc(&(dXdc.at(i).front()),
                                  &(it->dGdxia.front()));
            }
            else
            {
                nn->calculateDEdc(&(dXdc.at(i).front()));
            }
            // Finally sum up Jacobian.
            if (updateStrategy == US_ELEMENT) iu = i;
            else iu = 0;
            for (size_t j = 0; j < dXdc.at(i).size(); ++j)
            {
                jacobian->at(iu).at(offset.at(i) + j) += dXdc.at(i).at(j);
            }
        }
 
        // Sum up total potential energy or calculate force.
        if (force)
        {
            calculateForces(s);
            Atom const& a = s.atoms.at(c->a);
            currentRmseFraction.at(b) = fabs(a.fRef[c->c] - a.f[c->c])
                                      / rmseForcesTrain;
        }
        else
        {
            calculateEnergy(s);
            currentRmseFraction.at(b) = fabs(s.energyRef - s.energy)
                                      / (s.numAtoms * rmseEnergiesTrain);
        }

        // Now symmetry function memory is not required any more for this
        // update.
        if (freeMemory) s.freeAtoms(true);

        // Precalculate offset in error array.
        size_t offset2 = 0;
        if (parallelMode == PM_TRAIN_ALL && jacobianMode != JM_SUM)
        {
            offset2 += offsetPerTask->at(myRank);
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
            if (force)
            {
                Atom const& a = s.atoms.at(c->a);
                error->at(0).at(offset2) +=  a.fRef[c->c] - a.f[c->c];
            }
            else
            {
                error->at(0).at(offset2) += s.energyRef - s.energy;
            }
        }
        else if (updateStrategy == US_ELEMENT)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                if (force)
                {
                    Atom const& a = s.atoms.at(c->a);
                    error->at(i).at(offset2) += (a.fRef[c->c] - a.f[c->c])
                                              * a.numNeighborsPerElement.at(i)
                                              / a.numNeighbors;
                }
                else
                {
                    error->at(i).at(offset2) += (s.energyRef - s.energy)
                                              * s.numAtomsPerElement.at(i)
                                              / s.numAtoms;
                }
            }
        }
    }

    // Apply force update weight to error and Jacobian.
    if (force)
    {
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            for (size_t j = 0; j < error->at(i).size(); ++j)
            {
                error->at(i).at(j) *= forceWeight;
            }
            for (size_t j = 0; j < jacobian->at(i).size(); ++j)
            {
                jacobian->at(i).at(j) *= forceWeight;
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////
    // PART 3: Communicate error and Jacobian.
    ///////////////////////////////////////////////////////////////////////

    if (jacobianMode == JM_SUM)
    {
        if (parallelMode == PM_TRAIN_RK0)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                if (myRank == 0) MPI_Reduce(MPI_IN_PLACE           , &(error->at(i).front()), 1, MPI_DOUBLE, MPI_SUM, 0, comm);
                else             MPI_Reduce(&(error->at(i).front()), &(error->at(i).front()), 1, MPI_DOUBLE, MPI_SUM, 0, comm);
                if (myRank == 0) MPI_Reduce(MPI_IN_PLACE              , &(jacobian->at(i).front()), numWeightsPerUpdater.at(i), MPI_DOUBLE, MPI_SUM, 0, comm);
                else             MPI_Reduce(&(jacobian->at(i).front()), &(jacobian->at(i).front()), numWeightsPerUpdater.at(i), MPI_DOUBLE, MPI_SUM, 0, comm);
            }
        }
        else if (parallelMode == PM_TRAIN_ALL)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                MPI_Allreduce(MPI_IN_PLACE, &(error->at(i).front()), 1, MPI_DOUBLE, MPI_SUM, comm);
                MPI_Allreduce(MPI_IN_PLACE, &(jacobian->at(i).front()), numWeightsPerUpdater.at(i), MPI_DOUBLE, MPI_SUM, comm);
            }
        }
    }
    else if (jacobianMode == JM_TASK)
    {
        if (parallelMode == PM_TRAIN_RK0)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                if (myRank == 0) MPI_Gather(MPI_IN_PLACE           , 1, MPI_DOUBLE, &(error->at(i).front()),  1, MPI_DOUBLE, 0, comm);
                else             MPI_Gather(&(error->at(i).front()), 1, MPI_DOUBLE, NULL                   ,  1, MPI_DOUBLE, 0, comm);
                if (myRank == 0) MPI_Gather(MPI_IN_PLACE              , numWeightsPerUpdater.at(i), MPI_DOUBLE, &(jacobian->at(i).front()), numWeightsPerUpdater.at(i), MPI_DOUBLE, 0, comm);
                else             MPI_Gather(&(jacobian->at(i).front()), numWeightsPerUpdater.at(i), MPI_DOUBLE, NULL                      , numWeightsPerUpdater.at(i), MPI_DOUBLE, 0, comm);
            }
        }
        else if (parallelMode == PM_TRAIN_ALL)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                MPI_Allgather(MPI_IN_PLACE, 1, MPI_DOUBLE, &(error->at(i).front()),  1, MPI_DOUBLE, comm);
                MPI_Allgather(MPI_IN_PLACE, numWeightsPerUpdater.at(i), MPI_DOUBLE, &(jacobian->at(i).front()), numWeightsPerUpdater.at(i), MPI_DOUBLE, comm);
            }
        }
    }
    else if (jacobianMode == JM_FULL)
    {
        if (parallelMode == PM_TRAIN_RK0)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                if (myRank == 0) MPI_Gatherv(MPI_IN_PLACE           , 0                        , MPI_DOUBLE, &(error->at(i).front()), &(errorsPerTask->front()), &(offsetPerTask->front()), MPI_DOUBLE, 0, comm);
                else             MPI_Gatherv(&(error->at(i).front()), errorsPerTask->at(myRank), MPI_DOUBLE, NULL                   , NULL                     , NULL                     , MPI_DOUBLE, 0, comm);
                if (myRank == 0) MPI_Gatherv(MPI_IN_PLACE              , 0                               , MPI_DOUBLE, &(jacobian->at(i).front()), &(weightsPerTask->at(i).front()), &(offsetJacobian->at(i).front()), MPI_DOUBLE, 0, comm);
                else             MPI_Gatherv(&(jacobian->at(i).front()), weightsPerTask->at(i).at(myRank), MPI_DOUBLE, NULL                      , NULL                            , NULL                            , MPI_DOUBLE, 0, comm);
            }
        }
        else if (parallelMode == PM_TRAIN_ALL)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, &(error->at(i).front()), &(errorsPerTask->front()), &(offsetPerTask->front()), MPI_DOUBLE, comm);
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DOUBLE, &(jacobian->at(i).front()), &(weightsPerTask->at(i).front()), &(offsetJacobian->at(i).front()), MPI_DOUBLE, comm);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////
    // PART 4: Perform weight update and apply new weights.
    ///////////////////////////////////////////////////////////////////////

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif
    // Loop over all updaters.
    for (size_t i = 0; i < updaters.size(); ++i)
    {
        updaters.at(i)->setError(&(error->at(i).front()), error->at(i).size());
        updaters.at(i)->setJacobian(&(jacobian->at(i).front()),
                                    error->at(i).size());
        if (updaterType == UT_KF) 
        {
            KalmanFilter* kf = dynamic_cast<KalmanFilter*>(updaters.at(i));
            kf->setSizeObservation(error->at(i).size());
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

    ///////////////////////////////////////////////////////////////////////
    // PART 5: Communicate candidates and RMSE fractions and write log.
    ///////////////////////////////////////////////////////////////////////

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
            if (force)
            {
                MPI_Gatherv(MPI_IN_PLACE, 0, MPI_SIZE_T, &(indexAtom.front())           , &(currentUpdateCandidatesPerTask.front()), &(currentUpdateCandidatesOffset.front()), MPI_SIZE_T, 0, comm);
                MPI_Gatherv(MPI_IN_PLACE, 0, MPI_SIZE_T, &(indexCoordinate.front())     , &(currentUpdateCandidatesPerTask.front()), &(currentUpdateCandidatesOffset.front()), MPI_SIZE_T, 0, comm);
            }
        }
        else
        {
            MPI_Gatherv(&(currentRmseFraction.front()) , myCurrentUpdateCandidates, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, comm);
            MPI_Gatherv(&(thresholdLoopCount.front())  , myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
            MPI_Gatherv(&(procUpdateCandidate.front()) , myCurrentUpdateCandidates, MPI_INT   , NULL, NULL, NULL, MPI_INT   , 0, comm);
            MPI_Gatherv(&(indexStructure.front())      , myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
            MPI_Gatherv(&(indexStructureGlobal.front()), myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
            if (force)
            {
                MPI_Gatherv(&(indexAtom.front())           , myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
                MPI_Gatherv(&(indexCoordinate.front())     , myCurrentUpdateCandidates, MPI_SIZE_T, NULL, NULL, NULL, MPI_SIZE_T, 0, comm);
            }
        }

        if (myRank == 0)
        {
            for (size_t i = 0; i < procUpdateCandidate.size(); ++i)
            {
                if (force)
                {
                    addTrainingLogEntry(procUpdateCandidate.at(i),
                                        thresholdLoopCount.at(i),
                                        currentRmseFraction.at(i),
                                        indexStructureGlobal.at(i),
                                        indexStructure.at(i),
                                        indexAtom.at(i),
                                        indexCoordinate.at(i));
                }
                else
                {
                    addTrainingLogEntry(procUpdateCandidate.at(i),
                                        thresholdLoopCount.at(i),
                                        currentRmseFraction.at(i),
                                        indexStructureGlobal.at(i),
                                        indexStructure.at(i));
                }
            }
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
vector<double> > Training::calculateWeightDerivatives(Structure* structure)
{
    Structure& s = *structure;
#ifdef NOSFGROUPS
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
        size_t n = elements.at(i).neuralNetwork->getNumConnections();
        dEdc.at(i).resize(n, 0.0);
        dedc.at(i).resize(n, 0.0);
    }
    for (vector<Atom>::iterator it = s.atoms.begin();
         it != s.atoms.end(); ++it)
    {
        size_t i = it->element;
        NeuralNetwork* const& nn = elements.at(i).neuralNetwork;
        nn->setInput(&((it->G).front()));
        nn->propagate();
        nn->getOutput(&(it->energy));
        nn->calculateDEdc(&(dedc.at(i).front()));
        for (size_t j = 0; j < dedc.at(i).size(); ++j)
        {
            dEdc.at(i).at(j) += dedc.at(i).at(j);
        }
    }

    return dEdc;
}

// Doxygen requires namespace prefix for arguments...
vector<
vector<double> > Training::calculateWeightDerivatives(Structure*  structure,
                                                      std::size_t atom,
                                                      std::size_t component)
{
    Structure& s = *structure;
#ifdef NOSFGROUPS
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
        size_t n = elements.at(i).neuralNetwork->getNumConnections();
        dFdc.at(i).resize(n, 0.0);
        dfdc.at(i).resize(n, 0.0);
    }
    for (vector<Atom>::iterator it = s.atoms.begin();
         it != s.atoms.end(); ++it)
    {
        it->collectDGdxia(atom, component);
        size_t i = it->element;
        NeuralNetwork* const& nn = elements.at(i).neuralNetwork;
        nn->setInput(&((it->G).front()));
        nn->propagate();
        nn->getOutput(&(it->energy));
        nn->calculateDFdc(&(dfdc.at(i).front()), &(it->dGdxia.front()));
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
            NeuralNetwork const* const& nn = elements.at(i).neuralNetwork;
            nn->getConnections(&(weights.at(0).at(pos)));
            pos += nn->getNumConnections();
        }
    }
    else if (updateStrategy == US_ELEMENT)
    {
        for (size_t i = 0; i < numElements; ++i)
        {
            NeuralNetwork const* const& nn = elements.at(i).neuralNetwork;
            nn->getConnections(&(weights.at(i).front()));
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
            NeuralNetwork* const& nn = elements.at(i).neuralNetwork;
            nn->setConnections(&(weights.at(0).at(pos)));
            pos += nn->getNumConnections();
        }
    }
    else if (updateStrategy == US_ELEMENT)
    {
        for (size_t i = 0; i < numElements; ++i)
        {
            NeuralNetwork* const& nn = elements.at(i).neuralNetwork;
            nn->setConnections(&(weights.at(i).front()));
        }
    }

    return;
}
