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
#include <algorithm> // std::sort, std::fill
#include <cmath>     // fabs
#include <cstdlib>   // atoi
#include <gsl/gsl_rng.h>
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error, std::range_error

using namespace std;
using namespace nnp;

Training::Training() : Dataset(),
                       updaterType                (UT_GRADIENTDESCENT),
                       parallelMode               (PM_SERIAL         ),
                       updateStrategy             (US_COMBINED       ),
                       selectionMode              (SM_RANDOM         ),
                       hasUpdaters                (false             ),
                       hasStructures              (false             ),
                       useForces                  (false             ),
                       reapeatedEnergyUpdates     (false             ),
                       freeMemory                 (false             ),
                       writeTrainingLog           (false             ),
                       myStream                   (0                 ),
                       numStreams                 (0                 ),
                       numUpdaters                (0                 ),
                       numEnergiesTrain           (0                 ),
                       numForcesTrain             (0                 ),
                       numEpochs                  (0                 ),
                       epoch                      (0                 ),
                       writeEnergiesEvery         (0                 ),
                       writeForcesEvery           (0                 ),
                       writeWeightsEvery          (0                 ),
                       writeNeuronStatisticsEvery (0                 ),
                       writeEnergiesAlways        (0                 ),
                       writeForcesAlways          (0                 ),
                       writeWeightsAlways         (0                 ),
                       writeNeuronStatisticsAlways(0                 ),
                       posUpdateCandidatesEnergy  (0                 ),
                       posUpdateCandidatesForce   (0                 ),
                       rmseThresholdTrials        (0                 ),
                       countUpdates               (0                 ),
                       epochFractionEnergies      (0.0               ),
                       epochFractionForces        (0.0               ),
                       rmseEnergiesTrain          (0.0               ),
                       rmseEnergiesTest           (0.0               ),
                       rmseForcesTrain            (0.0               ),
                       rmseForcesTest             (0.0               ),
                       rmseThresholdEnergy        (0.0               ),
                       rmseThresholdForce         (0.0               ),
                       forceWeight                (0.0               ),
                       trainingLogFileName        ("train-log.out"   )
{
}

Training::~Training()
{
    for (vector<Updater*>::iterator it = updaters.begin();
         it != updaters.end(); ++it)
    {
        if (updaterType == UT_GRADIENTDESCENT)
        {
            delete dynamic_cast<GradientDescent*>(*it);
        }
        else if (updaterType == UT_KALMANFILTER)
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
    if (updateStrategy == US_COMBINED)
    {
        log << strpr("Combined updater for all elements selected: "
                     "UpdateStrategy::US_COMBINED (%d)\n", updateStrategy);
        numUpdaters = 1;
        log << strpr("Number of weight updaters    : %zu\n", numUpdaters);
        size_t n = 0;
        for (size_t i = 0; i < numElements; ++i)
        {
            n += elements.at(i).neuralNetwork->getNumConnections();
        }
        weights.resize(numUpdaters);
        weights.at(0).resize(n, 0.0);
        log << strpr("Total fit parameters         : %zu\n", n);
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
    if (updaterType == UT_GRADIENTDESCENT)
    {
        log << strpr("Weight update via gradient descent selected: "
                     "updaterType::UT_GRADIENTDESCENT (%d)\n",
                     updaterType);
    }
    else if (updaterType == UT_KALMANFILTER)
    {
        log << strpr("Weight update via Kalman filter selected: "
                     "updaterType::UT_KALMANFILTER (%d)\n",
                     updaterType);
    }
    else
    {
        throw runtime_error("ERROR: Unknown updater type.\n");
    }

    parallelMode = (ParallelMode)atoi(settings["parallel_mode"].c_str());
    if (parallelMode == PM_SERIAL)
    {
        log << strpr("Serial training selected: "
                     "ParallelMode::PM_SERIAL (%d)\n",
                     parallelMode);
        numStreams = 1;
        myStream = 0;
        log << strpr("Number of streams       : %zu\n", numStreams);
        log << strpr("Stream of this processor: %zu\n", myStream);
    }
    else if (parallelMode == PM_MSEKF ||
             parallelMode == PM_MSEKFNB ||
             parallelMode == PM_MSEKFR0 ||
             parallelMode == PM_MSEKFR0PX)
    {
        if (updaterType == UT_GRADIENTDESCENT)
        {
            throw runtime_error("ERROR: Multi-stream training with "
                                "gradient descent weight updater not "
                                "implemented.\n");
        }
        if (parallelMode == PM_MSEKF)
        {
            log << strpr("Multi-stream Kalman filter training selected: "
                         "ParallelMode::PM_MSEKF (%d)\n",
                         parallelMode);
        }
        else if (parallelMode == PM_MSEKFNB)
        {
            log << strpr("Multi-stream Kalman filter training with "
                         "non-blocking communication selected: "
                         "ParallelMode::PM_MSEKFNB (%d)\n",
                         parallelMode);
        }
        else if (parallelMode == PM_MSEKFR0)
        {
            log << strpr("Multi-stream Kalman filter training with update on "
                         "rank 0 selected: ParallelMode::PM_MSEKFR0 (%d)\n",
                         parallelMode);
        }
        else if (parallelMode == PM_MSEKFR0PX)
        {
            log << strpr("Multi-stream Kalman filter training, update on rank "
                         "0, partial X calculation selected: "
                         "ParallelMode::PM_MSEKFR0PX (%d)\n",
                         parallelMode);
        }
        numStreams = numProcs;
        myStream = myRank;
        log << strpr("Number of streams       : %zu\n", numStreams);
        log << strpr("Stream of this processor: %zu\n", myStream);
    }
    else
    {
        throw runtime_error("ERROR: Unknown parallelization mode.\n");
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
        colInfo.push_back("MPI process updating with this information.");
        colSize.push_back(3);
        colName.push_back("tl");
        colInfo.push_back("Threshold loop counter.");
        colSize.push_back(10);
        colName.push_back("rmse_frac");
        colInfo.push_back("Update candidates error divided by this "
                          "epochs RMSE.");
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            colSize.push_back(10);
            colName.push_back(strpr("absXi_%zu", i + 1));
            colInfo.push_back(strpr("Absolute value of error for updater %zu.",
                                    i + 1));
            colSize.push_back(10);
            colName.push_back(strpr("meanH_%zu", i + 1));
            colInfo.push_back(strpr("Mean magnitude of derivative vector for "
                                    "updater %zu.", i + 1));
        }
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

    log << "-----------------------------------------"
           "--------------------------------------\n";
    epochFractionEnergies = atof(settings["short_energy_fraction"].c_str());
    log << strpr("Fraction of energies used per epoch:    %.4f\n",
                 epochFractionEnergies);
    if (useForces)
    {
        epochFractionForces = atof(settings["short_force_fraction"].c_str());
        log << strpr("Fraction of forces used per epoch  :    %.4f\n",
                     epochFractionForces);
    }
    double projectedEnergyUpdates = updateCandidatesEnergy.size()
                                  * epochFractionEnergies;
    double projectedForceUpdates = 0.0;
    if (useForces)
    {
        projectedForceUpdates = updateCandidatesForce.size()
                              * epochFractionForces;
    }
    if (reapeatedEnergyUpdates)
    {
        projectedEnergyUpdates += projectedForceUpdates;
    }
    MPI_Allreduce(MPI_IN_PLACE, &projectedEnergyUpdates, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &projectedForceUpdates , 1, MPI_DOUBLE, MPI_SUM, comm);
    if (parallelMode == PM_MSEKF ||
        parallelMode == PM_MSEKFNB ||
        parallelMode == PM_MSEKFR0 ||
        parallelMode == PM_MSEKFR0PX)
    {
        projectedEnergyUpdates /= numStreams;
        projectedForceUpdates /= numStreams;
    }
    double totalUpdates = projectedEnergyUpdates + projectedForceUpdates;
    log << strpr("Projected energy updates per epoch : %7.0f (%5.1f%%)\n",
                 projectedEnergyUpdates,
                 100.0 * projectedEnergyUpdates / totalUpdates);
    if (useForces)
    {
        log << strpr("Projected forces updates per epoch : %7.0f (%5.1f%%)\n",
                     projectedForceUpdates,
                     100.0 * projectedForceUpdates / totalUpdates);
        log << strpr("Total projected updates per epoch  : %7.0f\n",
                     totalUpdates);
    }
    if (parallelMode == PM_MSEKF ||
        parallelMode == PM_MSEKFNB ||
        parallelMode == PM_MSEKFR0 ||
        parallelMode == PM_MSEKFR0PX)
    {
        log << strpr("Multi-stream training uses %d energies/forces"
                     " per weight update.\n", numProcs);
    }

    KalmanFilter::KalmanType kalmanType = KalmanFilter::KT_STANDARD;
    if (updaterType == UT_KALMANFILTER)
    {
        kalmanType = (KalmanFilter::KalmanType)
                     atoi(settings["kalman_type"].c_str());
    }
    GradientDescent::DescentType descentType = GradientDescent::DT_FIXED;
    if (updaterType == UT_GRADIENTDESCENT)
    {
        descentType = (GradientDescent::DescentType)
                      atoi(settings["gradient_type"].c_str());
    }

    for (size_t i = 0; i < numUpdaters; ++i)
    {
        size_t sizeObservation = 0;
        if (parallelMode == PM_SERIAL)
        {
            sizeObservation = 1;
        }
        else if (parallelMode == PM_MSEKF ||
                 parallelMode == PM_MSEKFNB ||
                 parallelMode == PM_MSEKFR0 ||
                 parallelMode == PM_MSEKFR0PX)
        {
            sizeObservation = numStreams;
        }

        size_t sizeState = 0;
        if (updateStrategy == US_COMBINED)
        {
            for (size_t j = 0; j < numElements; ++j)
            {
                sizeState += elements.at(j).neuralNetwork->getNumConnections();
            }
        }
        else if (updateStrategy == US_ELEMENT)
        {
            sizeState = elements.at(i).neuralNetwork->getNumConnections();
        }

        if (updaterType == UT_GRADIENTDESCENT)
        {
            updaters.push_back((Updater*)new GradientDescent(descentType,
                                                             sizeState));
            updaters.back()->setState(&(weights.at(i).front()));
        }
        else if (updaterType == UT_KALMANFILTER)
        {
            KalmanFilter::
            KalmanParallel kalmanParallelMode = KalmanFilter::KP_SERIAL;
            if (parallelMode == PM_MSEKF)
            {
                kalmanParallelMode = KalmanFilter::KP_SERIAL;
            }
            else if (parallelMode == PM_MSEKFNB)
            {
                kalmanParallelMode = KalmanFilter::KP_NBCOMM;
            }
            else if (parallelMode == PM_MSEKFR0)
            {
                kalmanParallelMode = KalmanFilter::KP_SERIAL;
            }
            else if (parallelMode == PM_MSEKFR0PX)
            {
                kalmanParallelMode = KalmanFilter::KP_PRECALCX;
            }
            // Modes with PM_*R0* require only updater at rank 0.
            if (myRank == 0 ||
                (parallelMode != PM_MSEKFR0 && parallelMode != PM_MSEKFR0PX))
            {
                updaters.push_back((Updater*)new KalmanFilter(
                                                            kalmanType,
                                                            kalmanParallelMode,
                                                            sizeState,
                                                            sizeObservation));
                updaters.back()->setState(&(weights.at(i).front()));
            }
            if (parallelMode == PM_MSEKFNB)
            {
                dynamic_cast<KalmanFilter*>(updaters.back())->setupMPI(&comm);
            }
        }
    }
    if (updaters.size() > 0) hasUpdaters = true;
    else hasUpdaters = false;

    if (updaterType == UT_GRADIENTDESCENT)
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
    }
    else if (hasUpdaters && (updaterType == UT_KALMANFILTER))
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
            //       pow(atof(settings["kalman_nue_short"].c_str()), numStreams);
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

    log << "Calculating neighbor lists for all structures.\n";
    log << strpr("Cutoff radius for neighbor lists: %f\n",
                 maxCutoffRadius);
    for (vector<Structure>::iterator it = structures.begin();
         it != structures.end(); ++it)
    {
        it->calculateNeighborList(maxCutoffRadius);
    }

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

    // Set maximum number of energy/force updates per epoch.
    size_t maxEnergyUpdates = numEnergiesTrain;
    size_t maxForceLoop = numForcesTrain / numEnergiesTrain;
    if (parallelMode == PM_MSEKF ||
        parallelMode == PM_MSEKFNB ||
        parallelMode == PM_MSEKFR0 ||
        parallelMode == PM_MSEKFR0PX)
    {
        maxEnergyUpdates /= numProcs;
    }

    // Calculate initial RMSE and write comparison files.
    swRmse.start();
    calculateRmseEpoch();
    swRmse.stop();

    // Write initial weights to files.
    if (myRank == 0) writeWeightsEpoch();

    // Write learning curve.
    if (myRank == 0) writeLearningCurve(false);

    // Write updater status to file.
    if (myRank == 0) writeUpdaterStatus(false);

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

        // Sort update candidates if requested.
        if (selectionMode == SM_SORT) sortUpdateCandidates();

        // Try energy/force updates.
        swTrain.start();
        for (size_t i = 0; i < maxEnergyUpdates; ++i)
        {
            // Energy update.
            if (gsl_rng_uniform(rngGlobal) < epochFractionEnergies)
            {
                update(false);
                numUpdatesEnergy++;
            }
            if (useForces)
            {
                for (size_t j = 0; j < maxForceLoop; ++j)
                {
                    // Force update.
                    if (gsl_rng_uniform(rngGlobal) < epochFractionForces)
                    {
                        update(true);
                        numUpdatesForce++;
                        if (reapeatedEnergyUpdates)
                        {
                            // Repeated energy update.
                            update(false);
                            numUpdatesEnergy++;
                        }
                    }
                }
            }
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
        if (myRank == 0) writeUpdaterStatus(true);

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
    // Prepare error and derivative vector.
    vector<vector<double> > xi;
    vector<vector<double> > H;
    vector<vector<double> > h;
    // Size is depending on number of updaters.
    xi.resize(numUpdaters);
    H.resize(numUpdaters);
    if (parallelMode == PM_MSEKFNB) h.resize(numUpdaters);
    for (size_t i = 0; i < numUpdaters; ++i)
    {
        xi.at(i).resize(numStreams, 0.0);
        H.at(i).resize(numStreams * weights.at(i).size(), 0.0);
        if (parallelMode == PM_MSEKFNB)
        {
            h.at(i).resize(weights.at(i).size(), 0.0);
        }
    }

    // Starting position in weights array for combined update strategy.
    vector<size_t> start;
    if (updateStrategy == US_COMBINED)
    {
        size_t pos = 0;
        for (size_t i = 0; i < numElements; ++i)
        {
            start.push_back(pos);
            pos += elements.at(i).neuralNetwork->getNumConnections();
        }
    }

    // Determine processor which contributes update candidate.
    int proc = 0;
    if (parallelMode == PM_SERIAL)
    {
        // Choose one processor to perform the weight update.
        proc = gsl_rng_uniform_int(rngGlobal, numProcs);
    }
    else if (parallelMode == PM_MSEKF ||
             parallelMode == PM_MSEKFNB ||
             parallelMode == PM_MSEKFR0 ||
             parallelMode == PM_MSEKFR0PX)
    {
        // Uncomment this line to make the PM_SERIAL version identical to the
        // other options for a run with only 1 MPI task (same number of RNG
        // calls with rngGlobal seed.
        //proc = gsl_rng_uniform_int(rngGlobal, numProcs);
        // All processors will provide update information.
        proc = myRank;
    }

    size_t is = 0; // Local structure index.
    size_t ia = 0; // Atom index.
    size_t ic = 0; // Force component index.
    size_t il = 0; // Threshold loop counter.
    size_t indexBest = 0; // Index of best update candidate.
    double rmseFraction = 0.0;
    double rmseFractionBest = 0.0;
    MPI_Request* requestsH = NULL;
    if (proc == myRank)
    {
        size_t iuc = 0; // Chosen update candidate.
        // For SM_THRESHOLD need to loop until candidate's RMSE is above
        // threshold. Other modes don't loop here.
        size_t trials = 1;
        if (selectionMode == SM_THRESHOLD) trials = rmseThresholdTrials;
        for (il = 0; il < trials; ++il)
        {
            if (force)
            {
                // Chose a force component for the update.
                if (selectionMode == SM_RANDOM ||
                    selectionMode == SM_THRESHOLD)
                {
                    iuc = gsl_rng_uniform_int(rng,
                                              updateCandidatesForce.size());
                }
                else if (selectionMode == SM_SORT)
                {
                    // Reiterate list if there are more updates than candidates.
                    iuc = posUpdateCandidatesForce
                        % updateCandidatesForce.size();
                    posUpdateCandidatesForce++;
                }
                is = updateCandidatesForce.at(iuc).s;
                ia = updateCandidatesForce.at(iuc).a;
                ic = updateCandidatesForce.at(iuc).c;
            }
            else
            {
                // Choose a structure for the energy update.
                if (selectionMode == SM_RANDOM ||
                    selectionMode == SM_THRESHOLD)
                {
                    iuc = gsl_rng_uniform_int(rng,
                                              updateCandidatesEnergy.size());
                }
                else if (selectionMode == SM_SORT)
                {
                    iuc = posUpdateCandidatesEnergy
                        % updateCandidatesEnergy.size();
                    posUpdateCandidatesEnergy++;
                }
                is = updateCandidatesEnergy.at(iuc).s;
            }
            Structure& s = structures.at(is);
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
                    Atom const& a = s.atoms.at(ia);
                    rmseFraction = fabs(a.fRef[ic] - a.f[ic])
                                 / rmseForcesTrain;
                    // If force RMSE is above threshold stop loop immediately.
                    if (rmseFraction > rmseThresholdForce) break;
                }
                else
                {
                    calculateEnergy(s);
                    rmseFraction = fabs(s.energyRef - s.energy)
                                 / (s.numAtoms * rmseEnergiesTrain);
                    // If energy RMSE is above threshold stop loop immediately.
                    if (rmseFraction > rmseThresholdEnergy) break;
                }
                // If loop continues, free memory and remember best candidate
                // so far.
                if (freeMemory)
                {
                    s.freeAtoms(true);
                }
                if (rmseFraction > rmseFractionBest)
                {
                    rmseFractionBest = rmseFraction;
                    indexBest = iuc;
                }
            }
            if (selectionMode == SM_RANDOM || selectionMode == SM_SORT) break;
        }

        // If loop was not stopped because of a proper update candidate found
        // (RMSE above threshold) use best candidate during iteration.
        if (selectionMode == SM_THRESHOLD && il == trials)
        {
            iuc = indexBest;
            rmseFraction = rmseFractionBest;
            if (force)
            {
                is = updateCandidatesForce.at(iuc).s;
                ia = updateCandidatesForce.at(iuc).a;
                ic = updateCandidatesForce.at(iuc).c;
            }
            else
            {
                is = updateCandidatesEnergy.at(iuc).s;
            }
            // Need to calculate the symmetry functions again, maybe results
            // were not stored.
            Structure& s = structures.at(is);
#ifdef NOSFGROUPS
            calculateSymmetryFunctions(s, force);
#else
            calculateSymmetryFunctionGroups(s, force);
#endif
        }

        Structure& s = structures.at(is);
        // Temporary storage for derivative contributions of atoms (dXdc stores
        // dEdc or dFdc for energy or force update, respectively.
        vector<vector<double> > dXdc;
        dXdc.resize(numElements);
        for (size_t i = 0; i < numElements; ++i)
        {
            size_t n = elements.at(i).neuralNetwork->getNumConnections();
            dXdc.at(i).resize(n, 0.0);
        }
        // Loop over atoms and calculate atomic energy contributions.
        for (vector<Atom>::iterator it = s.atoms.begin();
             it != s.atoms.end(); ++it)
        {
            // For force update save derivative of symmetry function with
            // respect to coordinate.
            if (force) it->collectDGdxia(ia, ic);
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
            // Update derivative vector (depends on update strategy).
            // For multi-streaming calculate additional offset in H array.
            if (updateStrategy == US_COMBINED)
            {
                size_t os = myStream * weights.at(0).size() + start.at(i);
                for (size_t j = 0; j < dXdc.at(i).size(); ++j)
                {
                    H.at(0).at(os + j) += dXdc.at(i).at(j);
                }
            }
            else if (updateStrategy == US_ELEMENT)
            {
                size_t os = myStream * nn->getNumConnections();
                for (size_t j = 0; j < dXdc.at(i).size(); ++j)
                {
                    H.at(i).at(os + j) += dXdc.at(i).at(j);
                }
            }
        }
        // Apply force update weight to derivatives and copy local h vector.
        if (force || (parallelMode == PM_MSEKFNB))
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                size_t os = myStream * weights.at(i).size();
                for (size_t j = 0; j < weights.at(i).size(); ++j)
                {
                    if (force) H.at(i).at(os + j) *= forceWeight;
                    if (parallelMode == PM_MSEKFNB)
                    {
                        h.at(i).at(j) = H.at(i).at(os + j);
                    }
                }
            }
        }
        // Start communicating H matrix already here!
        if (parallelMode == PM_MSEKFNB)
        {
            requestsH = new MPI_Request[numUpdaters];
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                size_t n = weights.at(i).size();
                MPI_Iallgather(MPI_IN_PLACE, n, MPI_DOUBLE, &(H.at(i).front()), n, MPI_DOUBLE, comm, &(requestsH[i]));
            }
        }
        else if (parallelMode == PM_MSEKFR0PX)
        {
            if (myRank == 0)
            {
                // Set current location of H storage.
                vector<KalmanFilter*> kf;
                for (size_t i = 0; i < numUpdaters; ++i)
                {
                    kf.push_back(dynamic_cast<KalmanFilter*>(updaters.at(i)));
                    kf.at(i)->setDerivativeMatrix(&(H.at(i).front()));
                    kf.at(i)->calculatePartialX(0);
                }
                // Receive parts of H from individual streams.
                size_t stream;
                for (size_t p = 1; p < numStreams; ++p)
                {
                    // Check which stream calls.
                    MPI_Recv(&stream, 1, MPI_SIZE_T, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);
                    for (size_t i = 0; i < numUpdaters; ++i)
                    {
                        // Receive partial H (one column).
                        size_t n = weights.at(i).size();
                        MPI_Recv(&(H.at(i).at(stream * n)), n, MPI_DOUBLE, stream, 0, comm, MPI_STATUS_IGNORE);
                        // Calculate partial temporary result (X = P * H).
                        kf.at(i)->calculatePartialX(stream);
                    }
                }
            }
            else
            {
                requestsH = new MPI_Request[numUpdaters + 1];
                MPI_Isend(&myStream, 1, MPI_SIZE_T, 0, 0, comm, &(requestsH[numUpdaters]));
                for (size_t i = 0; i < numUpdaters; ++i)
                {
                    size_t n = weights.at(i).size();
                    MPI_Isend(&(H.at(i).at(myStream * n)), n, MPI_DOUBLE, 0, 0, comm, &(requestsH[i]));
                }
            }
        }
        // Sum up total potential energy or calculate force.
        if (force)
        {
            calculateForces(s);
            Atom const& a = s.atoms.at(ia);
            rmseFraction = fabs(a.fRef[ic] - a.f[ic]) / rmseForcesTrain;
        }
        else
        {
            calculateEnergy(s);
            rmseFraction = fabs(s.energyRef - s.energy)
                         / (s.numAtoms * rmseEnergiesTrain);
        }

        // Now symmetry function memory is not required any more for this
        // update.
        if (freeMemory) s.freeAtoms(true);

        // Compute error vector (depends on update strategy).
        if (updateStrategy == US_COMBINED)
        {
            if (force)
            {
                Atom const& a = s.atoms.at(ia);
                xi.at(0).at(myStream) =  a.fRef[ic] - a.f[ic];
            }
            else
            {
                xi.at(0).at(myStream) = s.energyRef - s.energy;
            }
        }
        else if (updateStrategy == US_ELEMENT)
        {
            for (size_t i = 0; i < numUpdaters; ++i)
            {
                if (force)
                {
                    Atom const& a = s.atoms.at(ia);
                    xi.at(i).at(myStream) = (a.fRef[ic] - a.f[ic])
                                          * a.numNeighborsPerElement.at(i)
                                          / a.numNeighbors;
                }
                else
                {
                    xi.at(i).at(myStream) = (s.energyRef - s.energy)
                                          * s.numAtomsPerElement.at(i)
                                          / s.numAtoms;
                }
            }
        }
        // Apply force update weight to error.
        if (force)
        {
            for (size_t i = 0; i < xi.size(); ++i)
            {
                for (size_t j = 0; j < xi.at(i).size(); ++j)
                {
                    xi.at(i).at(j) *= forceWeight;
                }
            }
        }
    }
    // Communicate error and derivative vector.
    MPI_Request* requestsXi = NULL;
    if (parallelMode == PM_SERIAL)
    {
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            MPI_Bcast(&(xi.at(i).front()), 1, MPI_DOUBLE, proc, comm);
            size_t n = H.at(i).size();
            MPI_Bcast(&(H.at(i).front()), n, MPI_DOUBLE, proc, comm);
        }
    }
    else if (parallelMode == PM_MSEKF)
    {
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            size_t n = weights.at(i).size();
            MPI_Allgather(MPI_IN_PLACE, n, MPI_DOUBLE, &(H.at(i).front()), n, MPI_DOUBLE, comm);
            MPI_Allgather(MPI_IN_PLACE, 1, MPI_DOUBLE, &(xi.at(i).front()), 1, MPI_DOUBLE, comm);
        }
    }
    else if (parallelMode == PM_MSEKFNB)
    {
        requestsXi = new MPI_Request[numUpdaters];
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            MPI_Iallgather(MPI_IN_PLACE, 1, MPI_DOUBLE, &(xi.at(i).front()), 1, MPI_DOUBLE, comm, &(requestsXi[i]));
        }
    }
    else if (parallelMode == PM_MSEKFR0)
    {
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            size_t n = weights.at(i).size();
            if (myRank == 0)
            {
                MPI_Gather(MPI_IN_PLACE, n, MPI_DOUBLE, &(H.at(i).front()),  n, MPI_DOUBLE, 0, comm);
                MPI_Gather(MPI_IN_PLACE, 1, MPI_DOUBLE, &(xi.at(i).front()), 1, MPI_DOUBLE, 0, comm);
            }
            else
            {
                MPI_Gather(&(H.at(i).at(myStream * n)), n, MPI_DOUBLE, NULL, n, MPI_DOUBLE, 0, comm);
                MPI_Gather(&(xi.at(i).at(myStream))   , 1, MPI_DOUBLE, NULL, 1, MPI_DOUBLE, 0, comm);
            }
        }
    }
    else if (parallelMode == PM_MSEKFR0PX)
    {
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            if (myRank == 0)
            {
                MPI_Gather(MPI_IN_PLACE, 1, MPI_DOUBLE, &(xi.at(i).front()), 1, MPI_DOUBLE, 0, comm);
            }
            else
            {
                MPI_Gather(&(xi.at(i).at(myStream)), 1, MPI_DOUBLE, NULL, 1, MPI_DOUBLE, 0, comm);
            }
        }
    }
    // Loop over all updaters.
    for (size_t i = 0; i < updaters.size(); ++i)
    {
        updaters.at(i)->setError(&(xi.at(i).front()));
        if (parallelMode != PM_MSEKFR0PX)
        {
            updaters.at(i)->setDerivativeMatrix(&(H.at(i).front()));
        }
        if (updaterType == UT_KALMANFILTER && parallelMode == PM_MSEKFNB)
        {
            KalmanFilter* kf = dynamic_cast<KalmanFilter*>(updaters.at(i));
            kf->setDerivativeVector(&(h.at(i).front()));
            kf->setRequests(&(requestsXi[i]), &(requestsH[i]));
        }
        updaters.at(i)->update();
    }
    countUpdates++;

    if (parallelMode == PM_MSEKFNB)
    {
        delete[] requestsXi;
        delete[] requestsH;
    }

    // Redistribute weights to all procs.
    if (parallelMode == PM_MSEKFR0 ||
        parallelMode == PM_MSEKFR0PX)
    {
        for (size_t i = 0; i < weights.size(); ++i)
        {
            MPI_Bcast(&(weights.at(i).front()), weights.at(i).size(), MPI_DOUBLE, 0, comm);
        }
    }

    if (parallelMode == PM_MSEKFR0PX && myRank != 0)
    {
        MPI_Wait(&(requestsH[numUpdaters]), MPI_STATUS_IGNORE);
        for (size_t i = 0; i < numUpdaters; ++i)
        {
            MPI_Wait(&(requestsH[i]), MPI_STATUS_IGNORE);
        }
        delete[] requestsH;
    }

    // Set new weights in neural networks.
    setWeights();

    // Gather update information and write it to file.
    if (writeTrainingLog)
    {
        if (parallelMode == PM_SERIAL)
        {
            size_t isg = 0;
            if (myRank != 0 && proc == myRank)
            {
                isg = structures.at(is).index;
                MPI_Send(&rmseFraction, 1, MPI_DOUBLE, 0, 0, comm);
                MPI_Send(&il          , 1, MPI_SIZE_T, 0, 0, comm);
                MPI_Send(&isg         , 1, MPI_SIZE_T, 0, 0, comm);
                MPI_Send(&is          , 1, MPI_SIZE_T, 0, 0, comm);
                if (force)
                {
                    MPI_Send(&ia, 1, MPI_SIZE_T, 0, 0, comm);
                    MPI_Send(&ic, 1, MPI_SIZE_T, 0, 0, comm);
                }
            }
            if (myRank == 0 && proc != 0)
            {
                MPI_Status ms;
                MPI_Recv(&rmseFraction, 1, MPI_DOUBLE, proc, 0, comm, &ms);
                MPI_Recv(&il          , 1, MPI_SIZE_T, proc, 0, comm, &ms);
                MPI_Recv(&isg         , 1, MPI_SIZE_T, proc, 0, comm, &ms);
                MPI_Recv(&is          , 1, MPI_SIZE_T, proc, 0, comm, &ms);
                if (force)
                {
                    MPI_Recv(&ia, 1, MPI_SIZE_T, proc, 0, comm, &ms);
                    MPI_Recv(&ic, 1, MPI_SIZE_T, proc, 0, comm, &ms);
                }
            }
            if (myRank == 0)
            {
                if (proc == myRank) isg = structures.at(is).index;
                // Collect updater input statistics.
                vector<double> absXi(numUpdaters, 0.0);
                vector<double> meanH(numUpdaters, 0.0);
                for (size_t i = 0; i < numUpdaters; ++i)
                {
                    absXi.at(i) = fabs(xi.at(i).at(0));
                    for (size_t j = 0; j < H.at(i).size(); ++j)
                    {
                        meanH.at(i) += fabs(H.at(i).at(j));
                    }
                    meanH.at(i) /= H.at(i).size();
                }
                if (force) addTrainingLogEntry(proc,
                                               il,
                                               rmseFraction,
                                               absXi,
                                               meanH,
                                               isg,
                                               is,
                                               ia,
                                               ic);
                else addTrainingLogEntry(proc,
                                         il,
                                         rmseFraction,
                                         absXi,
                                         meanH,
                                         isg,
                                         is);
            }
        }
        else if (parallelMode == PM_MSEKF ||
                 parallelMode == PM_MSEKFNB ||
                 parallelMode == PM_MSEKFR0 ||
                 parallelMode == PM_MSEKFR0PX)
        {
            size_t isg = structures.at(is).index;
            if (myRank == 0)
            {
                vector<double> vrmseFraction;
                vector<size_t> vil;
                vector<size_t> visg;
                vector<size_t> vis;
                vector<size_t> via;
                vector<size_t> vic;
                vrmseFraction.resize(numStreams, 0.0);
                vil.resize(numStreams, 0);
                visg.resize(numStreams, 0);
                vis.resize(numStreams, 0);
                MPI_Gather(&rmseFraction, 1, MPI_DOUBLE, &(vrmseFraction.front()), 1, MPI_DOUBLE, 0, comm);
                MPI_Gather(&il          , 1, MPI_SIZE_T, &(vil          .front()), 1, MPI_SIZE_T, 0, comm);
                MPI_Gather(&isg         , 1, MPI_SIZE_T, &(visg         .front()), 1, MPI_SIZE_T, 0, comm);
                MPI_Gather(&is          , 1, MPI_SIZE_T, &(vis          .front()), 1, MPI_SIZE_T, 0, comm);
                if (force)
                {
                    via.resize(numStreams, 0);
                    vic.resize(numStreams, 0);
                    MPI_Gather(&ia, 1, MPI_SIZE_T, &(via.front()), 1, MPI_SIZE_T, 0, comm);
                    MPI_Gather(&ic, 1, MPI_SIZE_T, &(vic.front()), 1, MPI_SIZE_T, 0, comm);
                }
                for (size_t i = 0; i < numStreams; ++i)
                {
                    // Collect updater input statistics.
                    vector<double> absXi(numUpdaters, 0.0);
                    vector<double> meanH(numUpdaters, 0.0);
                    for (size_t j = 0; j < numUpdaters; ++j)
                    {
                        absXi.at(j) = fabs(xi.at(j).at(i));
                        for (size_t k = 0; k < weights.at(j).size(); ++k)
                        {
                            meanH.at(j) +=
                                fabs(H.at(j).at(i * weights.at(j).size() + k));
                        }
                        meanH.at(j) /= weights.at(j).size();
                    }
                    if (force)
                    {
                        addTrainingLogEntry(i,
                                            vil.at(i),
                                            vrmseFraction.at(i),
                                            absXi,
                                            meanH,
                                            visg.at(i),
                                            vis.at(i),
                                            via.at(i),
                                            vic.at(i));
                    }
                    else
                    {
                        addTrainingLogEntry(i,
                                            vil.at(i),
                                            vrmseFraction.at(i),
                                            absXi,
                                            meanH,
                                            visg.at(i),
                                            vis.at(i));
                    }
                }
            }
            else
            {
                MPI_Gather(&rmseFraction, 1, MPI_DOUBLE, NULL, 1, MPI_DOUBLE, 0, comm);
                MPI_Gather(&il          , 1, MPI_SIZE_T, NULL, 1, MPI_SIZE_T, 0, comm);
                MPI_Gather(&isg         , 1, MPI_SIZE_T, NULL, 1, MPI_SIZE_T, 0, comm);
                MPI_Gather(&is          , 1, MPI_SIZE_T, NULL, 1, MPI_SIZE_T, 0, comm);
                if (force)
                {
                    MPI_Gather(&ia, 1, MPI_SIZE_T, NULL, 1, MPI_SIZE_T, 0, comm);
                    MPI_Gather(&ic, 1, MPI_SIZE_T, NULL, 1, MPI_SIZE_T, 0, comm);
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
                                   std::vector<double> absXi,
                                   std::vector<double> meanH,
                                   std::size_t         isg,
                                   std::size_t         is)
{
    string s = strpr("  E %5zu %10zu %5d %3zu %10.2E",
                     epoch, countUpdates, proc, il + 1, f);
    for (size_t i = 0; i < absXi.size(); ++i)
    {
        s += strpr(" %10.2E %10.2E", absXi.at(i), meanH.at(i));
    }
    s += strpr(" %10zu %5zu\n", isg, is);
    trainingLog << s;

    return;
}

// Doxygen requires namespace prefix for arguments...
void Training::addTrainingLogEntry(int                 proc,
                                   std::size_t         il,
                                   double              f,
                                   std::vector<double> absXi,
                                   std::vector<double> meanH,
                                   std::size_t         isg,
                                   std::size_t         is,
                                   std::size_t         ia,
                                   std::size_t         ic)
{
    string s = strpr("  F %5zu %10zu %5d %3zu %10.2E",
                     epoch, countUpdates, proc, il + 1, f);
    for (size_t i = 0; i < absXi.size(); ++i)
    {
        s += strpr(" %10.2E %10.2E", absXi.at(i), meanH.at(i));
    }
    s += strpr(" %10zu %5zu %5zu %2zu\n", isg, is, ia, ic);
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
