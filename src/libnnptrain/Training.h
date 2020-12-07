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

#ifndef TRAINING_H
#define TRAINING_H

#include "Atom.h"
#include "Dataset.h"
#include "Stopwatch.h"
#include "Updater.h"
#include <cstddef> // std::size_t
#include <fstream> // std::ofstream
#include <map>     // std::map
#include <random>  // std::mt19937_64
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/// Training methods.
class Training : public Dataset
{
public:
    /// Type of update routine.
    enum UpdaterType
    {
        /// Simple gradient descent methods.
        UT_GD,
        /// Kalman filter-based methods.
        UT_KF,
        /// Levenberg-Marquardt algorithm.
        UT_LM
    };

    /** Training parallelization mode.
     *
     * This mode determines if and how individual MPI tasks contribute to
     * parallel training. Note that in all cases the data set gets distributed
     * among the MPI processes and RMSE computation is always parallelized.
     */
    enum ParallelMode
    {
        /** No training parallelization, only data set distribution.
         *
         * Data set is distributed via MPI, but for each weight update only
         * a single task is active, selects energy/force update candidates,
         * computes errors and gradients, and updates weights.
         */
        //PM_DATASET,
        /** Parallel gradient computation, update on rank 0.
         *
         * Data set is distributed via MPI, each tasks selects energy/force
         * update candidates, and computes errors and gradients, which are
         * collected on rank 0. Weight update is carried out on rank 0 and new
         * weights are redistributed to all tasks.
         */
        PM_TRAIN_RK0,
        /** Parallel gradient computation, update on each task.
         *
         * Data set is distributed via MPI, each tasks selects energy/force
         * update candidates, and computes errors and gradients, which are
         * collected on all MPI tasks. Identical weight updates are carried out
         * on each task. This mode is ideal if the update routine itself is
         * parallelized.
         */
        PM_TRAIN_ALL
    };

    /// Jacobian matrix preparation mode.
    enum JacobianMode
    {
        /// No Jacobian, sum up contributions from update candidates.
        JM_SUM,
        /// Prepare one Jacobian entry for each task, sum up within tasks.
        JM_TASK,
        /// Prepare full Jacobian matrix.
        JM_FULL
    };

    /// Update strategies available for Training.
    enum UpdateStrategy
    {
        /// One combined updater for all elements.
        US_COMBINED,
        /// Separate updaters for individual elements.
        US_ELEMENT
    };

    /// How update candidates are selected during Training.
    enum SelectionMode
    {
        /// Select candidates randomly.
        SM_RANDOM,
        /// Sort candidates according to their RMSE and pick worst first.
        SM_SORT,
        /// Select candidates randomly with RMSE above threshold.
        SM_THRESHOLD
    };

    /// Constructor.
    Training();
    /// Destructor, updater vector needs to be cleaned.
    ~Training();
    /** Randomly select training and test set structures.
     *
     * Also fills training candidates lists.
     */
    void                  selectSets();
    /** Write training and test set to separate files (train.data and
     * test.data, same format as input.data).
     */
    void                  writeSetsToFiles();
    /** Initialize weights for all elements.
     */
    void                  initializeWeights();
    /** Initialize weights vector according to update strategy.
     *
     * @param[in] updateStrategy Determines the shape of the weights array.
     */
    void                  initializeWeightsMemory(UpdateStrategy updateStrategy
                                                      = US_COMBINED);
    /** Set training stage (if multiple stages are needed for NNP type).
     *
     * @param[in] stage Training stage to set.
     */
    void                  setStage(std::size_t stage);
    /** General training settings and setup of weight update routine.
     */
    void                  setupTraining();
    /** Calculate neighbor lists for all structures.
     */
    void                  calculateNeighborLists();
    /** Calculate error metrics for all structures.
     *
     * @param[in] fileNames Map of properties to file names for training/test
     *                      comparison files.
     *
     * If fileNames map is empty, no files will be written.
     */
    void                  calculateError(
                                   std::map<std::string,
                                   std::pair<
                                   std::string, std::string>> const fileNames);
    /** Calculate error metrics per epoch for all structures with file names
     * used in training loop.
     *
     * Also write training curve to file.
     */
    void                  calculateErrorEpoch();
    /** Print training loop header on screen.
     */
    void                  printHeader();
    /** Print preferred error metric and timing information on screen.
     */
    void                  printEpoch();
    /** Write weights to files (one file for each element).
     *
     * @param[in] nnName Identifier for neural network.
     * @param[in] fileNameFormat String with file name format.
     */
    void                  writeWeights(
                                      std::string const& nnName,
                                      std::string const& fileNameFormat) const;
    /** Write weights to files during training loop.
     */
    void                  writeWeightsEpoch() const;
    /** Write current RMSEs and epoch information to file.
     *
     * @param[in] append If true, append to file, otherwise create new file.
     * @param[in] fileName File name for learning curve file.
     */
    void                  writeLearningCurve(bool              append,
                                             std::string const fileName
                                                 = "learning-curve.out") const;
    /** Write neuron statistics collected since last invocation.
     *
     * @param[in] nnName Identifier of neural network to process.
     * @param[in] fileName File name for statistics file.
     */
    void                  writeNeuronStatistics(
                                            std::string const& nnName,
                                            std::string const& fileName) const;
    /** Write neuron statistics during training loop.
     */
    void                  writeNeuronStatisticsEpoch() const;
    /** Reset neuron statistics for all elements.
     */
    void                  resetNeuronStatistics();
    /** Write updater information to file.
     *
     * @param[in] append If true, append to file, otherwise create new file.
     * @param[in] fileNameFormat String with file name format.
     */
    void                  writeUpdaterStatus(bool              append,
                                             std::string const fileNameFormat
                                                 = "updater.%03zu.out") const;
    /** Sort update candidates with descending RMSE.
     *
     * @param[in] property Training property.
     */
    void                  sortUpdateCandidates(std::string const& property);
    /** Shuffle update candidates.
     *
     * @param[in] property Training property.
     */
    void                  shuffleUpdateCandidates(std::string const& property);
    /** Check if selection mode should be changed.
     */
    void                  checkSelectionMode();
    /** Execute main training loop.
     */
    void                  loop();
    /** Select energies/forces schedule for one epoch.
     */
    void                  setEpochSchedule();
    /** Perform one update.
     *
     * @param[in] property Training property to use for update.
     */
    void                  update(std::string const& property);
    /** Get a single weight value.
     *
     * @param[in] element Element index of weight.
     * @param[in] index Weight index.
     *
     * @return Weight value.
     *
     * Note: This function is implemented for testing purposes and works
     * correctly only with update strategy #US_ELEMENT.
     */
    double                getSingleWeight(std::size_t element,
                                          std::size_t index);
    /** Set a single weight value.
     *
     * @param[in] element Element index of weight.
     * @param[in] index Weight index.
     * @param[in] value Weight value.
     *
     * Note: This function is implemented for testing purposes and works
     * correctly only with update strategy #US_ELEMENT.
     */
    void                  setSingleWeight(std::size_t element,
                                          std::size_t index,
                                          double      value);
    /** Calculate derivatives of energy with respect to weights.
     *
     * @param[in,out] structure Structure to process.
     *
     * @return Vector with derivatives of energy with respect to weights (per
     *         element).
     *
     * @note This function is implemented for testing purposes.
     */
    std::vector<
    std::vector<double> > calculateWeightDerivatives(Structure* structure);
    /** Calculate derivatives of force with respect to weights.
     *
     * @param[in,out] structure Structure to process.
     * @param[in] atom Atom index.
     * @param[in] component x, y or z-component of force (0, 1, 2).
     *
     * @return Vector with derivatives of force with respect to weights (per
     *         element).
     *
     * @note This function is implemented for testing purposes.
     */
    std::vector<
    std::vector<double> > calculateWeightDerivatives(Structure*  structure,
                                                     std::size_t atom,
                                                     std::size_t component);
    /** Set training log file name
     *
     * @param[in] fileName File name for training log.
     */
    void                  setTrainingLogFileName(std::string fileName);

private:
    /// Contains location of one update candidate (energy or force).
    struct UpdateCandidate
    {
        /// Structure index.
        std::size_t s;
        /// Atom index (only used for force candidates).
        std::size_t a;
        /// Component index (x,y,z -> 0,1,2, only used for force candidates).
        std::size_t c;
        /// Absolute value of error with respect to reference value.
        double      error;

        /// Overload < operator to sort in \em descending order.
        bool operator<(UpdateCandidate const& rhs) const {
            return this->error > rhs.error;
        }
    };

    /// Specific training quantity (e.g. energies, forces, charges).
    struct Property
    {
        /// Constructor.
        Property(std::string const& property);

        /// Copy of identifier within Property map.
        std::string                  property;
        /// Error metric for display.
        std::string                  displayMetric;
        /// Tiny abbreviation string for property.
        std::string                  tiny;
        /// Plural string of property;
        std::string                  plural;
        /// Selection mode for update candidates.
        SelectionMode                selectionMode;
        /// Number of training patterns in set.
        std::size_t                  numTrainPatterns;
        /// Number of training patterns in set.
        std::size_t                  numTestPatterns;
        /// Batch size for each MPI task.
        std::size_t                  taskBatchSize;
        /// Write comparison every this many epochs.
        std::size_t                  writeCompEvery;
        /// Up to this epoch comparison is written every epoch.
        std::size_t                  writeCompAlways;
        /// Current position in update candidate list (SM_SORT).
        std::size_t                  posUpdateCandidates;
        /// Maximum trials for SM_THRESHOLD selection mode.
        std::size_t                  rmseThresholdTrials;
        /// Counter for updates per epoch.
        std::size_t                  countUpdates;
        /// Number of desired updates per epoch.
        std::size_t                  numUpdates;
        /// Patterns used per update.
        std::size_t                  patternsPerUpdate;
        /// Patterns used per update (summed over all MPI tasks).
        std::size_t                  patternsPerUpdateGlobal;
        /// Global number of errors per update.
        std::size_t                  numErrorsGlobal;
        /// Desired update fraction per epoch.
        double                       epochFraction;
        /// RMSE threshold for update candidates.
        double                       rmseThreshold;
        /// Errors per task for each update.
        std::vector<int>             errorsPerTask;
        /// Offset for combined error per task.
        std::vector<int>             offsetPerTask;
        /// Error metrics available for this property.
        std::vector<std::string>     errorMetrics;
        /// Current error metrics of training patterns.
        std::map<
        std::string, double>         errorTrain;
        /// Current error metrics of test patterns.
        std::map<
        std::string, double>         errorTest;
        /// Vector with indices of training patterns.
        std::vector<UpdateCandidate> updateCandidates;
        /// Weights per task per updater.
        std::vector<
        std::vector<int>>            weightsPerTask;
        /// Stride for Jacobians per task per updater.
        std::vector<
        std::vector<int>>            offsetJacobian;
        /// Global error vector (per updater).
        std::vector<
        std::vector<double>>         error;
        /// Global Jacobian (per updater).
        std::vector<
        std::vector<double>>         jacobian;
        /// Schedule for varying selection mode.
        std::map<
        std::size_t, SelectionMode>  selectionModeSchedule;
    };

    /// Map of all training properties.
    struct PropertyMap : std::map<std::string, Property>
    {
        /// Overload [] operator to simplify access.
        Property& operator[](std::string const& key) {return this->at(key);}
        /// Overload [] operator to simplify access (const version).
        Property const& operator[](std::string const& key) const
        {
            return this->at(key);
        }
        /// Check if property is present.
        bool exists(std::string const& key)
        {
            return (this->find(key) != this->end());
        }
    };

    /// Updater type used.
    UpdaterType              updaterType;
    /// Parallelization mode used.
    ParallelMode             parallelMode;
    /// Jacobian mode used.
    JacobianMode             jacobianMode;
    /// Update strategy used.
    UpdateStrategy           updateStrategy;
    /// If this rank performs weight updates.
    bool                     hasUpdaters;
    /// If this rank holds structure information.
    bool                     hasStructures;
    /// Use forces for training.
    bool                     useForces;
    /// After force update perform energy update for corresponding structure.
    bool                     repeatedEnergyUpdates;
    /// Free symmetry function memory after calculation.
    bool                     freeMemory;
    /// Whether training log file is written.
    bool                     writeTrainingLog;
    /// Training stage.
    std::size_t              stage;
    /// Number of updaters (depends on update strategy).
    std::size_t              numUpdaters;
    /// Number of epochs requested.
    std::size_t              numEpochs;
    /// Current epoch.
    std::size_t              epoch;
    /// Write weights every this many epochs.
    std::size_t              writeWeightsEvery;
    /// Up to this epoch weights are written every epoch.
    std::size_t              writeWeightsAlways;
    /// Write neuron statistics every this many epochs.
    std::size_t              writeNeuronStatisticsEvery;
    /// Up to this epoch neuron statistics are written every epoch.
    std::size_t              writeNeuronStatisticsAlways;
    /// Update counter (for all training quantities together).
    std::size_t              countUpdates;
    /// Total number of weights.
    std::size_t              numWeights;
    /// Force update weight.
    double                   forceWeight;
    /// File name for training log.
    std::string              trainingLogFileName;
    /// ID of neural network the training is working on.
    std::string              nnId;
    /// Training log file.
    std::ofstream            trainingLog;
    /// Update schedule epoch (false = energy update, true = force update).
    std::vector<int>         epochSchedule;
    /// Number of weights per updater.
    std::vector<std::size_t> numWeightsPerUpdater;
    /// Offset of each element's weights in combined array.
    std::vector<std::size_t> weightsOffset;
    /// Vector of actually used training properties.
    std::vector<std::string> pk;
#ifndef NNP_FULL_SFD_MEMORY
    /// Derivative of symmetry functions with respect to one specific atom
    /// coordinate.
    std::vector<double>      dGdxia;
#endif
    /// Neural network weights and biases for each element.
    std::vector<
    std::vector<double> >    weights;
    /// Weight updater (combined or for each element).
    std::vector<Updater*>    updaters;
    /// Stopwatches for timing overview.
    std::map<
    std::string, Stopwatch>  sw;
    /// Per-task random number generator.
    std::mt19937_64          rngNew;
    /// Global random number generator.
    std::mt19937_64          rngGlobalNew;
    /// Actual training properties.
    PropertyMap              p;


    /** Check if training loop should be continued.
     *
     * @return True if further training should be performed, false otherwise.
     */
    bool advance() const;
    /** Get weights from neural network class.
     */
    void getWeights();
    /** Set weights in neural network class.
     */
    void setWeights();
    /** Write energy update data to training log file.
     *
     * @param[in] proc Processor which provided update candidate.
     * @param[in] il Loop index of threshold loop.
     * @param[in] f RMSE fraction of update candidate.
     * @param[in] is Local structure index.
     * @param[in] isg Global structure index.
     */
    void addTrainingLogEntry(int         proc,
                             std::size_t il,
                             double      f,
                             std::size_t isg,
                             std::size_t is);
    /** Write force update data to training log file.
     *
     * @param[in] proc Processor which provided update candidate.
     * @param[in] il Loop index of threshold loop.
     * @param[in] f RMSE fraction of update candidate.
     * @param[in] is Local structure index.
     * @param[in] isg Global structure index.
     * @param[in] ia Atom index.
     * @param[in] ic Component index.
     */
    void addTrainingLogEntry(int         proc,
                             std::size_t il,
                             double      f,
                             std::size_t isg,
                             std::size_t is,
                             std::size_t ia,
                             std::size_t ic);
    /** Write charge update data to training log file.
     *
     * @param[in] proc Processor which provided update candidate.
     * @param[in] il Loop index of threshold loop.
     * @param[in] f RMSE fraction of update candidate.
     * @param[in] is Local structure index.
     * @param[in] isg Global structure index.
     * @param[in] ia Atom index.
     */
    void addTrainingLogEntry(int         proc,
                             std::size_t il,
                             double      f,
                             std::size_t isg,
                             std::size_t is,
                             std::size_t ia);
#ifndef NNP_FULL_SFD_MEMORY
    /** Collect derivative of symmetry functions with repect to one atom's
     * coordinate.
     *
     * @param[in] atom The atom which owns the symmetry functions.
     * @param[in] indexAtom The index @f$i@f$ of the atom requested.
     * @param[in] indexComponent The component @f$\alpha@f$ of the atom
     *                           requested.
     *
     * This calculates an array of derivatives
     * @f[
     *   \left(\frac{\partial G_1}{\partial x_{i,\alpha}}, \ldots,
     *   \frac{\partial G_n}{\partial x_{i,\alpha}}\right),
     *
     * @f]
     * where @f$\{G_j\}_{j=1,\ldots,n}@f$ are the symmetry functions for this
     * atom and @f$x_{i,\alpha}@f$ is the @f$\alpha@f$-component of the
     * position of atom @f$i@f$. The result is stored in #dGdxia.
     */
    void collectDGdxia(Atom const& atom,
                       std::size_t indexAtom,
                       std::size_t indexComponent);
#endif
    /** Randomly initialize specificy neural network weights.
     *
     * @param[in] type Actual network type to initialize ("short" or "charge").
     */
    void randomizeNeuralNetworkWeights(std::string const& type);
    /** Set selection mode for specific training property.
     *
     * @param[in] property Training property (uses corresponding keyword).
     */
    void setupSelectionMode(std::string const& property);
    /** Set file output intervals for properties and other quantities.
     *
     * @param[in] type Training property or `weights` or `neuron-stats`.
     */
    void setupFileOutput(std::string const& type);
    /** Set up how often properties are updated.
     *
     * @param[in] property Training property (uses corresponding keyword).
     */
    void setupUpdatePlan(std::string const& property);
    /** Allocate error and Jacobian arrays for given property.
     *
     * @param[in] property Training property.
     */
    void allocateArrays(std::string const& property);
    /** Write timing data for all clocks.
     *
     * @param[in] append If true, append to file, otherwise create new file.
     * @param[in] fileName File name for timing data file.
     */
    void writeTimingData(bool              append,
                         std::string const fileName = "timing.out");
};

}

#endif
