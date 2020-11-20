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

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <fstream>   // std::ofstream
#include <string>    // std::string
#include <vector>    // std::vector

namespace nnp
{

/// This class implements a feed-forward neural network.
class NeuralNetwork
{
public:
    /// List of available activation function types.
    enum ActivationFunction
    {
        /// @f$f_a(x) = x@f$
        AF_IDENTITY,
        /// @f$f_a(x) = \tanh(x)@f$
        AF_TANH,
        /// @f$f_a(x) = 1 / (1 + \mathrm{e}^{-x})@f$
        AF_LOGISTIC,
        /// @f$f_a(x) = \ln (1 + \mathrm{e}^x)@f$
        AF_SOFTPLUS,
        /// @f$f_a(x) = \max(0, x)@f$ (NOT recommended for HDNNPs!)
        AF_RELU,
        /// @f$f_a(x) = \mathrm{e}^{-x^2 / 2}@f$
        AF_GAUSSIAN,
        /// @f$f_a(x) = \cos (x)@f$
        AF_COS,
        /// @f$f_a(x) = 1 - 1 / (1 + \mathrm{e}^{-x})@f$
        AF_REVLOGISTIC,
        /// @f$f_a(x) = \mathrm{e}^{-x}@f$
        AF_EXP,
        /// @f$f_a(x) = x^2@f$
        AF_HARMONIC
    };

    /// List of available connection modification schemes.
    enum ModificationScheme
    {
        /// Set all bias values to zero.
        MS_ZEROBIAS,
        /// Set all weights connecting to the output layer to zero.
        MS_ZEROOUTPUTWEIGHTS,
        /** Normalize weights via number of neuron inputs (fan-in).
         *
         * If initial weights are uniformly distributed in
         * @f$\left[-1, 1\right]@f$ they will be scaled to be in
         * @f$\left[\frac{-1}{\sqrt{n_\text{in}}},
         * \frac{1}{\sqrt{n_\text{in}}}\right]@f$, where @f$n_\text{in}@f$ is
         * the number of incoming weights of a neuron (if activation
         * function is of type #AF_TANH).
         */
        MS_FANIN,
        /** Normalize connections according to Glorot and Bengio.
         *
         * If initial weights are uniformly distributed in
         * @f$\left[-1, 1\right]@f$ they will be scaled to be in
         * @f$\left[-\sqrt{\frac{6}{n_\text{in} + n_\text{out}}},
         * \sqrt{\frac{6}{n_\text{in} + n_\text{out}}}\right]@f$, where
         * @f$n_\text{in}@f$ and @f$n_\text{out}@f$ are the number of incoming
         * and outgoing weights of a neuron, respectively (if activation
         * function is of type #AF_TANH).
         *
         * For details see:
         *  - X. Glorot and Y. Bengio, "Understanding the difficulty of
         *    training deep feedforward neural networks", International
         *    conference on artificial intelligence and statistics. 2010.
         */
        MS_GLOROTBENGIO,
        /** Initialize connections according to Nguyen-Widrow scheme.
         *
         * For details see:
         *  - D. Nguyen and B. Widrow, Improving the learning speed of 2-layer
         *    neural networks by choosing initial values of the adaptive
         *    weights, Proceedings of the International Joint Conference on
         *    Neural networks (IJCNN), pages 21-26, San Diego, 1990
         *  - T. Morawietz, Entwicklung eines effizienten Potentials für das
         *    Wasser-Dimer basierend auf künstlichen neuronalen Netzen,
         *    Master's thesis, pages 24-28, Bochum, 2010
         */
        MS_NGUYENWIDROW,
        /** Apply preconditioning to output layer connections.
         *
         * Multiply weights connecting to output neurons with @f$\sigma@f$ and
         * add "mean" to biases.
         *
         * Call #modifyConnections with two additional arguments:
         * #modifyConnections(NeuralNetwork::MS_PRECONDITIONOUTPUT,
         * mean, sigma);
         */
        MS_PRECONDITIONOUTPUT
    };

    /** Neural network class constructor.
     *
     * @param[in] numLayers Total number of layers (including in- and
     *            output layer).
     * @param[in] numNeuronsPerLayer Array with number of neurons per layer.
     * @param[in] activationFunctionsPerLayer Array with activation function
     *            type per layer (note: input layer activation function is
     *            is mandatory although it is never used).
     */
    NeuralNetwork(
                int                              numLayers,
                int const* const&                numNeuronsPerLayer,
                ActivationFunction const* const& activationFunctionsPerLayer);
    ~NeuralNetwork();
    // Prevent copying.
    //NeuralNetwork(const NeuralNetwork&) = delete;
    //NeuralNetwork(NeuralNetwork&&) = delete;
    /** Turn on/off neuron normalization.
     *
     * @param[in] normalizeNeurons true or false (default: false).
     */
    void                     setNormalizeNeurons(bool normalizeNeurons);
    /** Return total number of neurons.
     *
     *  Includes input and output layer.
     */
    int                      getNumNeurons() const;
    /** Return total number of connections.
     *
     *  Connections are all weights and biases.
     */
    int                      getNumConnections() const;
    /** Return number of weights.
     */
    int                      getNumWeights() const;
    /** Return number of biases.
     */
    int                      getNumBiases() const;
    /** Set neural network weights and biases.
     *
     * @param[in] connections One-dimensional array with neural network
     *                        connections in the following order:
     *            @f[
     *              \underbrace{
     *                \overbrace{
     *                  a^{01}_{00}, \ldots, a^{01}_{0m_1},
     *                  a^{01}_{10}, \ldots, a^{01}_{1m_1},
     *                  \ldots,
     *                  a^{01}_{n_00}, \ldots, a^{01}_{n_0m_1},
     *                }^{\text{Weights}}
     *                \overbrace{
     *                  b^{1}_{0}, \ldots, b^{1}_{m_1}
     *                }^{\text{Biases}}
     *              }_{\text{Layer } 0 \rightarrow 1},
     *              \underbrace{
     *                a^{12}_{00}, \ldots, b^{2}_{m_2}
     *              }_{\text{Layer } 1 \rightarrow 2},
     *              \ldots,
     *              \underbrace{
     *                a^{p-1,p}_{00}, \ldots, b^{p}_{m_p}
     *              }_{\text{Layer } p-1 \rightarrow p}
     *            @f]
     *            where @f$a^{i-1, i}_{jk}@f$ is the weight connecting neuron
     *            @f$j@f$ in layer @f$i-1@f$ to neuron @f$k@f$ in layer
     *            @f$i@f$ and @f$b^{i}_{k}@f$ is the bias assigned to neuron
     *            @f$k@f$ in layer @f$i@f$.
     */
    void                     setConnections(double const* const& connections);
    /** Get neural network weights and biases.
     *
     * @param[out] connections One-dimensional array with neural network
     *                         connections (same order as described in
     *                         #setConnections())
     */
    void                     getConnections(double* connections) const;
    /** Initialize connections with random numbers.
     *
     * @param[in] seed Random number generator seed.
     *
     * Weights are initialized with random values in the @f$[-1, 1]@f$
     * interval. The C standard library rand() function is used.
     */
    void                     initializeConnectionsRandomUniform(
                                                            unsigned int seed);
    /** Change connections according to a given modification scheme.
     *
     * @param[in] modificationScheme Defines how the connections are modified.
     *                               See #ModificationScheme for possible
     *                               options.
     */
    void                     modifyConnections(
                                        ModificationScheme modificationScheme);
    /** Change connections according to a given modification scheme.
     *
     * @param[in] modificationScheme Defines how the connections are modified.
     *                               See #ModificationScheme for possible
     *                               options.
     * @param[in] parameter1 Additional parameter (see #ModificationScheme).
     * @param[in] parameter2 Additional parameter (see #ModificationScheme).
     */
    void                     modifyConnections(
                                         ModificationScheme modificationScheme,
                                         double             parameter1,
                                         double             parameter2);
    /** Set neural network input layer node values.
     *
     * @param[in] input Input layer node values.
     */
    void                     setInput(double const* const& input) const;
    /** Set neural network input layer node values.
     *
     * @param[in] index Index of neuron to set.
     * @param[in] value Input layer neuron value.
     */
    void                     setInput(std::size_t const index,
                                      double const      value) const;
    /** Get neural network output layer node values.
     *
     * @param[out] output Output layer node values.
     */
    void                     getOutput(double* output) const;
    /** Propagate input information through all layers.
     *
     * With the input data set by #setInput() this will calculate all remaining
     * neuron values, the output in the last layer is acccessible via
     * #getOutput().
     */
    void                     propagate();
    /** Calculate derivative of output neuron with respect to input neurons.
     *
     * @param[out] dEdG Array containing derivative (length is number of input
     *                  neurons).
     *
     * __CAUTION__: This works only for neural networks with a single output
     * neuron!
     *
     * Returns @f$\left(\frac{dE}{dG_i}\right)_{i=1,\ldots,N}@f$, where
     * @f$E@f$ is the output neuron and @f$\left(G_i\right)_{i=1,\ldots,N}@f$
     * are the @f$N@f$ input neurons.
     */
    void                     calculateDEdG(double* dEdG) const;
    /** Calculate derivative of output neuron with respect to connections.
     *
     * @param[out] dEdc Array containing derivative (length is number of
     *                  connections, see #getNumConnections()).
     *
     * __CAUTION__: This works only for neural networks with a single output
     * neuron!
     *
     * Returns @f$\left(\frac{dE}{dc_i}\right)_{i=1,\ldots,N}@f$, where
     * @f$E@f$ is the output neuron and @f$\left(c_i\right)_{i=1,\ldots,N}@f$
     * are the @f$N@f$ connections (weights and biases) of the neural network.
     * See #setConnections() for details on the order of weights an biases.
     */
    void                     calculateDEdc(double* dEdc) const;
    /** Calculate "second" derivative of output with respect to connections.
     *
     * @param[out] dFdc Array containing derivative (length is number of
     *                  connections, see #getNumConnections()).
     * @param[in] dGdxyz Array containing derivative of input neurons with
     *                   respect to coordinates @f$\frac{\partial G_j}
     *                   {\partial x_{l, \gamma}}@f$.
     *
     * __CAUTION__: This works only for neural networks with a single output
     * neuron!
     *
     * In the context of the neural network potentials this function is used to
     * calculate derivatives of forces with respect to connections. The force
     * component @f$\gamma@f$ (where @f$\gamma@f$ is one of @f$x,y,z@f$)  of
     * particle @f$l@f$ is
     * @f[
     *   F_{l, \gamma} = - \frac{\partial}{\partial x_{l, \gamma}} \sum_i^{N}
     *   E_i = - \sum_i^N \sum_j^M \frac{\partial E_i}{\partial G_j}
     *   \frac{\partial G_j}{\partial x_{l, \gamma}},
     * @f]
     * where @f$N@f$ is the number of particles in the system and @f$M@f$ is
     * the number of symmetry functions (number of input neurons). Hence the
     * derivative of @f$F_{l, \gamma}@f$ with respect to the neural network
     * connection @f$c_n@f$ is
     * @f[
     *   \frac{\partial}{\partial c_n} F_{l, \gamma} = - \sum_i^N \sum_j^M
     *   \frac{\partial^2 E_i}{\partial c_n \partial G_j}
     *   \frac{\partial G_j}{\partial x_{l, \gamma}}.
     * @f]
     * Thus, with given @f$\frac{\partial G_j}{\partial x_{l, \gamma}}@f$ this
     * function calculates
     * @f[
     *   \sum_j^M \frac{\partial^2 E}{\partial c_n \partial G_j}
     *   \frac{\partial G_j}{\partial x_{l, \gamma}}
     * @f]
     * for the current network status and returns it via the output array.
     */
    void                     calculateDFdc(double*              dFdc,
                                           double const* const& dGdxyz) const;
    /** Write connections to file.
     *
     * @param[in,out] file File stream to write to.
     */
    void                     writeConnections(std::ofstream& file) const;
    /** Return gathered neuron statistics.
     *
     * @param[out] count Number of neuron output value calculations.
     * @param[out] min   Minimum neuron value encountered.
     * @param[out] max   Maximum neuron value encountered.
     * @param[out] sum   Sum of all neuron values encountered.
     * @param[out] sum2  Sum of squares of all neuron values encountered.
     *
     * __CAUTION__: This works only for neural networks with a single output
     * neuron!
     *
     * When neuron values are calculated (e.g. when #propagate() is called)
     * statistics about encountered values are automatically gathered. The
     * internal counters can be reset calling #resetNeuronStatistics(). Neuron
     * values are ordered layer by layer:
     * @f[
     *   \underbrace{y^0_1, \ldots, y^0_{N_0}}_\text{Input layer},
     *   \underbrace{y^1_1, \ldots, y^1_{N_1}}_{\text{Hidden Layer } 1},
     *   \ldots,
     *   \underbrace{y^p_1, \ldots, y^p_{N_p}}_\text{Output layer},
     * @f]
     * where @f$y^m_i@f$ is neuron @f$i@f$ in layer @f$m@f$ and @f$N_m@f$ is
     * the total number of neurons in this layer.
     */
    void                     getNeuronStatistics(long*   count,
                                                 double* min,
                                                 double* max,
                                                 double* sum,
                                                 double* sum2) const;
    /** Reset neuron statistics.
     *
     * Counters and summation variables for neuron statistics are reset.
     */
    void                     resetNeuronStatistics();
    //void   writeStatus(int, int);
    long                     getMemoryUsage();
    /** Print neural network architecture.
     */
    std::vector<std::string> info() const;

private:
    /// A single neuron.
    typedef struct
    {
        /// How often the value of this neuron has been evaluated.
        long    count;
        /// %Neuron value before application of activation function.
        double  x;
        /// %Neuron value.
        double  value;
        /// Derivative of activation function with respect to its argument
        /// \f$ \frac{\partial f_a}{\partial x} \f$.
        double  dfdx;
        /// Second derivative of activation function with respect to its
        /// argument \f$ \frac{\partial^2 f_a}{\partial x^2} \f$.
        double  d2fdx2;
        /// Bias value assigned to this neuron (if this is neuron \f$ y_i^n \f$
        /// this bias value is \f$ b_{i}^{n} \f$).
        double  bias;
        /// Derivative of neuron value before application of activation
        /// function with respect to input layer neurons
        /// \f$ \frac{\partial x}{\partial G} \f$.
        double  dxdG;
        /// Minimum neuron value over data set (neuron statistics).
        double  min;
        /// Maximum neuron value over data set (neuron statistics).
        double  max;
        /// Sum of neuron values over data set (neuron statistics).
        double  sum;
        /// Sum of squared neuron values over data set (neuron statistics).
        double  sum2;
        /// NN weights assigned to neuron.
        /// If this is neuron @f$i@f$ in layer @f$n@f$, \f$ y_i^n \f$,
        /// these weights are \f$ \left\{ a_{ji}^{n-1, n}
        /// \right\}_{j=1,\ldots,N_n} \f$.
        double* weights;
    } Neuron;

    /// One neural network layer.
    typedef struct
    {
        /// Number of neurons in this layer \f$ N_n \f$.
        int                numNeurons;
        /// Number of neurons in previous layer \f$ N_{n-1} \f$.
        int                numNeuronsPrevLayer;
        /// Common activation function for all neurons in this layer.
        ActivationFunction activationFunction;
        /// Array of neurons in this layer.
        Neuron*            neurons;
    } Layer;

    /// If neurons are normalized.
    bool   normalizeNeurons;
    /// Number of NN weights only.
    int    numWeights;
    /// Number of NN biases only.
    int    numBiases;
    /// Number of NN connections (weights + biases).
    int    numConnections;
    /// Total number of layers (includes input and output layers).
    int    numLayers;
    /// Number of hidden layers.
    int    numHiddenLayers;
    /// Offset adress of weights per layer in combined weights+bias array.
    int*   weightOffset;
    /// Offset adress of biases per layer in combined weights+bias array.
    int*   biasOffset;
    /// Offset adress of biases per layer in bias only array.
    int*   biasOnlyOffset;
    /// Pointer to input layer.
    Layer* inputLayer;
    /// Pointer to output layer.
    Layer* outputLayer;
    /// Neural network layers.
    Layer* layers;

    /** Calculate derivative of output neuron with respect to biases.
     *
     * @param[out] dEdb Array containing derivatives (length is number of
     *                  biases).
     *
     * __CAUTION__: This works only for neural networks with a single output
     * neuron!
     *
     * Similar to #calculateDEdc() but includes only biases. Used internally
     * by #calculateDFdc().
     */
    void   calculateDEdb(double* dEdb) const;
    /** Calculate derivative of neuron values before activation function with
     * respect to input neuron.
     *
     * @param[in] index Index of input neuron the derivative will be
     *                  calculated for.
     *
     * No output, derivatives are internally saved for each neuron in
     * Neuron::dxdG. Used internally by #calculateDFdc().
     */
    void   calculateDxdG(int index) const;
    /** Calculate second derivative of output neuron with respect to input
     * neuron and connections.
     *
     * @param[in] index Index of input neuron the derivative will be
     *                  calculated for.
     * @param[in] dEdb Derivatives of output neuron with respect to biases. See
     *                 #calculateDEdb().
     * @param[out] d2EdGdc Array containing the derivatives (ordered as
     *                     described in #setConnections()).
     *
     * __CAUTION__: This works only for neural networks with a single output
     * neuron!
     *
     * Used internally by #calculateDFdc().
     */
    void   calculateD2EdGdc(int                  index,
                            double const* const& dEdb,
                            double*              d2EdGdc) const;
    /** Allocate a single layer.
     *
     * @param[in,out] layer Neural network layer to allocate.
     * @param[in] numNeuronsPrevLayer Number of neurons in the previous layer.
     * @param[in] numNeurons Number of neurons in this layer.
     * @param[in] activationFunction Activation function to use for all neurons
     *                               in this layer.
     *
     * This function is internally called by the constructor to allocate all
     * neurons layer by layer.
     */
    void   allocateLayer(Layer&             layer,
                         int                numNeuronsPrevLayer,
                         int                numNeurons,
                         ActivationFunction activationFunction);
    /** Propagate information from one layer to the next.
     *
     * @param[in,out] layer %Neuron values in this layer will be calculated.
     * @param[in] layerPrev %Neuron values in this layer will be used as input.
     *
     * This function is internally looped by #propagate().
     */
    void   propagateLayer(Layer& layer, Layer& layerPrev);
};

}

#endif
