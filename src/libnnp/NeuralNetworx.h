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

#ifndef NEURALNETWORX_H
#define NEURALNETWORX_H

#include <cstddef> // std::size_t
#include <map>     // std::map
#include <string>  // std::string
#include <vector>  // std::vector
#include <Eigen/Core>

namespace nnp
{

/// This class implements a feed-forward neural network with Eigen.
class NeuralNetworx
{
public:

    /// List of available activation function types.
    enum class Activation
    {
        /// @f$f_a(x) = x@f$
        IDENTITY,
        /// @f$f_a(x) = \tanh(x)@f$
        TANH,
        /// @f$f_a(x) = 1 / (1 + \mathrm{e}^{-x})@f$
        LOGISTIC,
        /// @f$f_a(x) = \ln (1 + \mathrm{e}^x)@f$
        SOFTPLUS,
        /// @f$f_a(x) = \max(0, x)@f$ (NOT recommended for HDNNPs!)
        RELU,
        /// @f$f_a(x) = \mathrm{e}^{-x^2 / 2}@f$
        GAUSSIAN,
        /// @f$f_a(x) = \cos (x)@f$
        COS,
        /// @f$f_a(x) = 1 - 1 / (1 + \mathrm{e}^{-x})@f$
        REVLOGISTIC,
        /// @f$f_a(x) = \mathrm{e}^{-x}@f$
        EXP,
        /// @f$f_a(x) = x^2@f$
        HARMONIC
    };

    /// One layer of a feed-forward neural network.
    struct Layer
    {
        /** Constructor for a single layer.
         *
         * @param[in] numNeurons Number of neurons in this layer.
         * @param[in] numNeuronsPrevLayer Number of neurons in previous layer.
         * @param[in] activation Activation function used in this layer.
         */
        Layer(std::size_t numNeurons,
              std::size_t numNeuronsPrevLayer,
              Activation  activation);
        /** Propagate information through this layer.
         *
         * @param[in] yp Neuron values from previous layer.
         */
        void propagate(Eigen::VectorXd const& yp);
        /** Initialize #dydyi in this layer.
         */
        void initializeDerivInput();
        /** Propagate derivative information through this layer.
         *
         * @param[in] dypdyi Derivatives from previous layer.
         */
        void propagateDerivInput(Eigen::MatrixXd const& dypdyi);

        /// Number of neurons in this layer.
        std:: size_t    numNeurons;
        /// Number of neurons in previous layer.
        std:: size_t    numNeuronsPrevLayer;
        /// Common activation function for all neurons in this layer.
        Activation      fa;
        /// Weights from previous to this layer.
        Eigen::MatrixXd w;
        /// Biases assigned to neurons in this layer.
        Eigen::VectorXd b;
        /// Neuron weighted sum before activation function.
        Eigen::VectorXd x;
        /// Neuron values (activation function applied).
        Eigen::VectorXd y;
        /// Neuron value derivative w.r.t. activation function argument.
        Eigen::VectorXd dydx;
        /// Neuron value second derivative w.r.t. activation function argument.
        Eigen::VectorXd d2ydx2;
        /// Derivative of neurons w.r.t. neurons in input layer.
        Eigen::MatrixXd dydyi;
    };

    /** Neural network class constructor.
     *
     * @param[in] numNeuronsPerLayer Array with number of neurons per layer.
     * @param[in] activationPerLayer Array with activation function type per
     *                               layer (note: input layer activation 
     *                               function is mandatory although it is never
     *                               used).
     */
    NeuralNetworx(std::vector<size_t>     numNeuronsPerLayer,
                  std::vector<Activation> activationPerLayer);
    /** Neural network class constructor with strings for activation types.
     *
     * @param[in] numNeuronsPerLayer Array with number of neurons per layer.
     * @param[in] activationStringPerLayer Array with activation function type
     *                                     per layer (note: input layer
     *                                     activation function is mandatory
     *                                     although it is never used).
     */
    NeuralNetworx(std::vector<size_t>      numNeuronsPerLayer,
                  std::vector<std::string> activationStringPerLayer);
    /** Set input layer neuron values.
     *
     * @param[in] input Vector with input neuron values.
     */
    void                     setInput(std::vector<double> const& input);
    /** Propagate information through all layers (input already set).
     *
     * @param[in] deriv Propagate also derivative of output w.r.t. input
     *                  neurons.
     */
    void                     propagate(bool deriv = false);
    /** Propagate information through all layers.
     *
     * @param[in] input Vector with input neuron values.
     * @param[in] deriv Propagate also derivative of output w.r.t. input
     *                  neurons.
     */
    void                     propagate(std::vector<double> const& input,
                                       bool deriv = false);
    /** Get output layer neuron values.
     *
     * @param[out] output Vector with output layer neuron values.
     */
    void                     getOutput(std::vector<double>& output) const;
    /** Get derivative of output layer neurons w.r.t to input neurons.
     *
     * @param[out] derivInput Vector with output layer neuron values.
     */
    void                     getDerivInput(std::vector<std::vector<
                                               double>>& derivInput) const;
    /** Set neural network weights and biases.
     *
     * @param[in] connections One-dimensional vector with neural network
     *                        connections in the following order:
     *            @f[
     *              \underbrace{
     *                \overbrace{
     *                  a^{01}_{00}, \ldots, a^{01}_{n_00}, b^{1}_{0}
     *                }^{\text{Neuron}^{1}_{0}}
     *                \overbrace{
     *                  a^{01}_{01}, \ldots, a^{01}_{n_01}, b^{1}_{1}
     *                }^{\text{Neuron}^{1}_{1}}
     *                  \ldots,
     *                \overbrace{
     *                  a^{01}_{0n_1}, \ldots, a^{01}_{n_0n_1}, b^{1}_{n_1}
     *                }^{\text{Neuron}^{1}_{n_1}}
     *              }_{\text{Layer } 0 \rightarrow 1},
     *              \underbrace{
     *                a^{12}_{00}, \ldots, b^{2}_{n_2}
     *              }_{\text{Layer } 1 \rightarrow 2},
     *              \ldots,
     *              \underbrace{
     *                a^{p-1,p}_{00}, \ldots, b^{p}_{n_p}
     *              }_{\text{Layer } p-1 \rightarrow p}
     *            @f]
     *            where @f$a^{i-1, i}_{jk}@f$ is the weight connecting neuron
     *            @f$j@f$ in layer @f$i-1@f$ to neuron @f$k@f$ in layer
     *            @f$i@f$ and @f$b^{i}_{k}@f$ is the bias assigned to neuron
     *            @f$k@f$ in layer @f$i@f$.
     *
     *  This ordering scheme is used internally to store weights in memory.
     *  Because all weights and biases connected to a neuron are aligned in a
     *  continuous block this memory layout simplifies the implementation of
     *  node-decoupled Kalman filter training (NDEKF). However, for backward
     *  compatibility reasons this layout is NOT used for storing weights on
     *  disk (see setConnectionsAO()).
     */
    void                     setConnections(std::vector<double> const&
                                                connections);
    /** Set neural network weights and biases (alternative ordering)
     *
     * @param[in] connections One-dimensional vector with neural network
     *                        connections in the following order:
     *            @f[
     *              \underbrace{
     *                \overbrace{
     *                  a^{01}_{00}, \ldots, a^{01}_{0n_1},
     *                  a^{01}_{10}, \ldots, a^{01}_{1n_1},
     *                  \ldots,
     *                  a^{01}_{n_00}, \ldots, a^{01}_{n_0n_1},
     *                }^{\text{Weights}}
     *                \overbrace{
     *                  b^{1}_{0}, \ldots, b^{1}_{n_1}
     *                }^{\text{Biases}}
     *              }_{\text{Layer } 0 \rightarrow 1},
     *              \underbrace{
     *                a^{12}_{00}, \ldots, b^{2}_{n_2}
     *              }_{\text{Layer } 1 \rightarrow 2},
     *              \ldots,
     *              \underbrace{
     *                a^{p-1,p}_{00}, \ldots, b^{p}_{n_p}
     *              }_{\text{Layer } p-1 \rightarrow p}
     *            @f]
     *            where @f$a^{i-1, i}_{jk}@f$ is the weight connecting neuron
     *            @f$j@f$ in layer @f$i-1@f$ to neuron @f$k@f$ in layer
     *            @f$i@f$ and @f$b^{i}_{k}@f$ is the bias assigned to neuron
     *            @f$k@f$ in layer @f$i@f$.
     *
     *  This memory layout is used when storing weights on disk.
     */
    void                     setConnectionsAO(std::vector<double> const&
                                                  connections);
    /** Get neural network weights and biases.
     *
     * @param[out] connections One-dimensional vector with neural network
     *                         connections (same order as described in
     *                         #setConnections())
     */
    void                     getConnections(std::vector<double>&
                                                connections) const;
    /** Get neural network weights and biases (alternative ordering).
     *
     * @param[out] connections One-dimensional vector with neural network
     *                         connections (same order as described in
     *                         #setConnectionsAO())
     */
    void                     getConnectionsAO(std::vector<double>&
                                                  connections) const;
    /** Write connections to file.
     *
     * __CAUTION__: For compatibility reasons this format is NOT used for
     * storing NN weights to disk!
     *
     * @param[in,out] file File stream to write to.
     */
    void                     writeConnections(std::ofstream& file) const;
    /** Write connections to file (alternative ordering).
     *
     * This NN weights ordering layout is used when writing weights to disk.
     *
     * @param[in,out] file File stream to write to.
     */
    void                     writeConnectionsAO(std::ofstream& file) const;
    /** Get properties for a single neuron.
     *
     * @param[in] layer Index of layer (starting from 0 = input layer).
     * @param[in] neuron Index of neuron (starting from 0 = first neuron in
     *                   layer).
     *
     * @return Map from neuron's property (as string) to value.
     */
    std::map<
    std::string, double>     getNeuronProperties(std::size_t layer,
                                                 std::size_t neuron) const;
    /** Printable strings with neural network architecture information.
     *
     * @return Vector of lines for printing.
     */
    std::vector<std::string> info() const;
    /// Getter for #numLayers
    std::size_t              getNumLayers() const;
    /// Getter for #numConnections
    std::size_t              getNumConnections() const;
    /// Getter for #numWeights
    std::size_t              getNumWeights() const;
    /// Getter for #numBiases
    std::size_t              getNumBiases() const;

private:

    /** Class initialization.
     *
     * @param[in] numNeuronsPerLayer Array with number of neurons per layer.
     * @param[in] activationPerLayer Array with activation function type per
     *                               layer (note: input layer activation 
     *                               function is mandatory although it is never
     *                               used).
     *
     *  Avoid duplicate code due to overloaded constructor.
     */
    void initialize(std::vector<size_t>     numNeuronsPerLayer,
                    std::vector<Activation> activationPerLayer);

    /// Number of neural network layers.
    std::size_t        numLayers;
    /// Number of neural network connections (weights + biases).
    std::size_t        numConnections;
    /// Number of neural network weights.
    std::size_t        numWeights;
    /// Number of neural network biases.
    std::size_t        numBiases;
    /// Vector of neural network layers.
    std::vector<Layer> layers;

};

/** Convert string to activation function.
 *
 * @param[in] letter String representing activation function.
 *
 * @return Activation corresponding to string.
 */
NeuralNetworx::Activation activationFromString(std::string c);
/** Convert activation function to string.
 *
 * @param[in] af Input ActivationFunction type.
 *
 * @return String representing activation function.
 */
std::string stringFromActivation(NeuralNetworx::Activation a);

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline
std::size_t NeuralNetworx::getNumLayers() const {return numLayers;}

inline
std::size_t NeuralNetworx::getNumConnections() const {return numConnections;}

inline
std::size_t NeuralNetworx::getNumWeights() const {return numWeights;}

inline
std::size_t NeuralNetworx::getNumBiases() const {return numBiases;}

}

#endif
