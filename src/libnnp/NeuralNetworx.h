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
    /** Printable strings with neural network architecture information.
     *
     * @return Vector of lines for printing.
     */
    std::vector<std::string> info() const;

private:

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

}

#endif
