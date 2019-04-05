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

#include "Mode.h"
#include "NeuralNetwork.h"
#include "utility.h"
#include "version.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm> // std::min, std::max
#include <cstdlib>   // atoi, atof
#include <fstream>   // std::ifstream
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error

using namespace std;
using namespace nnp;

Mode::Mode() : normalize                 (false),
               checkExtrapolationWarnings(false),
               numElements               (0    ),
               maxCutoffRadius           (0.0  ),
               cutoffAlpha               (0.0  ),
               meanEnergy                (0.0  ),
               convEnergy                (1.0  ),
               convLength                (1.0  )
{
}

void Mode::initialize()
{

    log << "\n";
    log << "*****************************************"
           "**************************************\n";
    log << "\n";
    log << "   NNP LIBRARY v" NNP_VERSION "\n";
    log << "   ------------------\n";
    log << "\n";
    log << "Git branch  : " NNP_GIT_BRANCH "\n";
    log << "Git revision: " NNP_GIT_REV_SHORT " (" NNP_GIT_REV ")\n";
    log << "\n";
#ifdef _OPENMP
    log << strpr("Number of OpenMP threads: %d", omp_get_max_threads());
    log << "\n";
#endif
    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::loadSettingsFile(string const& fileName)
{
    log << "\n";
    log << "*** SETUP: SETTINGS FILE ****************"
           "**************************************\n";
    log << "\n";

    settings.loadFile(fileName);
    log << settings.info();

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupGeneric()
{
    setupNormalization();
    setupElementMap();
    setupElements();
    setupCutoff();
    setupSymmetryFunctions();
#ifndef NOSFGROUPS
    setupSymmetryFunctionGroups();
#endif
    setupNeuralNetwork();

    return;
}

void Mode::setupNormalization()
{
    log << "\n";
    log << "*** SETUP: NORMALIZATION ****************"
           "**************************************\n";
    log << "\n";

    if (settings.keywordExists("mean_energy") &&
        settings.keywordExists("conv_energy") &&
        settings.keywordExists("conv_length"))
    {
        normalize = true;
        meanEnergy = atof(settings["mean_energy"].c_str());
        convEnergy = atof(settings["conv_energy"].c_str());
        convLength = atof(settings["conv_length"].c_str());
        log << "Data set normalization is used.\n";
        log << strpr("Mean energy per atom     : %24.16E\n", meanEnergy);
        log << strpr("Conversion factor energy : %24.16E\n", convEnergy);
        log << strpr("Conversion factor length : %24.16E\n", convLength);
        if (settings.keywordExists("atom_energy"))
        {
            log << "\n";
            log << "Atomic energy offsets are used in addition to"
                   " data set normalization.\n";
            log << "Offsets will be subtracted from reference energies BEFORE"
                   " normalization is applied.\n";
        }
    }
    else if ((!settings.keywordExists("mean_energy")) &&
             (!settings.keywordExists("conv_energy")) &&
             (!settings.keywordExists("conv_length")))
    {
        normalize = false;
        log << "Data set normalization is not used.\n";
    }
    else
    {
        throw runtime_error("ERROR: Incorrect usage of normalization"
                            " keywords.\n"
                            "       Use all or none of \"mean_energy\", "
                            "\"conv_energy\" and \"conv_length\".\n");
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupElementMap()
{
    log << "\n";
    log << "*** SETUP: ELEMENT MAP ******************"
           "**************************************\n";
    log << "\n";

    elementMap.registerElements(settings["elements"]);
    log << strpr("Number of element strings found: %d\n", elementMap.size());
    for (size_t i = 0; i < elementMap.size(); ++i)
    {
        log << strpr("Element %2zu: %2s (%3zu)\n", i, elementMap[i].c_str(),
                     elementMap.atomicNumber(i));
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupElements()
{
    log << "\n";
    log << "*** SETUP: ELEMENTS *********************"
           "**************************************\n";
    log << "\n";

    numElements = (size_t)atoi(settings["number_of_elements"].c_str());
    if (numElements != elementMap.size())
    {
        throw runtime_error("ERROR: Inconsistent number of elements.\n");
    }
    log << strpr("Number of elements is consistent: %zu\n", numElements);

    for (size_t i = 0; i < numElements; ++i)
    {
        elements.push_back(Element(i, elementMap));
    }

    if (settings.keywordExists("atom_energy"))
    {
        Settings::KeyRange r = settings.getValues("atom_energy");
        for (Settings::KeyMap::const_iterator it = r.first;
             it != r.second; ++it)
        {
            vector<string> args    = split(reduce(it->second.first));
            size_t         element = elementMap[args.at(0)];
            elements.at(element).
                setAtomicEnergyOffset(atof(args.at(1).c_str()));
        }
    }
    log << "Atomic energy offsets per element:\n";
    for (size_t i = 0; i < elementMap.size(); ++i)
    {
        log << strpr("Element %2zu: %16.8E\n",
                     i, elements.at(i).getAtomicEnergyOffset());
    }

    log << "Energy offsets are automatically subtracted from reference "
           "energies.\n";
    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupCutoff()
{
    log << "\n";
    log << "*** SETUP: CUTOFF FUNCTIONS *************"
           "**************************************\n";
    log << "\n";

    vector<string> args = split(settings["cutoff_type"]);

    cutoffType = (CutoffFunction::CutoffType) atoi(args.at(0).c_str());
    if (args.size() > 1)
    {
        cutoffAlpha = atof(args.at(1).c_str());
        if (0.0 < cutoffAlpha && cutoffAlpha >= 1.0)
        {
            throw invalid_argument("ERROR: 0 <= alpha < 1.0 is required.\n");
        }
    }
    log << strpr("Parameter alpha for inner cutoff: %f\n", cutoffAlpha);
    log << "Inner cutoff = Symmetry function cutoff * alpha\n";

    log << "Equal cutoff function type for all symmetry functions:\n";
    if (cutoffType == CutoffFunction::CT_COS)
    {
        log << strpr("CutoffFunction::CT_COS (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = 1/2 * (cos(pi*x) + 1)\n";
    }
    else if (cutoffType == CutoffFunction::CT_TANHU)
    {
        log << strpr("CutoffFunction::CT_TANHU (%d)\n", cutoffType);
        log << "f(r) = tanh^3(1 - r/rc)\n";
        if (cutoffAlpha > 0.0)
        {
            log << "WARNING: Inner cutoff parameter not used in combination"
                   " with this cutoff function.\n";
        }
    }
    else if (cutoffType == CutoffFunction::CT_TANH)
    {
        log << strpr("CutoffFunction::CT_TANH (%d)\n", cutoffType);
        log << "f(r) = c * tanh^3(1 - r/rc), f(0) = 1\n";
        if (cutoffAlpha > 0.0)
        {
            log << "WARNING: Inner cutoff parameter not used in combination"
                   " with this cutoff function.\n";
        }
    }
    else if (cutoffType == CutoffFunction::CT_POLY1)
    {
        log << strpr("CutoffFunction::CT_POLY1 (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = (2x - 3)x^2 + 1\n";
    }
    else if (cutoffType == CutoffFunction::CT_POLY2)
    {
        log << strpr("CutoffFunction::CT_POLY2 (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = ((15 - 6x)x - 10)x^3 + 1\n";
    }
    else if (cutoffType == CutoffFunction::CT_POLY3)
    {
        log << strpr("CutoffFunction::CT_POLY3 (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = (x(x(20x - 70) + 84) - 35)x^4 + 1\n";
    }
    else if (cutoffType == CutoffFunction::CT_POLY4)
    {
        log << strpr("CutoffFunction::CT_POLY4 (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = (x(x((315 - 70x)x - 540) + 420) - 126)x^5 + 1\n";
    }
    else if (cutoffType == CutoffFunction::CT_EXP)
    {
        log << strpr("CutoffFunction::CT_EXP (%d)\n", cutoffType);
        log << "x := (r - rc * alpha) / (rc - rc * alpha)\n";
        log << "f(x) = exp(-1 / 1 - x^2)\n";
    }
    else if (cutoffType == CutoffFunction::CT_HARD)
    {
        log << strpr("CutoffFunction::CT_HARD (%d)\n", cutoffType);
        log << "f(r) = 1\n";
        log << "WARNING: Hard cutoff used!\n";
    }
    else
    {
        throw invalid_argument("ERROR: Unknown cutoff type.\n");
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctions()
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTIONS ***********"
           "**************************************\n";
    log << "\n";

    Settings::KeyRange r = settings.getValues("symfunction_short");
    for (Settings::KeyMap::const_iterator it = r.first; it != r.second; ++it)
    {
        vector<string> args    = split(reduce(it->second.first));
        size_t         element = elementMap[args.at(0)];

        elements.at(element).addSymmetryFunction(it->second.first,
                                                 it->second.second);
    }

    log << "Abbreviations:\n";
    log << "--------------\n";
    log << "ind .... Symmetry function index.\n";
    log << "ec ..... Central atom element.\n";
    log << "ty ..... Symmetry function type.\n";
    log << "e1 ..... Neighbor 1 element.\n";
    log << "e2 ..... Neighbor 2 element.\n";
    log << "eta .... Gaussian width eta.\n";
    log << "rs ..... Shift distance of Gaussian.\n";
    log << "la ..... Angle prefactor lambda.\n";
    log << "zeta ... Angle term exponent zeta.\n";
    log << "rc ..... Cutoff radius.\n";
    log << "ct ..... Cutoff type.\n";
    log << "ca ..... Cutoff alpha.\n";
    log << "ln ..... Line number in settings file.\n";
    log << "\n";
    maxCutoffRadius = 0.0;
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        if (normalize) it->changeLengthUnitSymmetryFunctions(convLength);
        it->sortSymmetryFunctions();
        maxCutoffRadius = max(it->getMaxCutoffRadius(), maxCutoffRadius);
        it->setCutoffFunction(cutoffType, cutoffAlpha);
        log << strpr("Short range atomic symmetry functions element %2s :\n",
                     it->getSymbol().c_str());
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << " ind ec ty e1 e2       eta        rs la "
               "zeta        rc ct   ca    ln\n";
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << it->infoSymmetryFunctionParameters();
        log << "-----------------------------------------"
               "--------------------------------------\n";
    }
    minNeighbors.resize(numElements, 0);
    minCutoffRadius.resize(numElements, maxCutoffRadius);
    for (size_t i = 0; i < numElements; ++i)
    {
        minNeighbors.at(i) = elements.at(i).getMinNeighbors();
        minCutoffRadius.at(i) = elements.at(i).getMinCutoffRadius();
        log << strpr("Minimum cutoff radius for element %2s: %f\n",
                     elements.at(i).getSymbol().c_str(),
                     minCutoffRadius.at(i) / convLength);
    }
    log << strpr("Maximum cutoff radius (global)      : %f\n",
                 maxCutoffRadius / convLength);

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctionScalingNone()
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTION SCALING ****"
           "**************************************\n";
    log << "\n";

    log << "No scaling for symmetry functions.\n";
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->setScalingNone();
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctionScaling(string const& fileName)
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTION SCALING ****"
           "**************************************\n";
    log << "\n";

    log << "Equal scaling type for all symmetry functions:\n";
    if (   ( settings.keywordExists("scale_symmetry_functions" ))
        && (!settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = SymmetryFunction::ST_SCALE;
        log << strpr("Scaling type::ST_SCALE (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmin) / (Gmax - Gmin)\n";
    }
    else if (   (!settings.keywordExists("scale_symmetry_functions" ))
             && ( settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = SymmetryFunction::ST_CENTER;
        log << strpr("Scaling type::ST_CENTER (%d)\n", scalingType);
        log << "Gs = G - Gmean\n";
    }
    else if (   ( settings.keywordExists("scale_symmetry_functions" ))
             && ( settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = SymmetryFunction::ST_SCALECENTER;
        log << strpr("Scaling type::ST_SCALECENTER (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmean) / (Gmax - Gmin)\n";
    }
    else if (settings.keywordExists("scale_symmetry_functions_sigma"))
    {
        scalingType = SymmetryFunction::ST_SCALESIGMA;
        log << strpr("Scaling type::ST_SCALESIGMA (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmean) / Gsigma\n";
    }
    else
    {
        scalingType = SymmetryFunction::ST_NONE;
        log << strpr("Scaling type::ST_NONE (%d)\n", scalingType);
        log << "Gs = G\n";
        log << "WARNING: No symmetry function scaling!\n";
    }

    double Smin = 0.0;
    double Smax = 0.0;
    if (scalingType == SymmetryFunction::ST_SCALE ||
        scalingType == SymmetryFunction::ST_SCALECENTER ||
        scalingType == SymmetryFunction::ST_SCALESIGMA)
    {
        if (settings.keywordExists("scale_min_short"))
        {
            Smin = atof(settings["scale_min_short"].c_str());
        }
        else
        {
            log << "WARNING: Keyword \"scale_min_short\" not found.\n";
            log << "         Default value for Smin = 0.0.\n";
            Smin = 0.0;
        }

        if (settings.keywordExists("scale_max_short"))
        {
            Smax = atof(settings["scale_max_short"].c_str());
        }
        else
        {
            log << "WARNING: Keyword \"scale_max_short\" not found.\n";
            log << "         Default value for Smax = 1.0.\n";
            Smax = 1.0;
        }

        log << strpr("Smin = %f\n", Smin);
        log << strpr("Smax = %f\n", Smax);
    }

    log << strpr("Symmetry function scaling statistics from file: %s\n",
                 fileName.c_str());
    log << "-----------------------------------------"
           "--------------------------------------\n";
    ifstream file;
    file.open(fileName.c_str());
    if (!file.is_open())
    {
        throw runtime_error("ERROR: Could not open file: \"" + fileName
                            + "\".\n");
    }
    string line;
    vector<string> lines;
    while (getline(file, line))
    {
        if (line.at(0) != '#') lines.push_back(line);
    }
    file.close();

    log << "\n";
    log << "Abbreviations:\n";
    log << "--------------\n";
    log << "ind ..... Symmetry function index.\n";
    log << "min ..... Minimum symmetry function value.\n";
    log << "max ..... Maximum symmetry function value.\n";
    log << "mean .... Mean symmetry function value.\n";
    log << "sigma ... Standard deviation of symmetry function values.\n";
    log << "sf ...... Scaling factor for derivatives.\n";
    log << "Smin .... Desired minimum scaled symmetry function value.\n";
    log << "Smax .... Desired maximum scaled symmetry function value.\n";
    log << "t ....... Scaling type.\n";
    log << "\n";
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->setScaling(scalingType, lines, Smin, Smax);
        log << strpr("Scaling data for symmetry functions element %2s :\n",
                     it->getSymbol().c_str());
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << " ind       min       max      mean     sigma        sf  Smin  Smax t\n";
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << it->infoSymmetryFunctionScaling();
        log << "-----------------------------------------"
               "--------------------------------------\n";
        lines.erase(lines.begin(), lines.begin() + it->numSymmetryFunctions());
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctionGroups()
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTION GROUPS *****"
           "**************************************\n";
    log << "\n";

    log << "Abbreviations:\n";
    log << "--------------\n";
    log << "ind .... Symmetry function group index.\n";
    log << "ec ..... Central atom element.\n";
    log << "ty ..... Symmetry function type.\n";
    log << "e1 ..... Neighbor 1 element.\n";
    log << "e2 ..... Neighbor 2 element.\n";
    log << "eta .... Gaussian width eta.\n";
    log << "rs ..... Shift distance of Gaussian.\n";
    log << "la ..... Angle prefactor lambda.\n";
    log << "zeta ... Angle term exponent zeta.\n";
    log << "rc ..... Cutoff radius.\n";
    log << "ct ..... Cutoff type.\n";
    log << "ca ..... Cutoff alpha.\n";
    log << "ln ..... Line number in settings file.\n";
    log << "mi ..... Member index.\n";
    log << "sfi .... Symmetry function index.\n";
    log << "e ...... Recalculate exponential term.\n";
    log << "\n";
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->setupSymmetryFunctionGroups();
        log << strpr("Short range atomic symmetry function groups "
                     "element %2s :\n", it->getSymbol().c_str());
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << " ind ec ty e1 e2       eta        rs la "
               "zeta        rc ct   ca    ln   mi  sfi e\n";
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << it->infoSymmetryFunctionGroups();
        log << "-----------------------------------------"
               "--------------------------------------\n";
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctionStatistics(bool collectStatistics,
                                           bool collectExtrapolationWarnings,
                                           bool writeExtrapolationWarnings,
                                           bool stopOnExtrapolationWarnings)
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTION STATISTICS *"
           "**************************************\n";
    log << "\n";

    log << "Equal symmetry function statistics for all elements.\n";
    log << strpr("Collect min/max/mean/sigma                        : %d\n",
                 (int)collectStatistics);
    log << strpr("Collect extrapolation warnings                    : %d\n",
                 (int)collectExtrapolationWarnings);
    log << strpr("Write extrapolation warnings immediately to stderr: %d\n",
                 (int)writeExtrapolationWarnings);
    log << strpr("Halt on any extrapolation warning                 : %d\n",
                 (int)stopOnExtrapolationWarnings);
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->statistics.collectStatistics = collectStatistics;
        it->statistics.collectExtrapolationWarnings =
                                                  collectExtrapolationWarnings;
        it->statistics.writeExtrapolationWarnings = writeExtrapolationWarnings;
        it->statistics.stopOnExtrapolationWarnings =
                                                   stopOnExtrapolationWarnings;
    }

    checkExtrapolationWarnings = collectStatistics
                              || collectExtrapolationWarnings
                              || writeExtrapolationWarnings
                              || stopOnExtrapolationWarnings;

    log << "*****************************************"
           "**************************************\n";
    return;
}

void Mode::setupNeuralNetwork()
{
    log << "\n";
    log << "*** SETUP: NEURAL NETWORKS **************"
           "**************************************\n";
    log << "\n";

    int const numLayers = 2 +
                          atoi(settings["global_hidden_layers_short"].c_str());
    int* numNeuronsPerLayer = new int[numLayers];
    NeuralNetwork::ActivationFunction* activationFunctionsPerLayer =
        new NeuralNetwork::ActivationFunction[numLayers];
    vector<string> numNeuronsPerHiddenLayer =
        split(reduce(settings["global_nodes_short"]));
    vector<string> activationFunctions =
        split(reduce(settings["global_activation_short"]));

    for (int i = 0; i < numLayers; i++)
    {
        if (i == 0)
        {
            numNeuronsPerLayer[i] = 0;
            activationFunctionsPerLayer[i] = NeuralNetwork::AF_IDENTITY;
        }
        else if (i == numLayers - 1)
        {
            numNeuronsPerLayer[i] = 1;
            activationFunctionsPerLayer[i] = NeuralNetwork::AF_IDENTITY;
        }
        else
        {
            numNeuronsPerLayer[i] =
                atoi(numNeuronsPerHiddenLayer.at(i-1).c_str());
            if (activationFunctions.at(i-1) == "l")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_IDENTITY;
            }
            else if (activationFunctions.at(i-1) == "t")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_TANH;
            }
            else if (activationFunctions.at(i-1) == "s")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_LOGISTIC;
            }
            else if (activationFunctions.at(i-1) == "p")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_SOFTPLUS;
            }
            else if (activationFunctions.at(i-1) == "r")
            {
                activationFunctionsPerLayer[i] = NeuralNetwork::AF_RELU;
            }
            else
            {
                throw runtime_error("ERROR: Unknown activation function.\n");
            }
        }

    }

    bool normalizeNeurons = settings.keywordExists("normalize_nodes");
    log << strpr("Normalize neurons (all elements): %d\n",
                 (int)normalizeNeurons);
    log << "-----------------------------------------"
           "--------------------------------------\n";

    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        numNeuronsPerLayer[0] = it->numSymmetryFunctions();
        it->neuralNetwork = new NeuralNetwork(numLayers,
                                              numNeuronsPerLayer,
                                              activationFunctionsPerLayer);
        it->neuralNetwork->setNormalizeNeurons(normalizeNeurons);
        log << strpr("Atomic short range NN for "
                     "element %2s :\n", it->getSymbol().c_str());
        log << it->neuralNetwork->info();
        log << "-----------------------------------------"
               "--------------------------------------\n";
    }

    delete[] numNeuronsPerLayer;
    delete[] activationFunctionsPerLayer;

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupNeuralNetworkWeights(string const& fileNameFormat)
{
    log << "\n";
    log << "*** SETUP: NEURAL NETWORK WEIGHTS *******"
           "**************************************\n";
    log << "\n";

    log << strpr("Weight file name format: %s\n", fileNameFormat.c_str());
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        string fileName = strpr(fileNameFormat.c_str(), it->getAtomicNumber());
        log << strpr("Weight file for element %2s: %s\n",
                     it->getSymbol().c_str(),
                     fileName.c_str());
        ifstream file;
        file.open(fileName.c_str());
        if (!file.is_open())
        {
            throw runtime_error("ERROR: Could not open file: \"" + fileName
                                + "\".\n");
        }
        string line;
        vector<double> weights;
        while (getline(file, line))
        {
            if (line.at(0) != '#')
            {
                vector<string> splitLine = split(reduce(line));
                weights.push_back(atof(splitLine.at(0).c_str()));
            }
        }
        it->neuralNetwork->setConnections(&(weights.front()));
        file.close();
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::calculateSymmetryFunctions(Structure& structure,
                                      bool const derivatives)
{
    // Skip calculation for whole structure if results are already saved.
    if (structure.hasSymmetryFunctionDerivatives) return;
    if (structure.hasSymmetryFunctions && !derivatives) return;

    Atom* a = NULL;
    Element* e = NULL;
#ifdef _OPENMP
    #pragma omp parallel for private (a, e)
#endif
    for (size_t i = 0; i < structure.atoms.size(); ++i)
    {
        // Pointer to atom.
        a = &(structure.atoms.at(i));

        // Skip calculation for individual atom if results are already saved.
        if (a->hasSymmetryFunctionDerivatives) continue;
        if (a->hasSymmetryFunctions && !derivatives) continue;

        // Get element of atom and set number of symmetry functions.
        e = &(elements.at(a->element));
        a->numSymmetryFunctions = e->numSymmetryFunctions();

#ifndef NONEIGHCHECK
        // Check if atom has low number of neighbors.
        size_t numNeighbors = a->getNumNeighbors(
                                            minCutoffRadius.at(e->getIndex()));
        if (numNeighbors < minNeighbors.at(e->getIndex()))
        {
            log << strpr("WARNING: Structure %6zu Atom %6zu : %zu "
                         "neighbors.\n",
                         a->indexStructure,
                         a->index,
                         numNeighbors);
        }
#endif

        // Allocate symmetry function data vectors in atom.
        a->allocate(derivatives);

        // Calculate symmetry functions (and derivatives).
        e->calculateSymmetryFunctions(*a, derivatives);

        // Remember that symmetry functions of this atom have been calculated.
        a->hasSymmetryFunctions = true;
        if (derivatives) a->hasSymmetryFunctionDerivatives = true;
    }

    // If requested, check extrapolation warnings or update statistics.
    // Needed to shift this out of the loop above to make it thread-safe.
    if (checkExtrapolationWarnings)
    {
        for (size_t i = 0; i < structure.atoms.size(); ++i)
        {
            a = &(structure.atoms.at(i));
            e = &(elements.at(a->element));
            e->updateSymmetryFunctionStatistics(*a);
        }
    }

    // Remember that symmetry functions of this structure have been calculated.
    structure.hasSymmetryFunctions = true;
    if (derivatives) structure.hasSymmetryFunctionDerivatives = true;

    return;
}

void Mode::calculateSymmetryFunctionGroups(Structure& structure,
                                           bool const derivatives)
{
    // Skip calculation for whole structure if results are already saved.
    if (structure.hasSymmetryFunctionDerivatives) return;
    if (structure.hasSymmetryFunctions && !derivatives) return;

    Atom* a = NULL;
    Element* e = NULL;
#ifdef _OPENMP
    #pragma omp parallel for private (a, e)
#endif
    for (size_t i = 0; i < structure.atoms.size(); ++i)
    {
        // Pointer to atom.
        a = &(structure.atoms.at(i));

        // Skip calculation for individual atom if results are already saved.
        if (a->hasSymmetryFunctionDerivatives) continue;
        if (a->hasSymmetryFunctions && !derivatives) continue;

        // Get element of atom and set number of symmetry functions.
        e = &(elements.at(a->element));
        a->numSymmetryFunctions = e->numSymmetryFunctions();

#ifndef NONEIGHCHECK
        // Check if atom has low number of neighbors.
        size_t numNeighbors = a->getNumNeighbors(
                                            minCutoffRadius.at(e->getIndex()));
        if (numNeighbors < minNeighbors.at(e->getIndex()))
        {
            log << strpr("WARNING: Structure %6zu Atom %6zu : %zu "
                         "neighbors.\n",
                         a->indexStructure,
                         a->index,
                         numNeighbors);
        }
#endif

        // Allocate symmetry function data vectors in atom.
        a->allocate(derivatives);

        // Calculate symmetry functions (and derivatives).
        e->calculateSymmetryFunctionGroups(*a, derivatives);

        // Remember that symmetry functions of this atom have been calculated.
        a->hasSymmetryFunctions = true;
        if (derivatives) a->hasSymmetryFunctionDerivatives = true;
    }

    // If requested, check extrapolation warnings or update statistics.
    // Needed to shift this out of the loop above to make it thread-safe.
    if (checkExtrapolationWarnings)
    {
        for (size_t i = 0; i < structure.atoms.size(); ++i)
        {
            a = &(structure.atoms.at(i));
            e = &(elements.at(a->element));
            e->updateSymmetryFunctionStatistics(*a);
        }
    }

    // Remember that symmetry functions of this structure have been calculated.
    structure.hasSymmetryFunctions = true;
    if (derivatives) structure.hasSymmetryFunctionDerivatives = true;

    return;
}

void Mode::calculateAtomicNeuralNetworks(Structure& structure,
                                         bool const derivatives) const
{
    for (vector<Atom>::iterator it = structure.atoms.begin();
         it != structure.atoms.end(); ++it)
    {
        Element const& e = elements.at(it->element);
        e.neuralNetwork->setInput(&((it->G).front()));
        e.neuralNetwork->propagate();
        if (derivatives) e.neuralNetwork->calculateDEdG(&((it->dEdG).front()));
        e.neuralNetwork->getOutput(&(it->energy));
    }

    return;
}

void Mode::calculateEnergy(Structure& structure) const
{
    // Loop over all atoms and add atomic contributions to total energy.
    structure.energy = 0.0;
    for (vector<Atom>::iterator it = structure.atoms.begin();
         it != structure.atoms.end(); ++it)
    {
        structure.energy += it->energy;
    }

    return;
}

void Mode::calculateForces(Structure& structure) const
{
    Atom* ai = NULL;
    // Loop over all atoms, center atom i (ai).
#ifdef _OPENMP
    #pragma omp parallel for private(ai)
#endif
    for (size_t i = 0; i < structure.atoms.size(); ++i)
    {
        // Set pointer to atom.
        ai = &(structure.atoms.at(i));

        // Reset forces.
        ai->f[0] = 0.0;
        ai->f[1] = 0.0;
        ai->f[2] = 0.0;

        // First add force contributions from atom i itself (gradient of
        // atomic energy E_i).
        for (size_t j = 0; j < ai->numSymmetryFunctions; ++j)
        {
            ai->f -= ai->dEdG.at(j) * ai->dGdr.at(j);
        }

        // Now loop over all neighbor atoms j of atom i. These may hold
        // non-zero derivatives of their symmetry functions with respect to
        // atom i's coordinates. Some atoms may appear multiple times in the
        // neighbor list because of periodic boundary conditions. To avoid
        // that the same contributions are added multiple times use the
        // "unique neighbor" list (but skip the first entry, this is always
        // atom i itself).
        for (vector<size_t>::const_iterator it =
             ai->neighborsUnique.begin() + 1;
             it != ai->neighborsUnique.end(); ++it)
        {
            // Define shortcut for atom j (aj).
            Atom& aj = structure.atoms.at(*it);

            // Loop over atom j's neighbors (n), atom i should be one of them.
            for (vector<Atom::Neighbor>::const_iterator n =
                 aj.neighbors.begin(); n != aj.neighbors.end(); ++n)
            {
                // If atom j's neighbor is atom i add force contributions.
                if (n->index == ai->index)
                {
                    for (size_t j = 0; j < aj.numSymmetryFunctions; ++j)
                    {
                        ai->f -= aj.dEdG.at(j) * n->dGdr.at(j);
                    }
                }
            }
        }
    }

    return;
}

void Mode::addEnergyOffset(Structure& structure, bool ref)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        if (ref)
        {
            structure.energyRef += structure.numAtomsPerElement.at(i)
                                 * elements.at(i).getAtomicEnergyOffset();
        }
        else
        {
            structure.energy += structure.numAtomsPerElement.at(i)
                              * elements.at(i).getAtomicEnergyOffset();
        }
    }

    return;
}

void Mode::removeEnergyOffset(Structure& structure, bool ref)
{
    for (size_t i = 0; i < numElements; ++i)
    {
        if (ref)
        {
            structure.energyRef -= structure.numAtomsPerElement.at(i)
                                 * elements.at(i).getAtomicEnergyOffset();
        }
        else
        {
            structure.energy -= structure.numAtomsPerElement.at(i)
                              * elements.at(i).getAtomicEnergyOffset();
        }
    }

    return;
}

double Mode::getEnergyOffset(Structure const& structure) const
{
    double result = 0.0;

    for (size_t i = 0; i < numElements; ++i)
    {
        result += structure.numAtomsPerElement.at(i)
                * elements.at(i).getAtomicEnergyOffset();
    }

    return result;
}

double Mode::getEnergyWithOffset(Structure const& structure, bool ref) const
{
    double result;
    if (ref) result = structure.energyRef;
    else     result = structure.energy;

    for (size_t i = 0; i < numElements; ++i)
    {
        result += structure.numAtomsPerElement.at(i)
                * elements.at(i).getAtomicEnergyOffset();
    }

    return result;
}

double Mode::normalizedEnergy(double energy) const
{
    return energy * convEnergy; 
}

double Mode::normalizedEnergy(Structure const& structure, bool ref) const
{
    if (ref)
    {
        return (structure.energyRef - structure.numAtoms * meanEnergy)
               * convEnergy; 
    }
    else
    {
        return (structure.energy - structure.numAtoms * meanEnergy)
               * convEnergy; 
    }
}

double Mode::normalizedForce(double force) const
{
    return force * convEnergy / convLength;
}

double Mode::physicalEnergy(double energy) const
{
    return energy / convEnergy; 
}

double Mode::physicalEnergy(Structure const& structure, bool ref) const
{
    if (ref)
    {
        return structure.energyRef / convEnergy + structure.numAtoms
               * meanEnergy; 
    }
    else
    {
        return structure.energy / convEnergy + structure.numAtoms * meanEnergy; 
    }
}

double Mode::physicalForce(double force) const
{
    return force * convLength / convEnergy;
}

void Mode::convertToNormalizedUnits(Structure& structure) const
{
    structure.toNormalizedUnits(meanEnergy, convEnergy, convLength);

    return;
}

void Mode::convertToPhysicalUnits(Structure& structure) const
{
    structure.toPhysicalUnits(meanEnergy, convEnergy, convLength);

    return;
}

void Mode::resetExtrapolationWarnings()
{
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        it->statistics.resetExtrapolationWarnings();
    }

    return;
}

size_t Mode::getNumExtrapolationWarnings() const
{
    size_t numExtrapolationWarnings = 0;

    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        numExtrapolationWarnings +=
            it->statistics.countExtrapolationWarnings();
    }

    return numExtrapolationWarnings;
}

vector<size_t> Mode::getNumSymmetryFunctions() const
{
    vector<size_t> v;

    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        v.push_back(it->numSymmetryFunctions());
    }

    return v;
}

bool Mode::settingsKeywordExists(std::string const& keyword) const
{
    return settings.keywordExists(keyword);
}

string Mode::settingsGetValue(std::string const& keyword) const
{
    return settings.getValue(keyword);
}


void Mode::writePrunedSettingsFile(vector<size_t> prune, string fileName) const
{
    ofstream file(fileName.c_str());
    vector<string> settingsLines = settings.getSettingsLines();
    for (size_t i = 0; i < settingsLines.size(); ++i)
    {
        if (find(prune.begin(), prune.end(), i) != prune.end())
        {
            file << "# ";
        }
        file << settingsLines.at(i) << '\n';
    }
    file.close();

    return;
}

void Mode::writeSettingsFile(ofstream* const& file) const
{
    settings.writeSettingsFile(file);

    return;
}

vector<size_t> Mode::pruneSymmetryFunctionsRange(double threshold)
{
    vector<size_t> prune;

    // Check if symmetry functions have low range.
    for (vector<Element>::const_iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        for (size_t i = 0; i < it->numSymmetryFunctions(); ++i)
        {
            SymmetryFunction const& s = it->getSymmetryFunction(i);
            if (fabs(s.getGmax() - s.getGmin()) < threshold)
            {
                prune.push_back(it->getSymmetryFunction(i).getLineNumber());
            }
        }
    }

    return prune;
}

vector<size_t> Mode::pruneSymmetryFunctionsSensitivity(
                                           double                  threshold,
                                           vector<vector<double> > sensitivity)
{
    vector<size_t> prune;

    for (size_t i = 0; i < numElements; ++i)
    {
        for (size_t j = 0; j < elements.at(i).numSymmetryFunctions(); ++j)
        {
            if (sensitivity.at(i).at(j) < threshold)
            {
                prune.push_back(
                        elements.at(i).getSymmetryFunction(j).getLineNumber());
            }
        }
    }

    return prune;
}
