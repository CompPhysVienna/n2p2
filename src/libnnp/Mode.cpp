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
#include <algorithm> // std::min, std::max, std::remove_if
#include <cstdlib>   // atoi, atof
#include <fstream>   // std::ifstream
#include <iostream>
#include <map>       // std::multimap
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <utility>   // std::piecewise_construct, std::forward_as_tuple

using namespace std;
using namespace nnp;
using namespace Eigen;

Mode::Mode() : nnpType                   (NNPType::HDNNP_2G),
               normalize                 (false            ),
               checkExtrapolationWarnings(false            ),
               numElements               (0                ),
               maxCutoffRadius           (0.0              ),
               cutoffAlpha               (0.0              ),
               meanEnergy                (0.0              ),
               convEnergy                (1.0              ),
               convLength                (1.0              ),
               convCharge                (1.0              )
{
}

void Mode::initialize()
{

    log << "\n";
    log << "*****************************************"
           "**************************************\n";
    log << "\n";
    log << "   NNP LIBRARY " NNP_GIT_VERSION "\n";
    log << "   ------------------\n";
    log << "\n";
    log << "Git branch  : " NNP_GIT_BRANCH "\n";
    log << "Git revision: " NNP_GIT_REV "\n";
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

    size_t numCriticalProblems = settings.loadFile(fileName);
    log << settings.info();
    if (numCriticalProblems > 0)
    {
        throw runtime_error(strpr("ERROR: %zu critical problem(s) were found "
                                  "in settings file.\n", numCriticalProblems));
    }

    if (settings.keywordExists("nnp_type"))
    {
        string nnpTypeString = settings["nnp_type"];
        if      (nnpTypeString == "2G-HDNNP") nnpType = NNPType::HDNNP_2G;
        else if (nnpTypeString == "4G-HDNNP") nnpType = NNPType::HDNNP_4G;
        else if (nnpTypeString == "Q-HDNNP")  nnpType = NNPType::HDNNP_Q;
        else nnpType = (NNPType)atoi(settings["nnp_type"].c_str());
    }

    if (nnpType == NNPType::HDNNP_2G)
    {
        log << "This settings file defines a short-range NNP (2G-HDNNP).\n";
    }
    else if (nnpType == NNPType::HDNNP_4G)
    {
        log << "This settings file defines a NNP with electrostatics and\n"
               "non-local charge transfer (4G-HDNNP).\n";
    }
    else if (nnpType == NNPType::HDNNP_Q)
    {
        log << "This settings file defines a short-range NNP similar to\n"
               "4G-HDNNP with additional charge NN but with neither\n"
               "electrostatics nor global charge equilibration\n"
               "(method by M. Bircher).\n";
    }
    else
    {
        throw runtime_error("ERROR: Unknown NNP type.\n");
    }

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
#ifndef NNP_FULL_SFD_MEMORY
    setupSymmetryFunctionMemory(false);
#endif
#ifndef NNP_NO_SF_CACHE
    setupSymmetryFunctionCache();
#endif
#ifndef NNP_NO_SF_GROUPS
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
        settings.keywordExists("conv_length") &&
        settings.keywordExists("conv_charge"))
    {
        normalize = true;
        meanEnergy = atof(settings["mean_energy"].c_str());
        convEnergy = atof(settings["conv_energy"].c_str());
        convLength = atof(settings["conv_length"].c_str());
        convCharge = atof(settings["conv_charge"].c_str());
        log << "Data set normalization is used.\n";
        log << strpr("Mean energy per atom     : %24.16E\n", meanEnergy);
        log << strpr("Conversion factor energy : %24.16E\n", convEnergy);
        log << strpr("Conversion factor length : %24.16E\n", convLength);
        log << strpr("Conversion factor charge : %24.16E\n", convCharge);
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
             (!settings.keywordExists("conv_length")) &&
             (!settings.keywordExists("conv_charge")))
    {
        normalize = false;
        log << "Data set normalization is not used.\n";
    }
    else
    {
        throw runtime_error("ERROR: Incorrect usage of normalization"
                            " keywords.\n"
                            "       Use all or none of \"mean_energy\", "
                            "\"conv_energy\", \"conv_charge\" and \"conv_length\".\n");
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

    if (nnpType == NNPType::HDNNP_4G)
    {
        Settings::KeyRange r = settings.getValues("fixed_gausswidth");
        for (Settings::KeyMap::const_iterator it = r.first;
             it != r.second; ++it)
        {
            vector<string> args    = split(reduce(it->second.first));
            size_t         element = elementMap[args.at(0)];
            elements.at(element).setQsigma(atof(args.at(1).c_str()));
        }
        log << "Gaussian width of charge distribution per element:\n";
        for (size_t i = 0; i < elementMap.size(); ++i)
        {
            log << strpr("Element %2zu: %16.8E\n",
                         i, elements.at(i).getQsigma());
        }
    }

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
    log << "tp ..... Symmetry function type.\n";
    log << "sbtp ... Symmetry function subtype (e.g. cutoff type).\n";
    log << "e1 ..... Neighbor 1 element.\n";
    log << "e2 ..... Neighbor 2 element.\n";
    log << "eta .... Gaussian width eta.\n";
    log << "rs/rl... Shift distance of Gaussian or left cutoff radius "
           "for polynomial.\n";
    log << "angl.... Left cutoff angle for polynomial.\n";
    log << "angr.... Right cutoff angle for polynomial.\n";
    log << "la ..... Angle prefactor lambda.\n";
    log << "zeta ... Angle term exponent zeta.\n";
    log << "rc ..... Cutoff radius / right cutoff radius for polynomial.\n";
    log << "a ...... Free parameter alpha (e.g. cutoff alpha).\n";
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
        log << "--------------------------------------------------"
               "-----------------------------------------------\n";
        log << " ind ec tp sbtp e1 e2       eta      rs/rl         "
               "rc   angl   angr la zeta    a    ln\n";
        log << "--------------------------------------------------"
               "-----------------------------------------------\n";
        log << it->infoSymmetryFunctionParameters();
        log << "--------------------------------------------------"
               "-----------------------------------------------\n";
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
        scalingType = SymFnc::ST_SCALE;
        log << strpr("Scaling type::ST_SCALE (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmin) / (Gmax - Gmin)\n";
    }
    else if (   (!settings.keywordExists("scale_symmetry_functions" ))
             && ( settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = SymFnc::ST_CENTER;
        log << strpr("Scaling type::ST_CENTER (%d)\n", scalingType);
        log << "Gs = G - Gmean\n";
    }
    else if (   ( settings.keywordExists("scale_symmetry_functions" ))
             && ( settings.keywordExists("center_symmetry_functions")))
    {
        scalingType = SymFnc::ST_SCALECENTER;
        log << strpr("Scaling type::ST_SCALECENTER (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmean) / (Gmax - Gmin)\n";
    }
    else if (settings.keywordExists("scale_symmetry_functions_sigma"))
    {
        scalingType = SymFnc::ST_SCALESIGMA;
        log << strpr("Scaling type::ST_SCALESIGMA (%d)\n", scalingType);
        log << "Gs = Smin + (Smax - Smin) * (G - Gmean) / Gsigma\n";
    }
    else
    {
        scalingType = SymFnc::ST_NONE;
        log << strpr("Scaling type::ST_NONE (%d)\n", scalingType);
        log << "Gs = G\n";
        log << "WARNING: No symmetry function scaling!\n";
    }

    double Smin = 0.0;
    double Smax = 0.0;
    if (scalingType == SymFnc::ST_SCALE ||
        scalingType == SymFnc::ST_SCALECENTER ||
        scalingType == SymFnc::ST_SCALESIGMA)
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
    log << "ind .... Symmetry function index.\n";
    log << "ec ..... Central atom element.\n";
    log << "tp ..... Symmetry function type.\n";
    log << "sbtp ... Symmetry function subtype (e.g. cutoff type).\n";
    log << "e1 ..... Neighbor 1 element.\n";
    log << "e2 ..... Neighbor 2 element.\n";
    log << "eta .... Gaussian width eta.\n";
    log << "rs/rl... Shift distance of Gaussian or left cutoff radius "
           "for polynomial.\n";
    log << "angl.... Left cutoff angle for polynomial.\n";
    log << "angr.... Right cutoff angle for polynomial.\n";
    log << "la ..... Angle prefactor lambda.\n";
    log << "zeta ... Angle term exponent zeta.\n";
    log << "rc ..... Cutoff radius / right cutoff radius for polynomial.\n";
    log << "a ...... Free parameter alpha (e.g. cutoff alpha).\n";
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
        log << "------------------------------------------------------"
               "----------------------------------------------------\n";
        log << " ind ec tp sbtp e1 e2       eta      rs/rl         "
               "rc   angl   angr la zeta    a    ln   mi  sfi e\n";
        log << "------------------------------------------------------"
               "----------------------------------------------------\n";
        log << it->infoSymmetryFunctionGroups();
        log << "------------------------------------------------------"
               "----------------------------------------------------\n";
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupSymmetryFunctionMemory(bool verbose)
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTION MEMORY *****"
           "**************************************\n";
    log << "\n";

    for (auto& e : elements)
    {
        e.setupSymmetryFunctionMemory();
        vector<
        size_t> symmetryFunctionNumTable = e.getSymmetryFunctionNumTable();
        vector<
        vector<size_t>> symmetryFunctionTable = e.getSymmetryFunctionTable();
        log << strpr("Symmetry function derivatives memory table "
                     "for element %2s :\n", e.getSymbol().c_str());
        log << "-----------------------------------------"
               "--------------------------------------\n";
        log << "Relevant symmetry functions for neighbors with element:\n";
        for (size_t i = 0; i < numElements; ++i)
        {
            log << strpr("- %2s: %4zu of %4zu (%5.1f %)\n",
                         elementMap[i].c_str(),
                         symmetryFunctionNumTable.at(i),
                         e.numSymmetryFunctions(),
                         (100.0 * symmetryFunctionNumTable.at(i))
                         / e.numSymmetryFunctions());
            if (verbose)
            {
                log << "-----------------------------------------"
                       "--------------------------------------\n";
                for (auto isf : symmetryFunctionTable.at(i))
                {
                    SymFnc const& sf = e.getSymmetryFunction(isf);
                    log << sf.parameterLine();
                }
                log << "-----------------------------------------"
                       "--------------------------------------\n";
            }
        }
        log << "-----------------------------------------"
               "--------------------------------------\n";
    }
    if (verbose)
    {
        for (auto& e : elements)
        {
            log << strpr("%2s - symmetry function per-element index table:\n",
                         e.getSymbol().c_str());
            log << "-----------------------------------------"
                   "--------------------------------------\n";
            log << " ind";
            for (size_t i = 0; i < numElements; ++i)
            {
                log << strpr(" %4s", elementMap[i].c_str());
            }
            log << "\n";
            log << "-----------------------------------------"
                   "--------------------------------------\n";
            for (size_t i = 0; i < e.numSymmetryFunctions(); ++i)
            {
                SymFnc const& sf = e.getSymmetryFunction(i);
                log << strpr("%4zu", sf.getIndex() + 1);
                vector<size_t> indexPerElement = sf.getIndexPerElement();
                for (auto ipe : sf.getIndexPerElement())
                {
                    if (ipe == numeric_limits<size_t>::max())
                    {
                        log << strpr("     ");
                    }
                    else
                    {
                        log << strpr(" %4zu", ipe + 1);
                    }
                }
                log << "\n";
            }
            log << "-----------------------------------------"
                   "--------------------------------------\n";
        }

    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

#ifndef NNP_NO_SF_CACHE
void Mode::setupSymmetryFunctionCache(bool verbose)
{
    log << "\n";
    log << "*** SETUP: SYMMETRY FUNCTION CACHE ******"
           "**************************************\n";
    log << "\n";

    for (size_t i = 0; i < numElements; ++i)
    {
        using SFCacheList = Element::SFCacheList;
        vector<vector<SFCacheList>> cacheLists(numElements);
        Element& e = elements.at(i);
        for (size_t j = 0; j < e.numSymmetryFunctions(); ++j)
        {
            SymFnc const& s = e.getSymmetryFunction(j);
            for (auto identifier : s.getCacheIdentifiers())
            {
                size_t ne = atoi(split(identifier)[0].c_str());
                bool unknown = true;
                for (auto& c : cacheLists.at(ne))
                {
                    if (identifier == c.identifier)
                    {
                        c.indices.push_back(s.getIndex());
                        unknown = false;
                        break;
                    }
                }
                if (unknown)
                {
                    cacheLists.at(ne).push_back(SFCacheList());
                    cacheLists.at(ne).back().element = ne;
                    cacheLists.at(ne).back().identifier = identifier;
                    cacheLists.at(ne).back().indices.push_back(s.getIndex());
                }
            }
        }
        if (verbose)
        {
            log << strpr("Multiple cache identifiers for element %2s:\n\n",
                         e.getSymbol().c_str());
        }
        double cacheUsageMean = 0.0;
        size_t cacheCount = 0;
        for (size_t j = 0; j < numElements; ++j)
        {
            if (verbose)
            {
                log << strpr("Neighbor %2s:\n", elementMap[j].c_str());
            }
            vector<SFCacheList>& c = cacheLists.at(j);
            c.erase(remove_if(c.begin(),
                              c.end(),
                              [](SFCacheList l)
                              {
                                  return l.indices.size() <= 1;
                              }), c.end());
            cacheCount += c.size();
            for (size_t k = 0; k < c.size(); ++k)
            {
                cacheUsageMean += c.at(k).indices.size();
                if (verbose)
                {
                    log << strpr("Cache %zu, Identifier \"%s\", "
                                 "Symmetry functions",
                                 k, c.at(k).identifier.c_str());
                    for (auto si : c.at(k).indices)
                    {
                        log << strpr(" %zu", si);
                    }
                    log << "\n";
                }
            }
        }
        e.setCacheIndices(cacheLists);
        //for (size_t j = 0; j < e.numSymmetryFunctions(); ++j)
        //{
        //    SymFnc const& sf = e.getSymmetryFunction(j);
        //    auto indices = sf.getCacheIndices();
        //    size_t count = 0;
        //    for (size_t k = 0; k < numElements; ++k)
        //    {
        //        count += indices.at(k).size();
        //    }
        //    if (count > 0)
        //    {
        //        log << strpr("SF %4zu:\n", sf.getIndex());
        //    }
        //    for (size_t k = 0; k < numElements; ++k)
        //    {
        //        if (indices.at(k).size() > 0)
        //        {
        //            log << strpr("- Neighbor %2s:", elementMap[k].c_str());
        //            for (size_t l = 0; l < indices.at(k).size(); ++l)
        //            {
        //                log << strpr(" %zu", indices.at(k).at(l));
        //            }
        //            log << "\n";
        //        }
        //    }
        //}
        cacheUsageMean /= cacheCount;
        log << strpr("Element %2s: in total %zu caches, "
                     "used %3.2f times on average.\n",
                     e.getSymbol().c_str(), cacheCount, cacheUsageMean);
        if (verbose)
        {
            log << "-----------------------------------------"
                   "--------------------------------------\n";
        }
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}
#endif

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

    string id;

    // Some NNP types require extra NNs.
    if (nnpType == NNPType::HDNNP_4G)
    {
        id = "elec";
        nnk.push_back(id);
        nns[id].id = id;
        nns.at(id).name = "electronegativity";
        nns.at(id).weightFileFormat = "weightse.%03zu.data";
        nns.at(id).keywordSuffix = "_electrostatic";
        nns.at(id).keywordSuffix2 = "_charge";
    }
    else if(nnpType == NNPType::HDNNP_Q)
    {
        id = "elec";
        nnk.push_back(id);
        nns[id].id = id;
        nns.at(id).name = "charge";
        nns.at(id).weightFileFormat = "weightse.%03zu.data";
        nns.at(id).keywordSuffix = "_electrostatic";
        nns.at(id).keywordSuffix2 = "_charge";
    }

    // All NNP types contain a short range NN.
    id = "short";
    nnk.push_back(id);
    nns[id].id = id;
    nns.at(id).name = "short range";
    nns.at(id).weightFileFormat = "weights.%03zu.data";
    nns.at(id).keywordSuffix = "_short";
    nns.at(id).keywordSuffix2 = "_short";

    // Loop over all NNs and set global properties.
    for (auto& k : nnk)
    {
        // Each elements number of hidden layers.
        size_t globalNumHiddenLayers = 0;
        // Abbreviation for current NN.
        NNSetup& nn = nns.at(k);
        // Set size of NN topology vector.
        nn.topology.resize(numElements);
        // First, check for global number of hidden layers.
        string keyword = "global_hidden_layers" + nn.keywordSuffix;
        if (settings.keywordExists(keyword))
        {
            globalNumHiddenLayers = atoi(settings[keyword].c_str());
            for (auto& t : nn.topology)
            {
                t.numLayers = globalNumHiddenLayers + 2;
            }
        }
        // Now, check for per-element number of hidden layers.
        keyword = "element_hidden_layers" + nn.keywordSuffix;
        if (settings.keywordExists(keyword))
        {
            Settings::KeyRange r = settings.getValues(keyword);
            for (Settings::KeyMap::const_iterator it = r.first;
                 it != r.second; ++it)
            {
                vector<string> args = split(reduce(it->second.first));
                size_t const e = elementMap[args.at(0)];
                size_t const n = atoi(args.at(1).c_str());
                nn.topology.at(e).numLayers = n + 2;
            }
        }
        // Check whether user has set all NN's number of layers correctly.
        for (auto& t : nn.topology)
        {
            if (t.numLayers == 0)
            {
                throw runtime_error("ERROR: Number of neural network hidden "
                                    "layers unset for some elements.\n");
            }
        }
        // Finally, allocate NN topologies data.
        for (auto& t : nn.topology)
        {
            t.numNeuronsPerLayer.resize(t.numLayers, 0);
            t.activationFunctionsPerLayer.resize(t.numLayers,
                                                 NeuralNetwork::AF_UNSET);
        }

        // Now read global number of neurons and activation functions.
        vector<string> globalNumNeuronsPerHiddenLayer;
        keyword = "global_nodes" + nn.keywordSuffix;
        if (settings.keywordExists(keyword))
        {
            globalNumNeuronsPerHiddenLayer = split(reduce(settings[keyword]));
            if (globalNumHiddenLayers != globalNumNeuronsPerHiddenLayer.size())
            {
                throw runtime_error(strpr("ERROR: Inconsistent global NN "
                                          "topology keyword \"%s\".\n",
                                          keyword.c_str()));
            }
        }
        vector<string> globalActivationFunctions;
        keyword = "global_activation" + nn.keywordSuffix;
        if (settings.keywordExists(keyword))
        {
            globalActivationFunctions = split(reduce(settings[keyword]));
            if (globalNumHiddenLayers != globalActivationFunctions.size() - 1)
            {
                throw runtime_error(strpr("ERROR: Inconsistent global NN "
                                          "topology keyword \"%s\".\n",
                                          keyword.c_str()));
            }
        }
        // Set global number of neurons and activation functions if provided.
        bool globalNumNeurons = (globalNumNeuronsPerHiddenLayer.size() != 0);
        bool globalActivation = (globalActivationFunctions.size() != 0);
        for (size_t i = 0; i < numElements; ++i)
        {
            NNSetup::Topology& t = nn.topology.at(i);
            size_t const nsf = elements.at(i).numSymmetryFunctions();
            // Set input layer. Number of input layer neurons depends on NNP
            // type and NN purpose.
            if (nnpType == NNPType::HDNNP_2G)
            {
                // Can assume NN id is "short".
                t.numNeuronsPerLayer.at(0) = nsf;
            }
            else if (nnpType == NNPType::HDNNP_4G ||
                     nnpType == NNPType::HDNNP_Q)
            {
                // NN with id "elec" requires only SFs.
                if (k == "elec") t.numNeuronsPerLayer.at(0) = nsf;
                // "short" NN needs extra charge neuron.
                else if (k == "short") t.numNeuronsPerLayer.at(0) = nsf + 1;
            }
            // Set dummy input neuron activation function.
            t.activationFunctionsPerLayer.at(0) = NeuralNetwork::AF_IDENTITY;
            // Set output layer. Assume single output neuron.
            t.numNeuronsPerLayer.at(t.numLayers - 1) = 1;
            // If this element's NN does not use the global number of hidden
            // layers it makes no sense to set the global number of hidden
            // neurons or activation functions. Hence, skip the settings here,
            // appropriate settings should follow later in the per-element
            // section.
            if ((size_t)t.numLayers != globalNumHiddenLayers + 2) continue;
            for (int j = 1; j < t.numLayers; ++j)
            {
                if ((j == t.numLayers - 1) && globalActivation)
                {
                    t.activationFunctionsPerLayer.at(j) = activationFromString(
                        globalActivationFunctions.at(t.numLayers - 2));
                }
                else
                {
                    if (globalNumNeurons)
                    {
                        t.numNeuronsPerLayer.at(j) = atoi(
                            globalNumNeuronsPerHiddenLayer.at(j - 1).c_str());
                    }
                    if (globalActivation)
                    {
                        t.activationFunctionsPerLayer.at(j) =
                            activationFromString(
                                globalActivationFunctions.at(j - 1));
                    }
                }
            }
        }
        // Override global number of neurons with per-element keyword.
        keyword = "element_nodes" + nn.keywordSuffix;
        if (settings.keywordExists(keyword))
        {
            Settings::KeyRange r = settings.getValues(keyword);
            for (Settings::KeyMap::const_iterator it = r.first;
                 it != r.second; ++it)
            {
                vector<string> args = split(reduce(it->second.first));
                size_t e = elementMap[args.at(0)];
                size_t n = args.size() - 1;
                NNSetup::Topology& t = nn.topology.at(e);
                if ((size_t)t.numLayers != n + 2)
                {
                    throw runtime_error(strpr("ERROR: Inconsistent per-element"
                                              " NN topology keyword \"%s\".\n",
                                              keyword.c_str()));
                }
                for (int j = 1; j < t.numLayers - 2; ++j)
                {
                    t.numNeuronsPerLayer.at(j) = atoi(args.at(j).c_str());
                }
            }
        }
        // Override global activation functions with per-element keyword.
        keyword = "element_activation" + nn.keywordSuffix;
        if (settings.keywordExists(keyword))
        {
            Settings::KeyRange r = settings.getValues(keyword);
            for (Settings::KeyMap::const_iterator it = r.first;
                 it != r.second; ++it)
            {
                vector<string> args = split(reduce(it->second.first));
                size_t e = elementMap[args.at(0)];
                size_t n = args.size() - 1;
                NNSetup::Topology& t = nn.topology.at(e);
                if ((size_t)t.numLayers != n + 1)
                {
                    throw runtime_error(strpr("ERROR: Inconsistent per-element"
                                              " NN topology keyword \"%s\".\n",
                                              keyword.c_str()));
                }
                for (int j = 1; j < t.numLayers - 1; ++j)
                {
                    t.activationFunctionsPerLayer.at(j) =
                        activationFromString(args.at(j).c_str());
                }
            }
        }

        // Finally check everything for any unset NN property.
        for (size_t i = 0; i < numElements; ++i)
        {
            NNSetup::Topology const& t = nn.topology.at(i);
            for (int j = 0; j < t.numLayers; ++j)
            {
                if (t.numNeuronsPerLayer.at(j) == 0)
                {
                    throw runtime_error(strpr(
                              "ERROR: NN \"%s\", element %2s: number of "
                              "neurons for layer %d unset.\n",
                              nn.id.c_str(),
                              elements.at(i).getSymbol().c_str(),
                              j));
                }
                if (t.activationFunctionsPerLayer.at(j)
                        == NeuralNetwork::AF_UNSET)
                {
                    throw runtime_error(strpr(
                              "ERROR: NN \"%s\", element %2s: activation "
                              "functions for layer %d unset.\n",
                              nn.id.c_str(),
                              elements.at(i).getSymbol().c_str(),
                              j));
                }
            }
        }
    }


    bool normalizeNeurons = settings.keywordExists("normalize_nodes");
    log << strpr("Normalize neurons (all elements): %d\n",
                 (int)normalizeNeurons);
    log << "-----------------------------------------"
           "--------------------------------------\n";

    // Finally, allocate all neural networks.
    for (auto& k : nnk)
    {
        for (size_t i = 0; i < numElements; ++i)
        {
            Element& e = elements.at(i);
            NNSetup::Topology const& t = nns.at(k).topology.at(i);
            e.neuralNetworks.emplace(
                                    piecewise_construct,
                                    forward_as_tuple(k),
                                    forward_as_tuple(
                                        t.numLayers,
                                        t.numNeuronsPerLayer.data(),
                                        t.activationFunctionsPerLayer.data()));
            e.neuralNetworks.at(k).setNormalizeNeurons(normalizeNeurons);
            log << strpr("Atomic %s NN for "
                         "element %2s :\n",
                         nns.at(k).name.c_str(),
                         e.getSymbol().c_str());
            log << e.neuralNetworks.at(k).info();
            log << "-----------------------------------------"
                   "--------------------------------------\n";
        }
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupNeuralNetworkWeights(map<string, string> fileNameFormats)
{
    setupNeuralNetworkWeights("", fileNameFormats);
    return;
}

void Mode::setupNeuralNetworkWeights(string              directoryPrefix,
                                     map<string, string> fileNameFormats)
{
    log << "\n";
    log << "*** SETUP: NEURAL NETWORK WEIGHTS *******"
           "**************************************\n";
    log << "\n";

    for (auto k : nnk)
    {
        string actualFileNameFormat;
        if (fileNameFormats.find(k) != fileNameFormats.end())
        {
            actualFileNameFormat = fileNameFormats.at(k);
        }
        else actualFileNameFormat = nns.at(k).weightFileFormat;
        actualFileNameFormat = directoryPrefix + actualFileNameFormat;
        log << strpr("%s weight file name format: %s\n",
                     cap(nns.at(k).name).c_str(),
                     actualFileNameFormat.c_str());
        readNeuralNetworkWeights(k, actualFileNameFormat);
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}

void Mode::setupAtomicHardness(string fileNameFormat)
{
    log << "\n";
    log << "*** SETUP: ATOMIC HARDNESS **************"
           "**************************************\n";
    log << "\n";

    log << strpr("Atomic hardness file name format: %s\n",
                 fileNameFormat.c_str());
    for (size_t i = 0; i < numElements; ++i)
    {
        string fileName = strpr(fileNameFormat.c_str(),
                                elements.at(i).getAtomicNumber());
        log << strpr("Atomic hardness for element %2s from file %s: ",
                     elements.at(i).getSymbol().c_str(),
                     fileName.c_str());
        vector<double> const data = readColumnsFromFile(fileName, {0}).at(0);
        if (data.size() != 1)
        {
            throw runtime_error("ERROR: Atomic hardness data is "
                                "inconsistent.\n");
        }
        elements.at(i).setHardness(data.at(0));
        log << strpr("%16.8E\n", elements.at(i).getHardness());
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

        // Inform atom if extra charge neuron is present in short-range NN.
        if (nnpType == NNPType::HDNNP_4G ||
            nnpType == NNPType::HDNNP_Q) a->useChargeNeuron = true;

        // Get element of atom and set number of symmetry functions.
        e = &(elements.at(a->element));
        a->numSymmetryFunctions = e->numSymmetryFunctions();
        if (derivatives)
        {
            a->numSymmetryFunctionDerivatives
                = e->getSymmetryFunctionNumTable();
        }
#ifndef NNP_NO_SF_CACHE
        a->cacheSizePerElement = e->getCacheSizes();
#endif

#ifndef NNP_NO_NEIGH_CHECK
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

        // Inform atom if extra charge neuron is present in short-range NN.
        if (nnpType == NNPType::HDNNP_4G ||
            nnpType == NNPType::HDNNP_Q) a->useChargeNeuron = true;

        // Get element of atom and set number of symmetry functions.
        e = &(elements.at(a->element));
        a->numSymmetryFunctions = e->numSymmetryFunctions();
        if (derivatives)
        {
            a->numSymmetryFunctionDerivatives
                = e->getSymmetryFunctionNumTable();
        }
#ifndef NNP_NO_SF_CACHE
        a->cacheSizePerElement = e->getCacheSizes();
#endif

#ifndef NNP_NO_NEIGH_CHECK
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
                                         bool const derivatives,
                                         string id)
{
    if (id == "") id = nnk.front();

    if (nnpType == NNPType::HDNNP_2G)
    {
        for (vector<Atom>::iterator it = structure.atoms.begin();
             it != structure.atoms.end(); ++it)
        {
            NeuralNetwork& nn = elements.at(it->element)
                                .neuralNetworks.at(id);
            nn.setInput(&((it->G).front()));
            nn.propagate();
            if (derivatives)
            {
                nn.calculateDEdG(&((it->dEdG).front()));
            }
            nn.getOutput(&(it->energy));
        }
    }
    else if (nnpType == NNPType::HDNNP_4G)
    {
        if (id == "elec")
        {
            for (auto& a : structure.atoms)
            {
                NeuralNetwork& nn = elements.at(a.element)
                                    .neuralNetworks.at(id);
                nn.setInput(&((a.G).front()));
                nn.propagate();
                nn.getOutput(&(a.chi));
                log << strpr("Atom %5zu (%2s) chi: %16.8E\n",
                             a.index, elementMap[a.element].c_str(), a.chi);
            }
        }
        else if (id == "short")
        {
            for (auto& a : structure.atoms)
            {
                NeuralNetwork& nn = elements.at(a.element)
                                    .neuralNetworks.at(id);
                nn.setInput(&((a.G).front()));
                // TODO: This part should simplify with improved NN class.
                for (size_t i = 0; i < a.G.size(); ++i)
                {
                    nn.setInput(i, a.G.at(i));
                }
                // Set additional charge neuron.
                nn.setInput(a.G.size(), a.charge);
                nn.propagate();
                nn.getOutput(&(a.energy));
                log << strpr("Atom %5zu (%2s) energy: %16.8E\n",
                             a.index, elementMap[a.element].c_str(), a.energy);
            }
        }
    }
    else if (nnpType == NNPType::HDNNP_Q)
    {
        // Ignore ID, both NNs are computed here.
        for (vector<Atom>::iterator it = structure.atoms.begin();
             it != structure.atoms.end(); ++it)
        {
            // First the charge NN.
            NeuralNetwork& nnCharge = elements.at(it->element)
                                      .neuralNetworks.at("elec");
            nnCharge.setInput(&((it->G).front()));
            nnCharge.propagate();
            if (derivatives)
            {
                nnCharge.calculateDEdG(&((it->dQdG).front()));
            }
            nnCharge.getOutput(&(it->charge));

            // Now the short-range NN (have to set input neurons individually).
            NeuralNetwork& nnShort = elements.at(it->element)
                                     .neuralNetworks.at("short");
            // TODO: This part should simplify with improved NN class.
            for (size_t i = 0; i < it->G.size(); ++i)
            {
                nnShort.setInput(i, it->G.at(i));
            }
            // Set additional charge neuron.
            nnShort.setInput(it->G.size(), it->charge);
            nnShort.propagate();
            if (derivatives)
            {
                nnShort.calculateDEdG(&((it->dEdG).front()));
            }
            nnShort.getOutput(&(it->energy));
        }
    }

    return;
}

// TODO: Make this const?
void Mode::chargeEquilibration(Structure& structure)
{
    Structure& s = structure;

    // Prepare hardness vector and precalculate gamma(i, j).
    VectorXd hardness(numElements);
    MatrixXd siggam(numElements, numElements);
    for (size_t i = 0; i < numElements; ++i)
    {
        hardness(i) = elements.at(i).getHardness();
        double const iSigma = elements.at(i).getQsigma();
        for (size_t j = 0; j < numElements; ++j)
        {
            double const jSigma = elements.at(j).getQsigma();
            if (i == j) siggam(i, j) = sqrt(M_PI) * iSigma;
            else        siggam(i, j) = sqrt(2.0 * (iSigma * iSigma
                                                + jSigma * jSigma));
        }
    }

    double const error = s.calculateElectrostaticEnergy(hardness, siggam);

    cout << "A: " << endl;
    cout << s.A << endl;
    log << strpr("Solve relative error: %16.8E\n", error);

    for (auto const& a : structure.atoms)
    {
        log << strpr("Atom %5zu (%2s) q: %16.8E\n",
                     a.index, elementMap[a.element].c_str(), a.charge);
        structure.charge += a.charge;
    }
    log << strpr("Total charge: %16.8E (ref: %16.8E)\n",
                 structure.charge, structure.chargeRef);

    log << strpr("Electrostatic energy: %16.8E\n", structure.energyElec);

    return;
}

void Mode::calculateEnergy(Structure& structure) const
{
    // Loop over all atoms and add atomic contributions to total energy.
    structure.energy = 0.0;
    structure.energyShort = 0.0;
    for (vector<Atom>::iterator it = structure.atoms.begin();
         it != structure.atoms.end(); ++it)
    {
        structure.energyShort += it->energy;
    }
    structure.energy = structure.energyShort + structure.energyElec;

    //cout << strpr("Electrostatic energy: %24.16E\n", structure.energyElec);
    //cout << strpr("Short-range   energy: %24.16E\n", structure.energyShort);
    //cout << strpr("Sum           energy: %24.16E\n", structure.energy);
    //cout << strpr("Offset        energy: %24.16E\n", getEnergyOffset(structure));
    //cout << "---------------------\n";
    //cout << strpr("Total         energy: %24.16E\n", structure.energy + getEnergyOffset(structure));
    //cout << strpr("Reference     energy: %24.16E\n", structure.energyRef + getEnergyOffset(structure));
    //cout << "---------------------\n";
    //cout << "without offset:      \n";
    //cout << strpr("Total         energy: %24.16E\n", structure.energy);
    //cout << strpr("Reference     energy: %24.16E\n", structure.energyRef);

    return;
}

void Mode::calculateCharge(Structure& structure) const
{
    // Loop over all atoms and add atomic charge contributions to total charge.
    structure.charge = 0.0;
    for (vector<Atom>::iterator it = structure.atoms.begin();
         it != structure.atoms.end(); ++it)
    {
        structure.charge += it->charge;
    }

    cout << "---------------------\n";
    cout << strpr("Total         charge: %24.16E\n", structure.charge);
    cout << strpr("Reference     charge: %24.16E\n", structure.chargeRef);

    //throw runtime_error("ERROR: Here ends code for 4G-HDNNPs\n");

    return;
}

void Mode::calculateForces(Structure& structure) const
{
    if (nnpType == NNPType::HDNNP_2G)
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
#ifndef NNP_FULL_SFD_MEMORY
               vector<vector<size_t> > const& tableFull
                   = elements.at(aj.element).getSymmetryFunctionTable();
#endif
               // Loop over atom j's neighbors (n), atom i should be one of them.
               for (vector<Atom::Neighbor>::const_iterator n =
                    aj.neighbors.begin(); n != aj.neighbors.end(); ++n)
               {
                   // If atom j's neighbor is atom i add force contributions.
                   if (n->index == ai->index)
                   {
#ifndef NNP_FULL_SFD_MEMORY
                       vector<size_t> const& table = tableFull.at(n->element);
                       for (size_t j = 0; j < n->dGdr.size(); ++j)
                       {
                           ai->f -= aj.dEdG.at(table.at(j)) * n->dGdr.at(j);
                       }
#else
                       for (size_t j = 0; j < aj.numSymmetryFunctions; ++j)
                       {
                           ai->f -= aj.dEdG.at(j) * n->dGdr.at(j);
                       }
#endif
                   }
               }
           }
       }
       return;
    }
    else if (nnpType == NNPType::HDNNP_Q)
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

           size_t idQ = ai->numSymmetryFunctions;

           // Reset forces.
           ai->f[0] = 0.0;
           ai->f[1] = 0.0;
           ai->f[2] = 0.0;

           // First add force contributions from atom i itself (gradient of
           // atomic energy E_i).
           // Formally:   \sum_G dE/dG*dG/dR + dE/dQ dQ/dR
           //           = \sum_G dE/dG*dG/dR + \sum_G dE/dQ dQ/dG dG/dR
           //           = \sum_G (dE/dG + dE/dQ * dQ/dG) * dG/dR

           for (size_t j = 0; j < idQ; ++j)
           {
               ai->f -= (ai->dEdG.at(j) + ai->dEdG.at(idQ) * ai->dQdG.at(j)) * ai->dGdr.at(j);
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
#ifndef NNP_FULL_SFD_MEMORY
               vector<vector<size_t> > const& tableFull
                   = elements.at(aj.element).getSymmetryFunctionTable();
#endif
               // Loop over atom j's neighbors (n), atom i should be one of them.
               for (vector<Atom::Neighbor>::const_iterator n =
                    aj.neighbors.begin(); n != aj.neighbors.end(); ++n)
               {
                   // If atom j's neighbor is atom i add force contributions.
                   if (n->index == ai->index)
                   {
#ifndef NNP_FULL_SFD_MEMORY
                       idQ  = aj.numSymmetryFunctions;
                       vector<size_t> const& table = tableFull.at(n->element);
                       for (size_t j = 0; j < n->dGdr.size(); ++j)
                       {
                           ai->f -= ( aj.dEdG.at(table.at(j)) + aj.dEdG.at(idQ) * aj.dQdG.at(table.at(j)) ) * n->dGdr.at(j);
                       }
#else
                       idQ  = aj.numSymmetryFunctions - 1;
                       for (size_t j = 0; j < idQ; ++j)
                       {
                           ai->f -=  ( aj.dEdG.at(j) + aj.dEdG.at(idQ) * aj.dQdG.at(j) ) * n->dGdr.at(j);
                       }
#endif
                   }
               }
           }
       }
       return;
    }
    else
    {
        cout << "WARNING: Forces are not yet implemented.\n";
        return;
    }
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

double Mode::normalized(string const& property, double value) const
{
    if      (property == "energy") return value * convEnergy;
    else if (property == "force")  return value * convEnergy / convLength;
    else if (property == "charge") return value * convCharge;
    else throw runtime_error("ERROR: Unknown property to convert to "
                             "normalized units.\n");
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

double Mode::physical(string const& property, double value) const
{
    if      (property == "energy") return value / convEnergy;
    else if (property == "force")  return value * convLength / convEnergy;
    else if (property == "charge") return value / convCharge;
    else throw runtime_error("ERROR: Unknown property to convert to physical "
                             "units.\n");
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

void Mode::convertToNormalizedUnits(Structure& structure) const
{
    structure.toNormalizedUnits(meanEnergy, convEnergy, convLength, convCharge);

    return;
}

void Mode::convertToPhysicalUnits(Structure& structure) const
{
    structure.toPhysicalUnits(meanEnergy, convEnergy, convLength, convCharge);

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
            SymFnc const& s = it->getSymmetryFunction(i);
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

void Mode::readNeuralNetworkWeights(string const& id,
                                    string const& fileNameFormat)
{
    for (vector<Element>::iterator it = elements.begin();
         it != elements.end(); ++it)
    {
        string fileName = strpr(fileNameFormat.c_str(),
                                it->getAtomicNumber());
        log << strpr("Setting weights for element %2s from file: %s\n",
                     it->getSymbol().c_str(),
                     fileName.c_str());
        vector<double> weights = readColumnsFromFile(fileName,
                                                     vector<size_t>(1, 0)
                                                    ).at(0);
        NeuralNetwork& nn = it->neuralNetworks.at(id);
        nn.setConnections(&(weights.front()));
    }

    return;
}
