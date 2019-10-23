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

#ifndef ELEMENT_H
#define ELEMENT_H

#include "CutoffFunction.h"
#include "ElementMap.h"
#include "SymmetryFunction.h"
#include "SymmetryFunctionStatistics.h"
#include <cstddef> // std::size_t
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class NeuralNetwork;
class SymmetryFunctionGroup;

/// Contains element-specific data.
class Element
{
public:
    /** Constructor using index.
     */
    Element(std::size_t const index, ElementMap const& elementMap);
    /** Destructor.
     *
     * Necessary because of #symmetryFunctions vector of pointers.
     */
    ~Element();
    /** Set #atomicEnergyOffset.
     */
    void                     setAtomicEnergyOffset(double atomicEnergyOffset);
    /** Get #index.
     */
    std::size_t              getIndex() const;
    /** Get #atomicNumber.
     */
    std::size_t              getAtomicNumber() const;
    /** Get #atomicEnergyOffset.
     */
    double                   getAtomicEnergyOffset() const;
    /** Get #symbol.
     */
    std::string              getSymbol() const;
    /** Add one symmetry function.
     *
     * @param[in] parameters String containing settings for symmetry function.
     * @param[in] lineNumber Line number of symmetry function in settings file.
     */
    void                     addSymmetryFunction(
                                                std::string const& parameters,
                                                std::size_t const& lineNumber);
    /** Change length unit for all symmetry functions.
     *
     * @param[in] convLength Length unit conversion factor.
     */
    void                     changeLengthUnitSymmetryFunctions(
                                                            double convLength);
    /** Sort all symmetry function.
     */
    void                     sortSymmetryFunctions();
    /** Print symmetry function parameter value information.
     */
    std::vector<std::string> infoSymmetryFunctionParameters() const;
    /** Print symmetry function parameter names and values.
     */
    std::vector<std::string> infoSymmetryFunction(std::size_t index) const;
    /** Print symmetry function scaling information.
     */
    std::vector<std::string> infoSymmetryFunctionScaling() const;
    /** Set up symmetry function groups.
     */
    void                     setupSymmetryFunctionGroups();
    /** Print symmetry function group info.
     */
    std::vector<std::string> infoSymmetryFunctionGroups() const;
    /** Set cutoff function for all symmetry functions.
     *
     * @param[in] cutoffType Type of cutoff function.
     * @param[in] cutoffAlpha Cutoff parameter for all functions.
     */
    void                     setCutoffFunction(
                                 CutoffFunction::CutoffType const cutoffType,
                                 double const                     cutoffAlpha);
    /** Set no scaling of symmetry function.
     *
     * Still scaling factors need to be initialized.
     */
    void                     setScalingNone() const;
    /** Set scaling of all symmetry functions.
     *
     * @param[in] scalingType Type of scaling, see
     *                        SymmetryFunction::ScalingType.
     * @param[in] statisticsLine Vector of strings containing statistics for
     *                           all symmetry functions.
     * @param[in] minS Minimum for scaling range.
     * @param[in] maxS Minimum for scaling range.
     */
    void                     setScaling(
                                SymmetryFunction::ScalingType   scalingType,
                                std::vector<std::string> const& statisticsLine,
                                double                          minS,
                                double                          maxS) const;
    /** Get number of symmetry functions.
     *
     * @return Number of symmetry functions.
     */
    std::size_t              numSymmetryFunctions() const;
    /** Get maximum of required minimum number of neighbors for all symmetry
     * functions for this element.
     *
     * @return Minimum number of neighbors required.
     */
    std::size_t              getMinNeighbors() const;
    /** Get minimum cutoff radius of all symmetry functions.
     *
     * @return Minimum cutoff radius.
     */
    double                   getMinCutoffRadius() const;
    /** Get maximum cutoff radius of all symmetry functions.
     *
     * @return Maximum cutoff radius.
     */
    double                   getMaxCutoffRadius() const;
    /** Calculate symmetry functions.
     *
     * @param[in] atom Atom whose symmetry functions are calculated.
     * @param[in] derivatives If symmetry function derivatives will be
     *                        calculated.
     */
    void                     calculateSymmetryFunctions(
                                                 Atom&      atom,
                                                 bool const derivatives) const;
    /** Calculate symmetry functions via groups.
     *
     * @param[in] atom Atom whose symmetry functions are calculated.
     * @param[in] derivatives If symmetry function derivatives will be
     *                        calculated.
     */
    void                     calculateSymmetryFunctionGroups(
                                                Atom&       atom,
                                                bool const  derivatives) const;
    /** Update symmetry function statistics.
     *
     * @param[in] atom Atom with symmetry function values.
     *
     * @return Number of extrapolation warnings encountered.
     *
     * This function checks also for extrapolation warnings.
     */
    std::size_t              updateSymmetryFunctionStatistics(
                                                             Atom const& atom);
    /** Get symmetry function instance.
     *
     * @param[in] index Symmetry function index.
     *
     * @return Symmetry function object.
     */
    SymmetryFunction const&  getSymmetryFunction(std::size_t index) const;

    /// Neural network pointer for this element.
    NeuralNetwork*             neuralNetwork;
    /// Symmetry function statistics.
    SymmetryFunctionStatistics statistics;

private:
    /// Copy of element map.
    ElementMap                          elementMap;
    /// Global index of this element.
    std::size_t                         index;
    /// Atomic number of this element.
    std::size_t                         atomicNumber;
    /// Offset energy for every atom of this element.
    double                              atomicEnergyOffset;
    /// Element symbol.
    std::string                         symbol;
    /// Vector of pointers to symmetry functions.
    std::vector<SymmetryFunction*>      symmetryFunctions;
    /// Vector of pointers to symmetry function groups.
    std::vector<SymmetryFunctionGroup*> symmetryFunctionGroups;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline void Element::setAtomicEnergyOffset(double atomicEnergyOffset)
{
    this->atomicEnergyOffset = atomicEnergyOffset;

    return;
}

inline size_t Element::getIndex() const
{
    return index;
}

inline size_t Element::getAtomicNumber() const
{
    return atomicNumber;
}

inline double Element::getAtomicEnergyOffset() const
{
    return atomicEnergyOffset;
}

inline std::string Element::getSymbol() const
{
    return symbol;
}

inline
std::vector<std::string> Element::infoSymmetryFunction(std::size_t index) const
{
    return symmetryFunctions.at(index)->parameterInfo();
}

inline size_t Element::numSymmetryFunctions() const
{
    return symmetryFunctions.size();
}

inline SymmetryFunction const& Element::getSymmetryFunction(
                                                       std::size_t index) const
{
    return *(symmetryFunctions.at(index));
}

}

#endif
