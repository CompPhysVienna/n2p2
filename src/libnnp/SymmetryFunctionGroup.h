// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef SYMMETRYFUNCTIONGROUP_H
#define SYMMETRYFUNCTIONGROUP_H

#include "CutoffFunction.h"
#include "ElementMap.h"
#include <cstddef> // std::size_t
#include <set>     // std::set
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class SymmetryFunction;
class SymmetryFunctionStatistics;

class SymmetryFunctionGroup
{
public:
    /** Virtual destructor.
     */
    virtual ~SymmetryFunctionGroup() {};
    /** Overload == operator.
     */
    virtual bool operator==(SymmetryFunctionGroup const& rhs) const = 0;
    /** Overload != operator.
     */
    virtual bool operator!=(SymmetryFunctionGroup const& rhs) const = 0;
    /** Overload < operator.
     */
    virtual bool operator<(SymmetryFunctionGroup const& rhs) const = 0;
    /** Overload > operator.
     */
    virtual bool operator>(SymmetryFunctionGroup const& rhs) const = 0;
    /** Overload <= operator.
     */
    virtual bool operator<=(SymmetryFunctionGroup const& rhs) const = 0;
    /** Overload >= operator.
     */
    virtual bool operator>=(SymmetryFunctionGroup const& rhs) const = 0;
    /** Potentially add a member to group.
     *
     * @param[in] symmetryFunction Candidate symmetry function.
     * @return If addition was successful.
     *
     * If symmetry function is compatible with common feature list its pointer
     * will be added to the members vector.
     */
    virtual bool addMember(SymmetryFunction const* const symmetryFunction) = 0;
    /** Sort member symmetry functions.
     *
     * Also allocate and precalculate additional stuff.
     */
    virtual void sortMembers() = 0;
    /** Fill #scalingFactors with values from member symmetry functions.
     */
    virtual void setScalingFactors() = 0;
    /** Calculate all symmetry functions of this group for one atom.
     *
     * @param[in,out] atom Atom for which symmetry functions are caluclated.
     * @param[in] derivatives If also symmetry function derivatives will be
     *                        calculated and saved.
     * @param[in,out] statistics Gathers statistics and extrapolation warnings.
     */
    virtual void calculate(Atom&                       atom,
                           bool const                  derivatives,
                           SymmetryFunctionStatistics& statistics) const = 0;
    /** Give symmetry function group parameters on multiple lines.
     *
     * @return Vector of string containing symmetry function parameters lines.
     */
    virtual std::vector<std::string>
                 parameterLines() const = 0;
    /** Set private #index member variable.
     *
     * @param[in] index Index number of symmetry function group.
     */
    void         setIndex(std::size_t index);
    /** Get private #index member variable.
     */
    std::size_t  getIndex() const;
    /** Get private #type member variable.
     */
    std::size_t  getType() const;
    /** Get private #ec member variable.
     */
    std::size_t  getEc() const;

protected:
    typedef std::map<std::string,
                     std::pair<std::string, std::string> > PrintFormat;
    typedef std::vector<std::string>                       PrintOrder;
    /// Symmetry function type.
    std::size_t                type;
    /// Copy of element map.
    ElementMap                 elementMap;
    /// Symmetry function group index.
    std::size_t                index;
    /// Element index of center atom (common feature).
    std::size_t                ec;
    /// Cutoff radius @f$r_c@f$ (common feature).
    double                     rc;
    /// Cutoff function parameter @f$\alpha@f$ (common feature).
    double                     cutoffAlpha;
    /// Cutoff function used by this symmetry function group.
    CutoffFunction             fc;
    /// Cutoff type used by this symmetry function group (common feature).
    CutoffFunction::CutoffType cutoffType;
    /// Scaling factors of all member symmetry functions.
    std::vector<double>        scalingFactors;
    /// Set of common parameters IDs.
    std::set<std::string>      parametersCommon;
    /// Set of common parameters IDs.
    std::set<std::string>      parametersMember;
    /// Map of parameter format strings and empty strings.
    static PrintFormat const   printFormat;
    /// Vector of parameters in order of printing.
    static PrintOrder const    printOrder;

    /** Constructor, sets type.
     *
     * @param[in] type Type of symmetry functions grouped.
     * @param[in] elementMap Element Map used.
     */
    SymmetryFunctionGroup(std::size_t type, ElementMap const& elementMap);
    /** Initialize static print format map for all possible parameters.
     */
    static PrintFormat const initializePrintFormat();
    /** Initialize static print order vector for all possible parameters.
     */
    static PrintOrder const  initializePrintOrder();
    /** Get common parameter line format string.
     *
     * @return C-style format string.
     */
    std::string              getPrintFormatCommon() const;
    /** Get member parameter line format string.
     *
     * @return C-style format string.
     */
    std::string              getPrintFormatMember() const;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline void SymmetryFunctionGroup::setIndex(size_t index)
{
    this->index = index;
    return;
}

inline std::size_t SymmetryFunctionGroup::getIndex() const
{
    return index;
}

inline std::size_t SymmetryFunctionGroup::getType() const
{
    return type;
}

inline std::size_t SymmetryFunctionGroup::getEc() const
{
    return ec;
}

}

#endif
