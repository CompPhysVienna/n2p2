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

#ifndef SYMFNC_H
#define SYMFNC_H

#include "ElementMap.h"
#include <cstddef> // std::size_t
#include <map>     // std::map
#include <set>     // std::set
#include <string>  // std::string
#include <utility> // std::pair
#include <vector>  // std::vector

namespace nnp
{

struct Atom;
class SymFncStatistics;

/** Symmetry function base class.
 *
 * Actual symmetry functions derive from this class. Provides common
 * functionality, e.g. scaling behavior.
 */
class SymFnc
{
public:
    /// List of available scaling types.
    enum ScalingType
    {
        /** @f$G_\text{scaled} = G@f$
         */
        ST_NONE,
        /** @f$G_\text{scaled} = S_\text{min} + \left(S_\text{max} -
         * S_\text{min}\right) \cdot \frac{G - G_\text{min}}
         * {G_\text{max} - G_\text{min}} @f$
         */
        ST_SCALE,
        /** @f$G_\text{scaled} = G - \left<G\right>@f$
         */
        ST_CENTER,
        /** @f$G_\text{scaled} = S_\text{min} + \left(S_\text{max} -
         * S_\text{min}\right) \cdot \frac{G - \left<G\right>}
         * {G_\text{max} - G_\text{min}} @f$
         */
        ST_SCALECENTER,
        /** @f$G_\text{scaled} = S_\text{min} + \left(S_\text{max} -
         * S_\text{min}\right) \cdot \frac{G - \left<G\right>}{\sigma_G} @f$
         */
        ST_SCALESIGMA
    };

    /** Virtual destructor
     */
    virtual             ~SymFnc() {};
    /** Overload == operator.
     */
    virtual bool        operator==(SymFnc const& rhs) const = 0;
    /** Overload < operator.
     */
    virtual bool        operator<(SymFnc const& rhs) const = 0;
    /** Overload != operator.
     */
    bool                operator!=(SymFnc const& rhs) const;
    /** Overload > operator.
     */
    bool                operator>(SymFnc const& rhs) const;
    /** Overload <= operator.
     */
    bool                operator<=(SymFnc const& rhs) const;
    /** Overload >= operator.
     */
    bool                operator>=(SymFnc const& rhs) const;
    /** Set parameters.
     *
     * @param[in] parameterString String containing all parameters for this
     *                            symmetry function.
     */
    virtual void        setParameters(std::string const& parameterString) = 0;
    /** Change length unit.
     *
     * @param[in] convLength Multiplicative length unit conversion factor.
     *
     * @note This will permanently change all symmetry function parameters with
     * dimension length. For convenience, some member functions for printing
     * (getSettingsLine(), parameterLine(), calculateRadialPart() and
     * calculateAngularPart()) will temporarily undo this change. In contrast,
     * the member getter functions, e.g. getRc(), will return the internally
     * stored values.
     */
    virtual void        changeLengthUnit(double convLength) = 0;
    /** Get settings file line from currently set parameters.
     *
     * @return Settings file string ("symfunction_short ...").
     */
    virtual std::string getSettingsLine() const = 0;
    /** Calculate symmetry function for one atom.
     *
     * @param[in,out] atom Atom for which the symmetry function is caluclated.
     * @param[in] derivatives If also symmetry function derivatives will be
     *                        calculated and saved.
     */
    virtual void        calculate(Atom&      atom,
                                  bool const derivatives) const = 0;
    /** Give symmetry function parameters in one line.
     *
     * @return String containing symmetry function parameter values.
     */
    virtual std::string parameterLine() const = 0;
    /** Get description with parameter names and values.
     *
     * @return Vector of parameter description strings.
     */
    virtual std::vector<std::string>
                        parameterInfo() const;
    /** Set symmetry function scaling type.
     *
     * @param[in] scalingType Desired symmetry function scaling type.
     * @param[in] statisticsLine String containing symmetry function statistics
     *                           ("min max mean sigma").
     * @param[in] Smin Minimum for scaling range @f$S_\text{min}@f$.
     * @param[in] Smax Maximum for scaling range @f$S_\text{max}@f$.
     */
    void                setScalingType(ScalingType scalingType,
                                       std::string statisticsLine,
                                       double      Smin,
                                       double      Smax);
    /** Apply symmetry function scaling and/or centering.
     *
     * @param[in] value Raw symmetry function value.
     * @return Scaled symmetry function value.
     */
    double              scale(double value) const;
    /** Undo symmetry function scaling and/or centering.
     *
     * @param[in] value Scaled symmetry function value.
     * @return Raw symmetry function value.
     */
    double              unscale(double value) const;
    /** Get private #type member variable.
     */
    std::size_t         getType() const;
    /** Get private #index member variable.
     */
    std::size_t         getIndex() const;
    /** Get private #lineNumber member variable.
     */
    std::size_t         getLineNumber() const;
    /** Get private #ec member variable.
     */
    std::size_t         getEc() const;
    /** Get private #minNeighbors member variable.
     */
    std::size_t         getMinNeighbors() const;
    /** Get private #rc member variable.
     */
    double              getRc() const;
    /** Get private #Gmin member variable.
     */
    double              getGmin() const;
    /** Get private #Gmax member variable.
     */
    double              getGmax() const;
    /** Get private #scalingFactor member variable.
     */
    double              getScalingFactor() const;
    /** Get private #convLength member variable.
     */
    double              getConvLength() const;
    /** Get private #parameters member variable.
     */
    std::set<
    std::string>        getParameters() const;
    /** Get private #indexPerElement member variable.
     */
    std::vector<
    std::size_t>        getIndexPerElement() const;
    /** Set private #index member variable.
     *
     * @param[in] index Symmetry function index.
     */
    void                setIndex(std::size_t index);
    /** Set private #indexPerElement member variable.
     *
     * @param[in] elementIndex Element index.
     * @param[in] index Symmetry function index.
     */
    void                setIndexPerElement(std::size_t elementIndex,
                                           std::size_t index);
    /** Set line number.
     *
     * @param[in] lineNumber Line number in settings file.
     */
    void                setLineNumber(std::size_t lineNumber);
    /** Get string with scaling information.
     *
     * @return Scaling information string.
     */
    std::string         scalingLine() const;
    /** Calculate (partial) symmetry function value for one given distance.
     *
     * @param[in] distance Distance between two atoms.
     * @return Symmetry function value.
     *
     * @note This function is not used for actual calculations but only for
     * plotting symmetry functions. Derived classes should implement a
     * meaningful symmetry function value for visualization.
     */
    virtual double      calculateRadialPart(double distance) const = 0;
    /** Calculate (partial) symmetry function value for one given angle.
     *
     * @param[in] angle Angle between triplet of atoms (in radians).
     * @return Symmetry function value.
     *
     * @note This function is not used for actual calculations but only for
     * plotting symmetry functions. Derived classes should implement a
     * meaningful symmetry function value for visualization.
     */
    virtual double      calculateAngularPart(double angle) const = 0;
    /** Check whether symmetry function is relevant for given element.
     *
     * @param[in] index Index of given element.
     * @return True if symmetry function is sensitive to given element, false
     *         otherwise.
     */
    virtual bool        checkRelevantElement(std::size_t index) const = 0;
#ifndef NNP_NO_SF_CACHE
    /** Get unique cache identifiers.
     *
     * @return Vector of string identifying the type of cache this symmetry
     *         function requires.
     */
    virtual std::vector<
    std::string>        getCacheIdentifiers() const;
    /** Add one cache index for given neighbor element and check identifier.
     *
     * @param[in] element Index of neighbor atom element.
     * @param[in] cacheIndex Cache index in Atom::Neighbor.
     * @param[in] cacheIdentifier Cache identifier for checking.
     */
    void                addCacheIndex(std::size_t element,
                                      std::size_t cacheIndex,
                                      std::string cacheIdentifier);
    /// Getter for #cacheIndices.
    std::vector<std::vector<
    std::size_t>>       getCacheIndices() const;
#endif

protected:
    typedef std::map<std::string,
                     std::pair<std::string, std::string> > PrintFormat;
    typedef std::vector<std::string>                       PrintOrder;
    /// Symmetry function type.
    std::size_t                type;
    /// Copy of element map.
    ElementMap                 elementMap;
    /// Symmetry function index (per element).
    std::size_t                index;
    /// Line number.
    std::size_t                lineNumber;
    /// Element index of center atom.
    std::size_t                ec;
    /// Minimum number of neighbors required.
    std::size_t                minNeighbors;
    /// Minimum for scaling range.
    double                     Smin;
    /// Maximum for scaling range.
    double                     Smax;
    /// Minimum unscaled symmetry function value.
    double                     Gmin;
    /// Maximum unscaled symmetry function value.
    double                     Gmax;
    /// Mean unscaled symmetry function value.
    double                     Gmean;
    /// Sigma of unscaled symmetry function values.
    double                     Gsigma;
    /// Cutoff radius @f$r_c@f$.
    double                     rc;
    /// Scaling factor.
    double                     scalingFactor;
    /// Data set normalization length conversion factor.
    double                     convLength;
    /// Symmetry function scaling type used by this symmetry function.
    ScalingType                scalingType;
    /// Set with symmetry function parameter IDs (lookup for printing).
    std::set<std::string>      parameters;
    /// Per-element index for derivative memory in Atom::Neighbor::dGdr arrays.
    std::vector<std::size_t>   indexPerElement;
#ifndef NNP_NO_SF_CACHE
    /// Cache indices for each element.
    std::vector<
    std::vector<std::size_t>>  cacheIndices;
#endif
    /// Width of the SFINFO parameter description field (see #parameterInfo()).
    static std::size_t const   sfinfoWidth;
    /// Map of parameter format strings and empty strings.
    static PrintFormat const   printFormat;
    /// Vector of parameters in order of printing.
    static PrintOrder const    printOrder;

    /** Constructor, initializes #type.
     */
    SymFnc(std::size_t type, ElementMap const&);
    /** Initialize static print format map for all possible parameters.
     */
    static PrintFormat const initializePrintFormat();
    /** Initialize static print order vector for all possible parameters.
     */
    static PrintOrder const  initializePrintOrder();
    /** Generate format string for symmetry function parameter printing.
     *
     * @return C-Style format string.
     */
    std::string              getPrintFormat() const;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool SymFnc::operator!=(SymFnc const& rhs) const
{
    return !((*this) == rhs);
}

inline bool SymFnc::operator>(SymFnc const& rhs) const
{
    return rhs < (*this);
}

inline bool SymFnc::operator<=(SymFnc const& rhs) const
{
    return !((*this) > rhs);
}

inline bool SymFnc::operator>=(SymFnc const& rhs) const
{
    return !((*this) < rhs);
}

inline std::size_t SymFnc::getType() const { return type; }
inline std::size_t SymFnc::getEc() const { return ec; }
inline std::size_t SymFnc::getIndex() const { return index; }
inline std::size_t SymFnc::getLineNumber() const { return lineNumber; }
inline std::size_t SymFnc::getMinNeighbors() const { return minNeighbors; }
inline double SymFnc::getRc() const { return rc; }
inline double SymFnc::getGmin() const { return Gmin; }
inline double SymFnc::getGmax() const { return Gmax; }
inline double SymFnc::getScalingFactor() const { return scalingFactor; }
inline double SymFnc::getConvLength() const { return convLength; }

inline void SymFnc::setIndex(std::size_t index)
{
    this->index = index;
    return;
}

inline void SymFnc::setLineNumber(std::size_t lineNumber)
{
    this->lineNumber = lineNumber;
    return;
}

inline std::set<std::string> SymFnc::getParameters() const
{
    return parameters;
}

inline std::vector<std::size_t> SymFnc::getIndexPerElement() const
{
    return indexPerElement;
}

inline void SymFnc::setIndexPerElement(std::size_t elementIndex,
                                       std::size_t index)
{
    indexPerElement.at(elementIndex) = index;
    return;
}

#ifndef NNP_NO_SF_CACHE
inline std::vector<std::vector<std::size_t>> SymFnc::getCacheIndices() const
{
    return cacheIndices;
}
#endif

}

#endif
