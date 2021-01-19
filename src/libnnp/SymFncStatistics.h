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

#ifndef SYMFNCSTATISTICS
#define SYMFNCSTATISTICS

#include <cstddef> // std::size_t
#include <map>     // std::map
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

class SymFncStatistics
{
public:
    /** Struct containing statistics gathered during symmetry function
     * calculation.
     */
    struct Container
    {
        /// Counts total number of symmetry function evaluations.
        std::size_t              count;
        /// Counts extrapolation warnings.
        std::size_t              countEW;
        /// Symmetry function type.
        std::size_t              type;
        /// Minimum symmetry function value encountered.
        double                   min;
        /// Maximum symmetry function value encountered.
        double                   max;
        /// Minimum symmetry function from scaling data.
        double                   Gmin;
        /// Maximum symmetry function from scaling data.
        double                   Gmax;
        /// Sum of symmetry function values (to compute mean).
        double                   sum;
        /// Sum of squared symmetry function values (to compute sigma).
        double                   sum2;
        /// Element string of central atom of symmetry function.
        std::string              element;
        /// Structure indices for which extrapolation warnings occured.
        std::vector<std::size_t> indexStructureEW;
        /// Atom indices for which extrapolation warnings occured.
        std::vector<std::size_t> indexAtomEW;
        /// Out-of-bounds values causing extrapolation warnings.
        std::vector<double>      valueEW;

        /** Constructor, initializes contents to zero.
         *
         * Except min, max which are initialized to largest positive and
         * negative double values, respectively.
         */
        Container();
        /** Reset all values.
         */
        void reset();
        /** Reset only statistics.
         */
        void resetStatistics();
        /** Reset only extrapolation warnings.
         */
        void resetExtrapolationWarnings();
    };

    /// Whether statistics are gathered.
    bool                             collectStatistics;
    /// Whether extrapolation warnings are logged.
    bool                             collectExtrapolationWarnings;
    /// Whether to write out extrapolation warnings immediately as they occur.
    bool                             writeExtrapolationWarnings;
    /// Whether to raise an exception in case of extrapolation warnings.
    bool                             stopOnExtrapolationWarnings;
    /// Map for all symmetry functions containing all gathered information.
    std::map<std::size_t, Container> data;

    /** Constructor, initializes bool variables.
     *
     * Defaults are #collectStatistics = `false`, #collectExtrapolationWarnings
     * = `false`, #writeExtrapolationWarnings = `false`,
     * #stopOnExtrapolationWarnings = `false`.
     */
    SymFncStatistics();
    /** Update symmetry function statistics with one value.
     *
     * @param[in] index Symmetry function index.
     * @param[in] value Symmetry function value.
     */
    void         addValue(std::size_t index, double value);
    /** Add extrapolation warning entry.
     *
     * @param[in] index Symmetry function index.
     * @param[in] type Symmetry function type.
     * @param[in] value Unscaled symmetry function value.
     * @param[in] Gmin Minimum symmetry function value from scaling data.
     * @param[in] Gmax Maximum symmetry function value from scaling data.
     * @param[in] element Symmetry function element string.
     * @param[in] indexStructure Index of structure of affected atom.
     * @param[in] indexAtom Index of affected atom.
     */
    void         addExtrapolationWarning(std::size_t index,
                                         std::size_t type,
                                         double      value,
                                         double      Gmin,
                                         double      Gmax,
                                         std::string element,
                                         std::size_t indexStructure,
                                         std::size_t indexAtom);
    /** Get lines with extrapolation warnings.
     *
     * @return Extrapolation warning lines.
     */
    std::vector<
    std::string> getExtrapolationWarningLines() const;
    /** Count total number of extrapolation warnings.
     *
     * @return Sum of extrapolation warnings for all symmetry functions.
     */
    std::size_t  countExtrapolationWarnings() const;
    /** Reset statistics for all symmetry functions at once.
     */
    void         resetStatistics();
    /** Reset extrapolation warnings for all symmetry functions at once.
     */
    void         resetExtrapolationWarnings();
    /** Completely erase database.
     */
    void         clear();
};

}

#endif
