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

#ifndef ELEMENTMAP_H
#define ELEMENTMAP_H

#include <cstddef> // std::size_t
#include <map>     // std::map
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/// Contains element map.
class ElementMap
{
public:
    /** Overload [] operator for index search.
     *
     * @param[in] index Element index.
     * @return Symbol of element in map.
     */
    std::string              operator[](std::size_t const index) const;
    /** Overload [] operator for symbol search.
     *
     * @param[in] symbol Element symbol.
     * @return Index of element in map.
     */
    std::size_t              operator[](std::string const symbol) const;
    /** Get element map size.
     *
     * @return Number of elements registered.
     */
    std::size_t              size() const;
    /** Get sorted list of elements in one string (space separated).
     *
     * @return Element string.
     */
    std::string              getElementsString() const;
    /** Get index of given element.
     *
     * @param[in] symbol Element symbol.
     * @return Element index.
     */
    std::size_t              index(std::string const& symbol) const;
    /** Get element symbol for given element index.
     *
     * @param[in] index Element index.
     * @return Element symbol.
     */
    std::string              symbol(std::size_t const index) const;
    /** Get atomic number from element index.
     *
     * @param[in] index Element index in map.
     * @return Atomic number (proton number Z) of element.
     */
    std::size_t              atomicNumber(std::size_t index) const;
    /** Extract all elements and store in element map.
     *
     * @param[in] elementLine String containing all elements, e.g.
     *                        "Cd S Zn", separated by whitespaces.
     * @return Number of registered elements.
     *
     * Sorts all elements according to their atomic number and populates
     * #forwardMap and #reverseMap with the symbol and the corresponding
     * index number.
     */
    std::size_t              registerElements(std::string const& elementLine);
    /** Clear element map.
     */
    void                     deregisterElements();
    /** Get element symbol from atomic number.
     *
     * @param[in] atomicNumber Atomic number of element.
     * @return Element symbol.
     */
    static std::string       symbolFromAtomicNumber(
                                               std::size_t const atomicNumber);
    /** Get atomic number from element string.
     *
     * @param[in] symbol Element symbol (e.g. "He" for Helium).
     * @return Atomic number (proton number Z) of element.
     */
    static std::size_t       atomicNumber(std::string const& symbol);
    /** Get map information as a vector of strings.
     *
     * @return Lines with map information.
     */
    std::vector<std::string> info() const;

private:
    /// Map of elements present and corresponding index number.
    std::map<std::string, std::size_t> forwardMap;
    /// Reverse element map.
    std::map<std::size_t, std::string> reverseMap;
    /// List of element symbols (e.g. "He" for Helium).
    static std::string const           knownElements[];

    /** Check if arguments are sorted according to atomic number.
     *
     * @param[in] symbol1 Element symbol 1.
     * @param[in] symbol2 Element symbol 2.
     * @return `true` if atomic number of element 1 is smaller than atomic
     *         number of element 2, `false` otherwise.
     *
     * Used for sorting in #registerElements().
     */
    static bool compareAtomicNumber(std::string const& symbol1,
                                    std::string const& symbol2);
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline std::string ElementMap::operator[](std::size_t const index) const
{
    return symbol(index);
}

inline std::size_t ElementMap::operator[](std::string const symbol) const
{
    return index(symbol);
}

inline std::size_t ElementMap::size() const
{
    return forwardMap.size();
}

inline std::size_t ElementMap::atomicNumber(std::size_t index) const
{
    return atomicNumber(symbol(index));
}

inline bool ElementMap::compareAtomicNumber(std::string const& symbol1,
                                            std::string const& symbol2)
{
    return atomicNumber(symbol1) < atomicNumber(symbol2);
}

}

#endif
