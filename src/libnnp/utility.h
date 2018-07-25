// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef UTILITY_H
#define UTILITY_H

#include <cstdio>    // FILE
#include <fstream>   // std::ofstream
#include <map>       // std::map
#include <sstream>   // std::stringstream
#include <stdexcept> // std::range_error
#include <string>    // std::string
#include <vector>    // std::vector

namespace nnp
{

/** Split string at each delimiter.
 *
 * @param[in] input Input string.
 * @param[in] delimiter Delimiter character (default ' ').
 * @return Vector containing the string parts.
 */
std::vector<std::string> split(std::string const& input,
                               char               delimiter = ' ');
/** Remove leading and trailing whitespaces from string.
 *
 * @param[in] line Input string.
 * @param[in] whitespace All characters identified as whitespace (default:
 *                       " \t").
 * @return String without leading and trailing whitespaces.
 */
std::string              trim(std::string const& line,
                              std::string const& whitespace = " \t");
/** Replace multiple whitespaces with fill.
 *
 * @param[in] line Input string.
 * @param[in] whitespace All characters identified as whitespace (default:
 *                       " \t").
 * @param[in] fill Replacement character (default: " ").
 * @return String with fill replacements.
 *
 * Calls #trim() at the beginning.
 */
std::string              reduce(std::string const& line,
                                std::string const& whitespace = " \t",
                                std::string const& fill = " ");
/** Pad string to given length with fill character.
 *
 * @param[in] input Input string.
 * @param[in] num Desired width of padded string.
 * @param[in] fill Padding character.
 * @param[in] right If true, pad right. Pad left otherwise.
 *
 * @return String padded to desired length or original string if length of old
 *         string is not shorter than desired length.
 */
std::string              pad(std::string const& input,
                             std::size_t        num,
                             char               fill = ' ',
                             bool               right = true);
/** String version of printf function.
 */
std::string              strpr(const char* format, ...);
/** Generate output file header with info section and column labels.
 *
 * @param[in] title Multiple lines of title text.
 * @param[in] colLength Length of the individual columns.
 * @param[in] colName Names of the individual columns.
 * @param[in] colInfo Description of the individual columns.
 * @param[in] commentChar Character used for starting each line and to create
 *                        separator lines.
 *
 * @return Vector of strings, one entry per line.
 */
std::vector<std::string> createFileHeader(
                            std::vector<std::string> const& title,
                            std::vector<std::size_t> const& colLength,
                            std::vector<std::string> const& colName,
                            std::vector<std::string> const& colInfo,
                            char const&                     commentChar = '#');
/** Append multiple lines of strings to open file stream.
 *
 * @param[in,out] file File stream to append lines to.
 * @param[in] lines Vector of strings with line content.
 *
 */
void                     appendLinesToFile(
                                         std::ofstream&                 file,
                                         std::vector<std::string> const lines);
/** Append multiple lines of strings to open file pointer.
 *
 * @param[in,out] file File pointer.
 * @param[in] lines Vector of strings with line content.
 *
 */
void                     appendLinesToFile(
                                         FILE* const&                   file,
                                         std::vector<std::string> const lines);
/** Safely access map entry.
 *
 * @param[in] stdMap Map to search in.
 * @param[in] key Key requested.
 *
 * The .at() member function of maps is available only for C++11.
 */
template<typename K, typename V>
V const&                 safeFind(std::map<K, V> const&           stdMap,
                                  typename
                                  std::map<K, V>::key_type const& key)
{
    if (stdMap.find(key) == stdMap.end())
    {
        std::stringstream message;
        message << "ERROR: No map entry found for key \"";
        message << key;
        message << "\".\n";
        throw std::range_error(message.str());
    }
    return stdMap.find(key)->second;
}

/** Compare pointer targets
 *
 * @param[in] lhs Pointer to left side of comparison.
 * @param[in] rhs Pointer to right side of comparison.
 * @return `True` if dereferenced `lhs` < dereferenced `rhs`, `False`
 *         otherwise.
 */
template<typename T>
bool                     comparePointerTargets(T* lhs, T* rhs)
{
    return ((*lhs) < (*rhs));
}

/** Integer version of power function, "fast exponentiation algorithm".
 *
 * @param[in] x Base number.
 * @param[in] n Integer exponent.
 *
 * @return x^n
 */
double                   pow_int(double x, int n);

}

#endif
