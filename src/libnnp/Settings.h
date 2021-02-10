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

#ifndef SETTINGS_H
#define SETTINGS_H

#include <cstddef> // std::size_t
#include <fstream> // std::ofstream
#include <map>     // std::multimap
#include <memory>  // std::shared_ptr
#include <string>  // std::string
#include <utility> // std::pair
#include <vector>  // std::vector

namespace nnp
{

/// Reads and analyzes settings file and stores parameters.
class Settings
{
public:
    /// Keyword properties.
    struct Key
    {
        /// Whether this keyword has no alternative definitions or spellings.
        bool isUnique() const {return (words.size() == 1);}

        /// A short description of the keyword.
        std::string              description;
        /// Alternative keywords (first entry is main name).
        std::vector<std::string> words;
    };

    typedef std::multimap<std::string,
                          std::pair<std::string, std::size_t> > KeyMap;
    typedef std::pair<KeyMap::const_iterator,
                      KeyMap::const_iterator>                   KeyRange;
    typedef std::map<std::string,
                     std::shared_ptr<Key>> const                KeywordList;

    /** Overload [] operator.
     *
     * @param[in] keyword Keyword string.
     * @return Value string corresponding to keyword.
     *
     * Internally calls #getValue().
     */
    std::string              operator[](std::string const& keyword) const;
    /** Load a file with settings.
     *
     * @param[in] fileName Name of file containing settings.
     *
     * @return Number of critical problems detected.
     */
    std::size_t              loadFile(
                                     std::string const& fileName = "input.nn");
    /** Check if keyword is present in settings file.
     *
     * @param[in] keyword Keyword string.
     * @param[in] exact If `True` check only for exact spelling, no
     *                  alternative keywords allowed.
     *
     * @return `True` if keyword exists, `False` otherwise.
     *
     * @note This throws an error if the keyword is not in the list of known
     *       keywords.
     */
    bool                     keywordExists(
                                       std::string const& keyword,
                                       bool               exact = false) const;
    /** Check for keyword and alternatives, throw exception if not present.
     *
     * @param[in] keyword Original keyword for which alternatives should be
     *                    searched.
     *
     * @return Keyword or alternative found in file contents.
     */
    std::string              keywordCheck(std::string const& keyword) const;
    /** Get value for given keyword.
     *
     * @param[in] keyword Keyword string.
     *
     * @return Value string corresponding to keyword.
     *
     * If keyword is present multiple times only the value of the first
     * keyword-value pair is returned (use #getValues() instead).
     */
    std::string              getValue(std::string const& keyword) const;
    /** Get all keyword-value pairs for given keyword.
     *
     * @param[in] keyword Keyword string.
     *
     * @return Pair with begin and end values for iteration.
     *
     * Useful if keyword appears multiple times. Returns a pair representing
     * begin and end for an iterator over all corresponding keyword-value
     * pairs. Use like this:
     * ```
     * Settings* s = new Settings("input.nn");
     * Settings::KeyRange r = s->getValues("symfunction_short");
     * for (Settings::KeyMap::const_iterator i = r.first; i != r.second; i++)
     * {
     *     cout << "keyword: " << i->first
     *          << " value: " << i->second.first
     *          << " line : " << i->second.second << "\n";
     * }
     * ```
     */
    KeyRange                 getValues(std::string const& keyword) const;
    /** Get logged information about settings file.
     *
     * @return Vector with log lines.
     */
    std::vector<std::string> info() const;
    /** Get complete settings file.
     *
     * @return Vector with settings file lines.
     */
    std::vector<std::string> getSettingsLines() const;
    /** Write complete settings file.
     *
     * @param[in,out] file Settings file.
     */
    void                     writeSettingsFile(
                                             std::ofstream* const& file) const;

private:
    /// Vector of all lines in settings file.
    std::vector<std::string>                        lines;
    /// Vector with log lines.
    std::vector<std::string>                        log;
    /// Map containing all keyword-value pairs.
    KeyMap                                          contents;
    /// Map containing all known keywords and a description.
    static KeywordList                              knownKeywords;
    /// %Settings file name.
    std::string                                     fileName;

    /** Read file once and save all lines in #lines vector.
     */
    void                      readFile();
    /** Parse lines and create #contents map.
     *
     * @return Number of critical problems detected.
     */
    std::size_t               parseLines();
    /** Check if all keywords are in known-keywords database and for
     * duplicates.
     *
     * @return Number of detected/critical problems.
     */
    std::pair<
    std::size_t, std::size_t> sanityCheck();
};

}

#endif
