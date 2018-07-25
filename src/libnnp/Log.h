// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef LOG_H
#define LOG_H

#include <cstdio>  // FILE
#include <fstream> // std::ofstream
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/** Logging class for library output.
 *
 * Log entries are saved in internal memory and may be redirected to multiple
 * destinations: existing C-style FILE pointer or existing C++ stream.
 */
class Log
{
public:
    /** Constructor.
     *
     * Default output to stdout.
     */
    Log();
    /** Overload << operator.
     *
     * @param[in] entry New log entry.
     *
     * @return Reference to log itself.
     */
    Log&                     operator<<(std::string const& entry);
    /** Overload << operator.
     *
     * @param[in] entries Vector with log entries.
     *
     * @return Reference to log itself.
     */
    Log&                     operator<<(
                                      std::vector<std::string> const& entries);
    /** Add string as new log entry.
     *
     * @param[in] entry New log entry.
     */
    void                     addLogEntry(std::string const& entry);
    /** Add multiple log entries at once.
     *
     * @param[in] entries Vector with log entries.
     */
    void                     addMultipleLogEntries(
                                      std::vector<std::string> const& entries);
    /** Register new C-style FILE* pointer for output.
     *
     * @param[in] filePointer Address of C-style file pointer (FILE*).
     */
    void                     registerCFilePointer(FILE** const& filePointer);
    /** Register new C++ ofstream pointer.
     *
     * @param[in] streamPointer Address of C++ ofstream pointer.
     */
    void                     registerStreamPointer(
                                          std::ofstream* const& streamPointer);
    /** Get complete log memory.
     *
     * @return Log memory.
     */
    std::vector<std::string> getLog() const;

    /// Turn on/off output to stdout.
    bool writeToStdout;

private:
    /// Memory with all log entries.
    std::vector<std::string>    memory;
    /// Storage for C-style FILE* pointers.
    std::vector<FILE**>         cFilePointers;
    /// Storage for C++ ofstream pointers.
    std::vector<std::ofstream*> streamPointers;

    /** Write introductory lines for NNP library.
     */
    void addIntro();
};

}

#endif
