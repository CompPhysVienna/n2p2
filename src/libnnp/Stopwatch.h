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

#ifndef STOPWATCH_H
#define STOPWATCH_H

#if !(defined(__linux__) || defined(__MACH__))
#pragma message("WARNING: Platform not supported.")
#define NOTIME
#endif

#ifdef NOTIME
#pragma message("WARNING: Compiling dummy Stopwatch class (-DNOTIME).")
#endif

#include <ctime>
#ifdef __MACH__
#include <mach/mach_time.h>
#endif

namespace nnp
{

/// Implements a simple stopwatch on different platforms.
class Stopwatch
{

public:
    Stopwatch();
    /// Start the stopwatch.
    void                start();
    /// Take a split time (returns total time elapsed).
    double              split();
    /// Take a split time (returns total time elapsed and
    /// saves split time in argument).
    double              split(double* lap);
    /// Stop and return elapsed time.
    double              stop();
    /// Return total time elapsed (of a stopped watch).
    double              getTimeElapsed() const;
    /// Reset stopwatch (total time zero, not running).
    void                reset();

private:
    enum State
    {
        STOPPED,
        RUNNING
    };

    State               state;
    static const double NSEC;
    double              timeElapsed;
#ifdef NOTIME
#elif __linux__
    timespec            time;
#elif __MACH__
    uint64_t            time;
#endif

    double              updateTime();

};

}

#endif
