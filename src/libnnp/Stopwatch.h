// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
