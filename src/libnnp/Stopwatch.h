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
#define NNP_NO_TIME
#endif

#ifdef NNP_NO_TIME
#pragma message("WARNING: Compiling dummy Stopwatch class (-DNNP_NO_TIME).")
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
    /** Start the stopwatch.
     *
     * @param[in] resetLoop Optional. If `true` reset loop time, i.e. this
     *                      start() call is then also the beginning of a new
     *                      loop. Use `false` if you want to accumulate further
     *                      time in the loop timer. Default: `true`.
     */
    void                      start(bool newLoop = true);
    /// Stop and return total time.
    double                    stop();
    /// Stop and return loop time.
    double                    loop();
    /// Return total time elapsed (of a stopped watch).
    double                    getTotal() const {return timeTotal;};
    /// Return time elapsed in last loop interval (of a stopped watch).
    double                    getLoop() const {return timeLoop;};
    /// Reset stopwatch (total and loop time zero, clock not running).
    void                      reset();

private:
    enum State
    {
        STOPPED,
        RUNNING
    };

    State               state;
    bool                resetLoop;
    static const double NSEC;
    double              timeTotal;
    double              timeLoop;
#ifdef NNP_NO_TIME
#elif __linux__
    timespec            time;
#elif __MACH__
    uint64_t            time;
#endif

    void                stopTime();
    double              updateTime();
};

}

#endif
