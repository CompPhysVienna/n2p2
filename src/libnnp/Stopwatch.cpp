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

#include <cstdio>
#include "Stopwatch.h"

using namespace nnp;

const double Stopwatch::NSEC = 1E-9;

Stopwatch::Stopwatch()
{
    state        = STOPPED;
    timeTotal    = 0.0;
    timeLoop     = 0.0;
#ifdef NNP_NO_TIME
    //fprintf(stderr, "WARNING: Stopwatch is using dummy implementation.\n");
#elif __linux__
    time.tv_sec  = 0;
    time.tv_nsec = 0;
#elif __MACH__
    time         = 0;
#endif

}

void Stopwatch::start(bool newLoop)
{
    if (state == STOPPED)
    {
#ifdef NNP_NO_TIME
#elif __linux__
        clock_gettime(CLOCK_MONOTONIC, &time);
#elif __MACH__
        time = mach_absolute_time();
#endif
        if (newLoop) timeLoop = 0.0;
        state = RUNNING;
    }
    else
    {
        fprintf(stderr,
                "WARNING: Unable to start clock, clock already running.\n");
    }

    return;
}

double Stopwatch::stop()
{
    stopTime();

    return getTotal();
}

double Stopwatch::loop()
{
    stopTime();

    return getLoop();
}

void Stopwatch::reset()
{
    state        = STOPPED;
    timeTotal    = 0.0;
    timeLoop     = 0.0;
#ifdef NNP_NO_TIME
#elif __linux__
    time.tv_sec  = 0;
    time.tv_nsec = 0;
#elif __MACH__
    time         = 0;
#endif

    return;
}

void Stopwatch::stopTime()
{
    double timeInterval = updateTime();
    timeLoop += timeInterval;
    timeTotal += timeInterval;
    state = STOPPED;

    return;
}

double Stopwatch::updateTime()
{
    if (state == RUNNING)
    {
#ifdef NNP_NO_TIME
        return 1.0;
#elif __linux__
        time_t   secLast  = time.tv_sec;
        long     nsecLast = time.tv_nsec;

        clock_gettime(CLOCK_MONOTONIC, &time);
        return (double)(time.tv_sec  - secLast)
             + (double)(time.tv_nsec - nsecLast) * NSEC;
#elif __MACH__
        uint64_t                         lastTime = time;
        uint64_t                         time     = mach_absolute_time();
        static mach_timebase_info_data_t info     = {0, 0};

        if (info.denom == 0) mach_timebase_info(&info);
        return ((time - lastTime) * (info.numer / info.denom)) * NSEC;
#endif
    }
    else
    {
        fprintf(stderr,
                "WARNING: Unable to update time, clock not running.\n");
        return 0.0;
    }
}
