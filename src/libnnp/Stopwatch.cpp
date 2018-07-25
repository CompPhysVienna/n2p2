// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstdio>
#include "Stopwatch.h"

using namespace nnp;

const double Stopwatch::NSEC = 1E-9;

Stopwatch::Stopwatch()
{
    state        = STOPPED;
    timeElapsed  = 0.0;
#ifdef NOTIME
    //fprintf(stderr, "WARNING: Stopwatch is using dummy implementation.\n");
#elif __linux__
    time.tv_sec  = 0;
    time.tv_nsec = 0;
#elif __MACH__
    time         = 0;
#endif

}

void Stopwatch::start()
{
    if (state == STOPPED)
    {
#ifdef NOTIME
#elif __linux__
        clock_gettime(CLOCK_MONOTONIC, &time);
#elif __MACH__
        time = mach_absolute_time();
#endif
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
    timeElapsed += updateTime();

    state = STOPPED;

    return timeElapsed;
}

double Stopwatch::split()
{
    double splitTime = updateTime();

    timeElapsed += splitTime;

    return timeElapsed;
}

double Stopwatch::split(double* lap)
{
    double splitTime = updateTime();

    timeElapsed += splitTime;
    *lap = splitTime;

    return timeElapsed;
}

double Stopwatch::getTimeElapsed() const
{
    return timeElapsed;
}

void Stopwatch::reset()
{
    state           = STOPPED;
    timeElapsed     = 0.0;
#ifdef NOTIME
#elif __linux__
    time.tv_sec  = 0;
    time.tv_nsec = 0;
#elif __MACH__
    time = 0;
#endif

    return;
}

double Stopwatch::updateTime()
{
    if (state == RUNNING)
    {
#ifdef NOTIME
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
        return ((time - LastTime) * (info.numer / info.denom)) * NSEC
#endif
    }
    else
    {
        fprintf(stderr,
                "WARNING: Unable to update time, clock not running.\n");
        return 0.0;
    }
}
