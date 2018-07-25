// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Log.h"
#include <iostream> // std::cout

using namespace std;
using namespace nnp;

Log::Log() : writeToStdout(true)
{
}

Log& Log::operator<<(string const& entry)
{
    addLogEntry(entry);
    return *this;
}

Log& Log::operator<<(vector<string> const& entries)
{
    addMultipleLogEntries(entries);
    return *this;
}

void Log::addLogEntry(string const& entry)
{
    memory.push_back(entry);

    if (writeToStdout)
    {
        cout << entry;
        cout.flush();
    }

    for (vector<FILE**>::const_iterator it = cFilePointers.begin();
         it != cFilePointers.end(); ++it)
    {
        if ((**it) != 0)
        {
            fprintf((**it), "%s", entry.c_str());
            fflush((**it));
        }
    }

    for (vector<ofstream*>::const_iterator it = streamPointers.begin();
         it != streamPointers.end(); ++it)
    {
        if ((**it).is_open())
        {
            (**it) << entry;
            (**it).flush();
        }
    }

    return;
}

void Log::addMultipleLogEntries(vector<string> const& entries)
{
    for (vector<string>::const_iterator it = entries.begin();
         it != entries.end(); ++it)
    {
        addLogEntry(*it);
    }

    return;
}

void Log::registerCFilePointer(FILE** const& filePointer)
{
    cFilePointers.push_back(filePointer);
    return;
}

void Log::registerStreamPointer(ofstream* const& streamPointer)
{
    streamPointers.push_back(streamPointer);
    return;
}

vector<string> Log::getLog() const
{
    return memory;
}
