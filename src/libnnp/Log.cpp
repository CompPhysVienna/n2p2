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
