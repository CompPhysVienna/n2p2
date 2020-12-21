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

#include "utility.h"
#include <algorithm> // std::max
#include <cstdio>    // vsprintf
#include <cstdarg>   // va_list, va_start, va_end
#include <iomanip>   // std::setw
#include <limits>    // std::numeric_limits
#include <sstream>   // std::istringstream
#include <stdexcept> // std::runtime_error

#define STRPR_MAXBUF 1024

using namespace std;

namespace nnp
{

vector<string> split(string const& input, char delimiter)
{
    vector<string> parts;
    stringstream ss;

    ss.str(input);
    string item;
    while (getline(ss, item, delimiter)) {
        parts.push_back(item);
    }

    return parts;
}

string trim(string const& line, string const& whitespace)
{
    size_t const begin = line.find_first_not_of(whitespace);
    if (begin == string::npos)
    {
        return "";
    }
    size_t const end = line.find_last_not_of(whitespace);
    size_t const range = end - begin + 1;

    return line.substr(begin, range);
}

string reduce(string const& line,
              string const& whitespace,
              string const& fill)
{
    string result = trim(line, whitespace);

    size_t begin = result.find_first_of(whitespace);
    while (begin != string::npos)
    {
        size_t const end = result.find_first_not_of(whitespace, begin);
        size_t const range = end - begin;
        result.replace(begin, range, fill);
        size_t const newBegin = begin + fill.length();
        begin = result.find_first_of(whitespace, newBegin);
    }

    return result;
}

string pad(string const& input, size_t num, char fill, bool right)
{
    string result = input;

    if (input.size() >= num) return result;

    string pad(num - input.size(), fill);
    if (right) return result + pad;
    else return pad + result;
}

string strpr(const char* format, ...)
{
    char buffer[STRPR_MAXBUF];

    va_list args;
    va_start(args, format);
    vsprintf(buffer, format, args);
    va_end(args);

    string s(buffer);

    return s;
}

vector<string> createFileHeader(vector<string> const& title,
                                vector<size_t> const& colSize,
                                vector<string> const& colName,
                                vector<string> const& colInfo,
                                char           const& commentChar)
{
    size_t const stdWidth = 80;
    vector<string> h;
    if (!(colSize.size() == colName.size() &&
          colName.size() == colInfo.size()))
    {
        throw runtime_error("ERROR: Could not create file header, column sizes"
                            " are not equal.\n");
    }
    size_t numCols = colSize.size();

    // Separator line definition.
    string sepLine(stdWidth, commentChar);
    sepLine += '\n';
    // Start string.
    string startStr(1, commentChar);

    // Add title lines.
    h.push_back(sepLine);
    for (vector<string>::const_iterator it = title.begin();
         it != title.end(); ++it)
    {
        h.push_back(startStr + ' ' + (*it) + '\n');
    }
    h.push_back(sepLine);

    // Determine maximum width of names.
    size_t widthNames = 0;
    for (vector<string>::const_iterator it = colName.begin();
         it != colName.end(); ++it)
    {
        widthNames = max(widthNames, it->size());
    }

    // Column description section.
    stringstream s;
    s << startStr << ' '
      << left << setw(4) << "Col" << ' '
      << left << setw(widthNames) << "Name" << ' '
      << left << "Description\n";
    h.push_back(s.str());
    h.push_back(sepLine);
    for (size_t i = 0; i < numCols; ++i)
    {
        s.clear();
        s.str(string());
        s << startStr << ' '
          << left << setw(4) << i + 1 << ' '
          << left << setw(widthNames) << colName.at(i) << ' '
          << colInfo.at(i) << '\n';
        h.push_back(s.str());
    }

    // Determine total data width.
    size_t widthData = 0;
    for (vector<size_t>::const_iterator it = colSize.begin();
         it != colSize.end(); ++it)
    {
        widthData += (*it) + 1;
    }
    widthData -= 1;

    if (widthData > stdWidth)
    {
        // Long separator line.
        h.push_back(string(widthData, commentChar) + "\n");
    }
    else
    {
        h.push_back(sepLine);
    }

    // Write column number.
    s.clear();
    s.str(string());
    s << startStr;
    for (size_t i = 0; i < numCols; ++i)
    {
        size_t n = colSize.at(i);
        if (i == 0) n -= 2;
        s << ' ' << right << setw(n) << i + 1;
    }
    s << '\n';
    h.push_back(s.str());

    // Write column name.
    s.clear();
    s.str(string());
    s << startStr;
    for (size_t i = 0; i < numCols; ++i)
    {
        size_t n = colSize.at(i);
        if (i == 0) n -= 2;
        string name = colName.at(i);
        if (name.size() > n)
        {
            name.erase(n, string::npos);
        }
        s << ' ' << right << setw(n) << name;
    }
    s << '\n';
    h.push_back(s.str());

    // Long separator line.
    h.push_back(string(widthData, commentChar) + "\n");

    return h;
}

void appendLinesToFile(ofstream& file, vector<string> const lines)
{
    for (vector<string>::const_iterator it = lines.begin();
         it != lines.end(); ++it)
    {
        file << (*it);
    }

    return;
}

void appendLinesToFile(FILE* const& file, vector<string> const lines)
{
    for (vector<string>::const_iterator it = lines.begin();
         it != lines.end(); ++it)
    {
        fprintf(file, "%s", it->c_str());
    }

    return;
}

map<size_t, vector<double>> readColumnsFromFile(string         fileName,
                                                vector<size_t> columns,
                                                char           comment)
{
    map<size_t, vector<double>> result;

    sort(columns.begin(), columns.end());
    for (auto col : columns)
    {
        result[col] = vector<double>();
    }

    ifstream file;
    file.open(fileName.c_str());
    if (!file.is_open())
    {
        throw runtime_error("ERROR: Could not open file: \"" + fileName
                            + "\".\n");
    }
    string line;
    while (getline(file, line))
    {
        if (line.size() == 0) continue;
        if (line.at(0) != comment)
        {
            vector<string> splitLine = split(reduce(line));
            for (auto col : columns)
            {
                if (col >= splitLine.size())
                {
                    result[col].push_back(
                        std::numeric_limits<double>::quiet_NaN());
                }
                else
                {
                    result[col].push_back(atof(splitLine.at(col).c_str()));
                }
            }
        }
    }
    file.close();

    return result;
}

double pow_int(double x, int n)
{
    // Need an unsigned integer for bit-shift divison.
    unsigned int m;

    // If negative exponent, take the inverse of x and change sign of n.
    if (n < 0)
    {
        x = 1.0 / x;
        m = -n;
    }
    else m = n;

    // Exponentiation by squaring, "fast exponentiation algorithm"
    double result = 1.0;
    do
    {
        if (m & 1) result *= x;
        // Division by 2.
        m >>= 1;
        x *= x;
    }
    while (m);

    return result;
}

}
