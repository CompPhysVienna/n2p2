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

#ifndef SETUPANALYSIS_H
#define SETUPANALYSIS_H

#include "Mode.h"
#include <string> // std::string

namespace nnp
{

class SetupAnalysis : public Mode
{
public:
    /** Write radial and angular part of symmetry functions to file.
     *
     * @param[in] numPoints Number of symmetry function points to plot.
     * @param[in] fileNameFormat File name with place holder for element and
     *                           symmetry function number.
     */
    void writeSymmetryFunctionShape(std::size_t       numPoints,
                                    std::string const fileNameFormat
                                        = "sf.%03zu.%04zu.out");
};

}

#endif
