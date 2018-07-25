// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

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
