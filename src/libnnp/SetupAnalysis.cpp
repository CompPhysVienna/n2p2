// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "SetupAnalysis.h"
#include "utility.h"
#include <cmath> // M_PI

using namespace std;
using namespace nnp;

void SetupAnalysis::writeSymmetryFunctionShape(size_t       numPoints,
                                               string const fileNameFormat)
{
    log << "\n";
    log << "*** ANALYSIS: SYMMETRY FUNCTION SHAPE ***"
           "**************************************\n";
    log << "\n";

    log << strpr("Symmetry function file name format: %s\n",
                 fileNameFormat.c_str());
    log << strpr("Number of points                  : %zu\n", numPoints);
    double dr = getMaxCutoffRadius() / numPoints;
    log << strpr("Distance step                     : %f\n", dr);
    double da = M_PI / numPoints;
    log << strpr("Angle step (radians)              : %f\n", da);

    for (size_t i = 0; i < numElements; ++i)
    {
        Element const& e = elements.at(i);
        for (size_t j = 0; j < e.numSymmetryFunctions(); ++j)
        {
            SymmetryFunction const& s = e.getSymmetryFunction(j);
            ofstream file;
            file.open(strpr(fileNameFormat.c_str(),
                            e.getAtomicNumber(),
                            j + 1).c_str());
            vector<string> info = e.infoSymmetryFunction(s.getIndex());
            for (vector<string>::const_iterator it = info.begin();
                 it != info.end(); ++it)
            {
                file << strpr("#SFINFO %s\n", it->c_str());
            }

            // File header.
            vector<string> title;
            vector<string> colName;
            vector<string> colInfo;
            vector<size_t> colSize;
            title.push_back("Symmetry function shape.");
            colSize.push_back(16);
            colName.push_back("distance");
            colInfo.push_back("Distance from center.");
            colSize.push_back(16);
            colName.push_back("sf_radial");
            colInfo.push_back("Radial part of symmetry function at given "
                              "distance.");
            colSize.push_back(16);
            colName.push_back("angle");
            colInfo.push_back("Angle in degrees.");
            colSize.push_back(16);
            colName.push_back("sf_angular");
            colInfo.push_back("Angular part of symmetry function at given "
                              "angle.");
            appendLinesToFile(file,
                              createFileHeader(title,
                                               colSize,
                                               colName,
                                               colInfo));

            for (size_t k = 0; k < numPoints + 1; ++k)
            {
                file << strpr("%16.8E %16.8E %16.8E %16.8E\n",
                              k * dr,
                              s.calculateRadialPart(k * dr),
                              k * da,
                              s.calculateAngularPart(k * da));
            }
            file.close();
        }
    }

    log << "*****************************************"
           "**************************************\n";

    return;
}
