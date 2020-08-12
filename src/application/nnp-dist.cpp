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

#include "ElementMap.h"
#include "Log.h"
#include "Structure.h"
#include "utility.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace nnp;

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        cout << "USAGE: " << argv[0] << " <rcut> <nbin> <adf> <elem1 "
             << "<elem2 <elem3...>>>\n"
             << "       <rcut> .... Cutoff radius.\n"
             << "       <nbin> .... Number of histogram bins.\n"
             << "       <adf> ..... Calculate angle distribution (0/1).\n"
             << "       <elemN> ... Symbol for Nth element.\n"
             << "       Execute in directory with these NNP files present:\n"
             << "       - input.data (structure file)\n";
        return 1;
    }

    ofstream logFile;
    logFile.open("nnp-dist.log");
    Log log;
    log.registerStreamPointer(&logFile);

    log << "\n";
    log << "*** NNP-DIST ****************************"
           "**************************************\n";
    log << "\n";

    size_t numElements = argc - 4;
    log << strpr("Number of elements: %zu\n", numElements);
    string elements;
    elements += argv[4];
    for (size_t i = 5; i < numElements + 4; ++i)
    {
        elements += " ";
        elements += argv[i];
    }
    log << strpr("Element string    : %s\n", elements.c_str());
    double cutoffRadius = atof(argv[1]);
    log << strpr("Cutoff radius     : %f\n", cutoffRadius);
    size_t numBins = (size_t)atoi(argv[2]);
    log << strpr("Histogram bins    : %zu\n", numBins);
    bool calcAdf = (bool)atoi(argv[3]);
    log << strpr("Calculate ADF     : %d\n", calcAdf);

    size_t numRdf = numElements * (numElements + 1);
    numRdf /= 2;
    log << strpr("Number of RDFs    : %zu\n", numRdf); 
    size_t numAdf = 0;
    if (calcAdf)
    {
        numAdf = numElements * numElements * (numElements + 1);
        numAdf /= 2;
    }
    log << strpr("Number of ADFs    : %zu\n", numAdf); 
    log << "*****************************************"
           "**************************************\n";

    ElementMap elementMap;
    elementMap.registerElements(elements);

    double dr = cutoffRadius / numBins;
    map<pair<size_t, size_t>, vector<double>* > rhist;
    map<pair<size_t, size_t>, vector<double>* > rdf;
    double da = 180.0 / numBins;
    vector<map<pair<size_t, size_t>, vector<double>* > > ahist(numElements);
    vector<map<pair<size_t, size_t>, vector<double>* > > adf(numElements);

    for (size_t i = 0; i < numElements; ++i)
    {
        for (size_t j = i; j < numElements; ++j)
        {
            pair<size_t, size_t> e(i, j);
            rhist[e] = new vector<double>(numBins, 0);
            rdf  [e] = new vector<double>(numBins, 0.0);
            if (i != j)
            {
                rhist[make_pair(j, i)] = rhist[e];
                rdf  [make_pair(j, i)] = rdf  [e];
            }
        }
    }
    if (calcAdf)
    {
        for (size_t i = 0; i < numElements; ++i)
        {
            for (size_t j = 0; j < numElements; ++j)
            {
                for (size_t k = j; k < numElements; ++k)
                {
                    pair<size_t, size_t> e(j, k);
                    ahist.at(i)[e] = new vector<double>(numBins, 0);
                    adf.at(i)  [e] = new vector<double>(numBins, 0.0);
                    if (j != k)
                    {
                        ahist.at(i)[make_pair(k, j)] = ahist.at(i)[e];
                        adf.at(i)  [make_pair(k, j)] = adf.at(i)  [e];
                    }
                }
            }
        }
    }

    ifstream inputFile;
    inputFile.open("input.data");
    Structure structure;
    structure.setElementMap(elementMap);

    size_t countStructures = 0;
    size_t countPeriodicStructures = 0;
    vector<double> numberDensity(numElements, 0.0);
    while (inputFile.peek() != EOF)
    {
        structure.readFromFile(inputFile);
        structure.calculateNeighborList(cutoffRadius);
        for (vector<Atom>::const_iterator it = structure.atoms.begin();
             it != structure.atoms.end(); ++it)
        {
            for (size_t j = 0; j < it->numNeighbors; ++j)
            {
                size_t const ei = it->element;
                Atom::Neighbor const& nj = it->neighbors.at(j);
                size_t const ej = nj.element;
                double const rij = nj.d;
                pair<size_t, size_t> e(ei, ej);
                if (ei == ej)
                {
                    rhist[e]->at((size_t)floor(rij / dr)) += 1.0;
                }
                else
                {
                    rhist[e]->at((size_t)floor(rij / dr)) += 0.5;
                }
                if (calcAdf)
                {
                    for (size_t k = j + 1; k < it->numNeighbors; ++k)
                    {
                        Atom::Neighbor const& nk = it->neighbors.at(k);
                        size_t const ek = nk.element;
                        e = make_pair(ej, ek);
                        double theta = nj.dr * nk.dr / (nj.d * nk.d);
                        // Use first bin for 0 degree angle and catch problems
                        // with rounding errors.
                        if (theta >= 1.0)
                        {
                            ahist.at(ei)[e]->at(0) += 1.0;
                        }
                        // Use last bin for 180 degree angle and catch problems
                        // with rounding errors.
                        else if (theta <= -1.0)
                        {
                            ahist.at(ei)[e]->at(numBins - 1) += 1.0;
                        }
                        else
                        {
                            theta = acos(theta) * 180.0 / M_PI;
                            ahist.at(ei)[e]
                                ->at((size_t)floor(theta / da)) += 1.0;
                        }
                    }
                }
            }
        }
        countStructures++;
        if (structure.isPeriodic) countPeriodicStructures++;
        double const volume = structure.volume;
        for (size_t i = 0; i < numElements; ++i)
        {
            size_t const nni = structure.numAtomsPerElement.at(i);
            if (structure.isPeriodic)
            {
                numberDensity.at(i) += nni / volume;
            }
            for (size_t j = i; j < numElements; ++j)
            {
                size_t const nnj = structure.numAtomsPerElement.at(j);
                pair<size_t, size_t> e(i, j);
                for (size_t n = 0; n < numBins; ++n)
                {
                    double r = (n + 0.5) * dr;
                    if (structure.isPeriodic)
                    {
                        rhist[e]->at(n) *= volume / (nni * nnj);
                    }
                    rdf[e]->at(n) += rhist[e]->at(n)
                                   / (4.0 * M_PI * r * r * dr);
                }
                rhist[e]->clear();
                rhist[e]->resize(numBins, 0.0);
            }
        }
        if (calcAdf)
        {
            for (size_t i = 0; i < numElements; ++i)
            {
                for (size_t j = 0; j < numElements; ++j)
                {
                    for (size_t k = j; k < numElements; ++k)
                    {
                        pair<size_t, size_t> e(j, k);
                        double countAngles = 0.0;
                        for (size_t n = 0; n < numBins; ++n)
                        {
                            countAngles += ahist.at(i)[e]->at(n);
                        }
                        if (countAngles > 0)
                        {
                            for (size_t n = 0; n < numBins; ++n)
                            {
                                adf.at(i)[e]->at(n) += ahist.at(i)[e]->at(n)
                                                     / countAngles / da;
                            }
                        }
                        ahist.at(i)[e]->clear();
                        ahist.at(i)[e]->resize(numBins, 0.0);
                    }
                }
            }
        }
        log << strpr("Configuration %7zu: %7zu atoms\n",
                     countStructures,
                     structure.numAtoms);
        structure.reset();
    }
    log << "*****************************************"
           "**************************************\n";
    log << strpr("Number of          structures: %9zu\n", countStructures);
    log << strpr("Number of periodic structures: %9zu\n",
                 countPeriodicStructures);
    bool calcCn = false;
    if (countStructures == countPeriodicStructures) calcCn = true;
    if (calcCn)
    {
        vector<double>::const_iterator minnd = min_element(
                                   numberDensity.begin(), numberDensity.end());
        for (size_t i = 0; i < numElements; ++i)
        {
            log << strpr("Number density (ratio) of element %2s: %16.8E "
                         "(%.2f)\n", elementMap[i].c_str(),
                         numberDensity.at(i), numberDensity.at(i) / (*minnd));
        }
    }
    log << "*****************************************"
           "**************************************\n";


    ofstream outputFile;
    for (size_t i = 0; i < numElements; ++i)
    {
        for (size_t j = i; j < numElements; ++j)
        {
            pair<size_t, size_t> e(i, j);
            string fileName = strpr("rdf_%s_%s.out",
                                    elementMap[i].c_str(),
                                    elementMap[j].c_str());
            log << strpr("Writing RDF for element combination (%2s/%2s) "
                         "to file %s.\n",
                         elementMap[i].c_str(),
                         elementMap[j].c_str(),
                         fileName.c_str());
            outputFile.open(fileName.c_str());

            // File header.
            vector<string> title;
            vector<string> colName;
            vector<string> colInfo;
            vector<size_t> colSize;
            title.push_back(strpr("Radial distribution function for element "
                                  "combination %2s-%2s.",
                                   elementMap[i].c_str(),
                                   elementMap[j].c_str()));
            colSize.push_back(16);
            colName.push_back("dist_bin_l");
            colInfo.push_back("Distance, left bin limit.");
            colSize.push_back(16);
            colName.push_back("dist_bin_r");
            colInfo.push_back("Distance, right bin limit.");
            colSize.push_back(16);
            colName.push_back("rdf");
            colInfo.push_back("Radial distribution function, standard "
                              "normalization (-> 1 for r -> inf)");
            colSize.push_back(16);
            colName.push_back("rdf_max1");
            colInfo.push_back("Radial distribution function, maximum "
                              "normalized to 1.");
            if (calcCn)
            {
                colSize.push_back(16);
                colName.push_back("cn");
                colInfo.push_back("Coordination number.");
            }
            appendLinesToFile(outputFile,
                              createFileHeader(title,
                                               colSize,
                                               colName,
                                               colInfo));

            vector<double> cn(rdf[e]->size(), 0.0);
            double integral = 0.0;
            double const pre = 4.0 * M_PI * dr * 0.5
                             / countStructures / countStructures;
            double maxRdf = *max_element(rdf[e]->begin(), rdf[e]->end());
            for (size_t n = 0; n < numBins; ++n)
            {
                double const low = n * dr;
                double const center = (n + 0.5) * dr;
                double const high = (n + 1) * dr;
                outputFile << strpr("%16.8E %16.8E %16.8E %16.8E",
                                    low,
                                    high,
                                    rdf[e]->at(n) / countStructures,
                                    rdf[e]->at(n) / maxRdf);
                if (calcCn)
                {
                    if (n > 0)
                    {
                        integral += pre * center * center * numberDensity.at(j)
                                  * (rdf[e]->at(n-1) + rdf[e]->at(n));
                    }
                    outputFile << strpr(" %16.8E", integral);
                }
                outputFile << '\n';
            }
            outputFile.close();
            delete rdf  [e];
            delete rhist[e];
        }
    }
    if (calcAdf)
    {
        for (size_t i = 0; i < numElements; ++i)
        {
            for (size_t j = 0; j < numElements; ++j)
            {
                for (size_t k = j; k < numElements; ++k)
                {
                    pair<size_t, size_t> e(j, k);
                    string fileName = strpr("adf_%s_%s_%s.out",
                                            elementMap[i].c_str(),
                                            elementMap[j].c_str(),
                                            elementMap[k].c_str());
                    log << strpr("Writing ADF for element combination "
                                 "(%2s/%2s/%2s) to file %s.\n",
                                 elementMap[i].c_str(),
                                 elementMap[j].c_str(),
                                 elementMap[k].c_str(),
                                 fileName.c_str());
                    outputFile.open(fileName.c_str());

                    // File header.
                    vector<string> title;
                    vector<string> colName;
                    vector<string> colInfo;
                    vector<size_t> colSize;
                    title.push_back(strpr("Angular distribution function for "
                                          "element combination %2s-%2s-%2s.",
                                          elementMap[i].c_str(),
                                          elementMap[j].c_str(),
                                          elementMap[k].c_str()));
                    colSize.push_back(16);
                    colName.push_back("angle_bin_l");
                    colInfo.push_back("Angle [degree], left bin limit.");
                    colSize.push_back(16);
                    colName.push_back("angle_bin_r");
                    colInfo.push_back("Angle [degree], right bin limit.");
                    colSize.push_back(16);
                    colName.push_back("adf");
                    colInfo.push_back("Angular distribution function, "
                                      "probability normalization "
                                      "(integral = 1)");
                    colSize.push_back(16);
                    colName.push_back("adf_max1");
                    colInfo.push_back("Angular distribution function, maximum "
                                      "normalized to 1.");
                    appendLinesToFile(outputFile,
                                      createFileHeader(title,
                                                       colSize,
                                                       colName,
                                                       colInfo));

                    double maxAdf = *max_element(adf.at(i)[e]->begin(),
                                                 adf.at(i)[e]->end());
                    for (size_t n = 0; n < numBins; ++n)
                    {
                        double const low = n * da;
                        double const high = (n + 1) * da;
                        outputFile << strpr("%16.8E %16.8E %16.8E %16.8E\n",
                                            low,
                                            high,
                                            adf.at(i)[e]->at(n)
                                            / countStructures,
                                            adf.at(i)[e]->at(n) / maxAdf);
                    }
                    outputFile.close();
                    delete adf.at(i)  [e];
                    delete ahist.at(i)[e];
                }
            }
        }
    }

    log << "*****************************************"
           "**************************************\n";
    logFile.close();

    return 0;
}
