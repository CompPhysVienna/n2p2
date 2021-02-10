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

#include "SymFnc.h"
#include "utility.h"
#include <cstdlib>   // atof, atoi
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error, std::out_of_range

using namespace std;
using namespace nnp;

size_t const SymFnc::sfinfoWidth = 12;

SymFnc::PrintFormat const SymFnc::printFormat = initializePrintFormat();

SymFnc::PrintOrder const SymFnc::printOrder = initializePrintOrder();

vector<string> SymFnc::parameterInfo() const
{
    vector<string> v;
    string s;
    size_t w = sfinfoWidth;

    s = "lineNumber";
    v.push_back(strpr((pad(s, w) + "%zu"   ).c_str(), lineNumber + 1));
    s = "index";
    v.push_back(strpr((pad(s, w) + "%zu"   ).c_str(), index + 1));
    s = "type";
    v.push_back(strpr((pad(s, w) + "%zu"   ).c_str(), type));
    s = "ec";
    v.push_back(strpr((pad(s, w) + "%s"    ).c_str(), elementMap[ec].c_str()));
    s = "rc";
    v.push_back(strpr((pad(s, w) + "%14.8E").c_str(), rc / convLength));

    return v;
}

void SymFnc::setScalingType(ScalingType scalingType,
                            string      statisticsLine,
                            double      Smin,
                            double      Smax)
{
    this->scalingType = scalingType;

    vector<string> s = split(reduce(statisticsLine));
    if (((size_t)atoi(s.at(0).c_str()) != ec + 1) &&
        ((size_t)atoi(s.at(1).c_str()) != index + 1))
    {
        throw runtime_error("ERROR: Inconsistent scaling statistics.\n");
    }
    Gmin       = atof(s.at(2).c_str());
    Gmax       = atof(s.at(3).c_str());
    Gmean      = atof(s.at(4).c_str());
    // Older versions may not supply sigma.
    if (s.size() > 5)
    {
        Gsigma = atof(s.at(5).c_str());
    }
    this->Smin = Smin;
    this->Smax = Smax;

    if(scalingType == ST_NONE)
    {
        scalingFactor = 1.0;
    }
    else if (scalingType == ST_SCALE)
    {
        scalingFactor = (Smax - Smin) / (Gmax - Gmin);
    }
    else if (scalingType == ST_CENTER)
    {
        scalingFactor = 1.0;
    }
    else if (scalingType == ST_SCALECENTER)
    {
        scalingFactor = (Smax - Smin) / (Gmax - Gmin);
    }
    else if (scalingType == ST_SCALESIGMA)
    {
        scalingFactor = (Smax - Smin) / Gsigma;
    }

    return;
}

#ifndef NNP_NO_SF_CACHE
vector<string> SymFnc::getCacheIdentifiers() const
{
    return vector<string>();
}

void SymFnc::addCacheIndex(size_t element,
                           size_t cacheIndex,
                           string cacheIdentifier)
{
    // Check if provided cache identifier is identical to the originally
    // supplied one.
    vector<vector<string>> identifiersPerElement(elementMap.size());
    for (string id : getCacheIdentifiers())
    {
        size_t ne = atoi(split(id)[0].c_str());
        identifiersPerElement.at(ne).push_back(id);
    }
    size_t current = cacheIndices.at(element).size();
    if (identifiersPerElement.at(element).at(current) != cacheIdentifier)
    {
        throw runtime_error(strpr("ERROR: Cache identifiers do no match:\n"
                                  "%s\n"
                                  "    !=\n"
                                  "%s\n",
                                  identifiersPerElement.at(element)
                                  .at(current).c_str(),
                                  cacheIdentifier.c_str()));
    }
    cacheIndices.at(element).push_back(cacheIndex);

    return;
}
#endif

SymFnc::SymFnc(size_t type, ElementMap const& elementMap) :
    type         (type      ),
    elementMap   (elementMap),
    index        (0         ),
    ec           (0         ),
    minNeighbors (0         ),
    Smin         (0.0       ),
    Smax         (0.0       ),
    Gmin         (0.0       ),
    Gmax         (0.0       ),
    Gmean        (0.0       ),
    Gsigma       (0.0       ),
    rc           (0.0       ),
    scalingFactor(1.0       ),
    convLength   (1.0       ),
    scalingType  (ST_NONE   )
{
    // Add standard parameter IDs to set.
    parameters.insert("index");
    parameters.insert("ec");
    parameters.insert("type");
    parameters.insert("rc");
    parameters.insert("lineNumber");

    // Initialize per-element index vector, use max to indicate
    // "uninitialized" state.
    indexPerElement.resize(elementMap.size(), numeric_limits<size_t>::max());

#ifndef NNP_NO_SF_CACHE
    // Initialize cache indices vector.
    cacheIndices.resize(elementMap.size(), vector<size_t>());
#endif
}

double SymFnc::scale(double value) const
{
    if (scalingType == ST_NONE)
    {
        return value;
    }
    else if (scalingType == ST_SCALE)
    {
        return Smin + scalingFactor * (value - Gmin);
    }
    else if (scalingType == ST_CENTER)
    {
        return value - Gmean;
    }
    else if (scalingType == ST_SCALECENTER)
    {
        return Smin + scalingFactor * (value - Gmean);
    }
    else if (scalingType == ST_SCALESIGMA)
    {
        return Smin + scalingFactor * (value - Gmean);
    }
    else
    {
        return 0.0;
    }
}

double SymFnc::unscale(double value) const
{
    if (scalingType == ST_NONE)
    {
        return value;
    }
    else if (scalingType == ST_SCALE)
    {
        return (value - Smin) / scalingFactor + Gmin;
    }
    else if (scalingType == ST_CENTER)
    {
        return value + Gmean;
    }
    else if (scalingType == ST_SCALECENTER)
    {
        return (value - Smin) / scalingFactor + Gmean;
    }
    else if (scalingType == ST_SCALESIGMA)
    {
        return (value - Smin) / scalingFactor + Gmean;
    }
    else
    {
        return 0.0;
    }
}

string SymFnc::scalingLine() const
{
    return strpr("%4zu %9.2E %9.2E %9.2E %9.2E %9.2E %5.2f %5.2f %d\n",
                 index + 1,
                 Gmin,
                 Gmax,
                 Gmean,
                 Gsigma,
                 scalingFactor,
                 Smin,
                 Smax,
                 scalingType);
}

SymFnc::PrintFormat const SymFnc::initializePrintFormat()
{
    PrintFormat pf;

    pf["index"]       = make_pair("%4zu"  , string(4,  ' '));
    pf["ec"]          = make_pair("%2s"   , string(2,  ' '));
    pf["type"]        = make_pair("%2zu"  , string(2,  ' '));
    pf["subtype"]     = make_pair("%4s"   , string(4,  ' '));
    pf["e1"]          = make_pair("%2s"   , string(2,  ' '));
    pf["e2"]          = make_pair("%2s"   , string(2,  ' '));
    pf["eta"]         = make_pair("%9.3E" , string(9,  ' '));
    pf["rs/rl"]       = make_pair("%10.3E", string(10, ' '));
    pf["rc"]          = make_pair("%10.3E", string(10, ' '));
    pf["angleLeft"]   = make_pair("%6.1f" , string(6,  ' '));
    pf["angleRight"]  = make_pair("%6.1f" , string(6,  ' '));
    pf["lambda"]      = make_pair("%2.0f" , string(2,  ' '));
    pf["zeta"]        = make_pair("%4.1f" , string(4,  ' '));
    pf["alpha"]       = make_pair("%4.2f" , string(4,  ' '));
    pf["lineNumber"]  = make_pair("%5zu"  , string(5,  ' '));

    return pf;
}

SymFnc::PrintOrder const SymFnc::initializePrintOrder()
{
    vector<string> po;

    po.push_back("index"     );
    po.push_back("ec"        );
    po.push_back("type"      );
    po.push_back("subtype"   );
    po.push_back("e1"        );
    po.push_back("e2"        );
    po.push_back("eta"       );
    po.push_back("rs/rl"     );
    po.push_back("rc"        );
    po.push_back("angleLeft" );
    po.push_back("angleRight");
    po.push_back("lambda"    );
    po.push_back("zeta"      );
    po.push_back("alpha"     );
    po.push_back("lineNumber");

    return po;
}

string SymFnc::getPrintFormat() const
{
    string s;

    for (PrintOrder::const_iterator it = printOrder.begin();
         it != printOrder.end(); ++it)
    {
        // If parameter is present add format string.
        if (parameters.find(*it) != parameters.end())
        {
            s += safeFind(printFormat, (*it)).first + ' ';
        }
        // Else add just enough empty spaces.
        else 
        {
            s += safeFind(printFormat, (*it)).second + ' ';
        }
    }
    // Remove extra space at the end.
    if (s.size () > 0)  s.resize (s.size () - 1);
    s += '\n';

    return s;
}

