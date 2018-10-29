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

#include "GradientDescent.h"
#include "utility.h"
#include <cstddef>

using namespace std;
using namespace nnp;

GradientDescent::GradientDescent(DescentType const type,
                                 size_t const      sizeState) :
    Updater(),
    sizeState  (0          ),
    eta        (0.0        ),
    state      (NULL       ),
    error      (NULL       ),
    derivatives(NULL       )
{
    if (!(type == DT_FIXED))
    {
        throw runtime_error("ERROR: Unknown GradientDescent type.\n");
    }

    if (sizeState < 1)
    {
        throw runtime_error("ERROR: Wrong GradientDescent dimensions.\n");
    }

    this->type = type;
    this->sizeState = sizeState;
}

void GradientDescent::setState(double* state)
{
    this->state = state;

    return;
}

void GradientDescent::setError(double const* const error)
{
    this->error = error;

    return;
}

void GradientDescent::setDerivativeMatrix(double const* const derivatives)
{
    this->derivatives = derivatives;

    return;
}

void GradientDescent::update()
{
    for (std::size_t i = 0; i < sizeState; ++i)
    {
        state[i] -= eta * (*error) * -derivatives[i];
    }

    return;
}

void GradientDescent::setParametersFixed(double const eta)
{
    this->eta = eta;

    return;
}

string GradientDescent::status(size_t epoch) const
{
    return strpr("%10zu %16.8E\n", epoch, eta);
}

vector<string> GradientDescent::statusHeader() const
{
    vector<string> header;

    vector<string> title;
    vector<string> colName;
    vector<string> colInfo;
    vector<size_t> colSize;
    title.push_back("Gradient descent status report.");
    colSize.push_back(10);
    colName.push_back("epoch");
    colInfo.push_back("Training epoch.");
    colSize.push_back(16);
    colName.push_back("eta");
    colInfo.push_back("Step size.");
    header = createFileHeader(title, colSize, colName, colInfo);

    return header;
}

vector<string> GradientDescent::info() const
{
    vector<string> v;

    if (type == DT_FIXED)
    {
        v.push_back(strpr("GradientDescentType::DT_FIXED (%d)\n", type));
        v.push_back(strpr("sizeState       = %zu\n", sizeState));
        v.push_back(strpr("eta             = %12.4E\n", eta));
    }

    return v;
}
