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
#include <cmath>

using namespace std;
using namespace nnp;

GradientDescent::GradientDescent(size_t const      sizeState,
                                 DescentType const type) :
    Updater(sizeState),
    eta     (0.0        ),
    beta1   (0.0        ),
    beta2   (0.0        ),
    epsilon (0.0        ),
    beta1t  (0.0        ),
    beta2t  (0.0        ),
    state   (NULL       ),
    error   (NULL       ),
    gradient(NULL       )
{
    if (!(type == DT_FIXED ||
          type == DT_ADAM))
    {
        throw runtime_error("ERROR: Unknown GradientDescent type.\n");
    }

    if (sizeState < 1)
    {
        throw runtime_error("ERROR: Wrong GradientDescent dimensions.\n");
    }

    this->type = type;

    if (type == DT_ADAM)
    {
        m.resize(sizeState, 0.0);
        v.resize(sizeState, 0.0);
    }
}

void GradientDescent::setState(double* state)
{
    this->state = state;

    return;
}

void GradientDescent::setError(double const* const error,
                               size_t const /* size */)
{
    this->error = error;

    return;
}

void GradientDescent::setJacobian(double const* const jacobian,
                                  size_t const        /* columns*/)
{
    this->gradient = jacobian;

    return;
}

void GradientDescent::update()
{
    if (type == DT_FIXED)
    {
        for (std::size_t i = 0; i < sizeState; ++i)
        {
            state[i] -= eta * (*error) * -gradient[i];
        }
    }
    else if (type == DT_ADAM)
    {
        for (std::size_t i = 0; i < sizeState; ++i)
        {
            double const g = (*error) * -gradient[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

            // Standard implementation
            // (Algorithm 1 in publication).
            //double const mhat = m[i] / (1.0 - beta1t);
            //double const vhat = v[i] / (1.0 - beta2t);
            //state[i] -= eta * mhat / (sqrt(vhat) + epsilon);
            
            // Faster (?) alternative
            // (see last paragraph in Section 2 of publication).
            // This is actually only marginally faster
            // (less statements, but two sqrt() calls)!
            eta = eta0 * sqrt(1 - beta2t) / (1 - beta1t); 
            state[i] -= eta * m[i] / (sqrt(v[i]) + epsilon);
        }

        // Update betas.
        beta1t *= beta1;
        beta2t *= beta2;
    }

    return;
}

void GradientDescent::setParametersFixed(double const eta)
{
    this->eta = eta;

    return;
}

void GradientDescent::setParametersAdam(double const eta,
                                        double const beta1,
                                        double const beta2,
                                        double const epsilon)
{
    this->eta     = eta;
    this->beta1   = beta1;
    this->beta2   = beta2;
    this->epsilon = epsilon;

    eta0   = eta;
    beta1t = beta1;
    beta2t = beta2;

    return;
}

string GradientDescent::status(size_t epoch) const
{
    string s = strpr("%10zu %16.8E", epoch, eta);

    if (type == DT_ADAM)
    {
        double meanm = 0.0;
        double meanv = 0.0;
        for (std::size_t i = 0; i < sizeState; ++i)
        {
            meanm += abs(m[i]);
            meanv += abs(v[i]);
        }
        meanm /= sizeState;
        meanv /= sizeState;
        s += strpr(" %16.8E %16.8E %16.8E %16.8E",
                   beta1t, beta2t, meanm, meanv);
    }
    s += '\n';

    return s;
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
    if (type == DT_ADAM)
    {
        colSize.push_back(16);
        colName.push_back("beta1t");
        colInfo.push_back("Decay rate 1 to the power of t.");
        colSize.push_back(16);
        colName.push_back("beta2t");
        colInfo.push_back("Decay rate 2 to the power of t.");
        colSize.push_back(16);
        colName.push_back("mag_m");
        colInfo.push_back("Mean of absolute first momentum (m).");
        colSize.push_back(16);
        colName.push_back("mag_v");
        colInfo.push_back("Mean of absolute second momentum (v).");
    }
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
    else if (type == DT_ADAM)
    {
        v.push_back(strpr("GradientDescentType::DT_ADAM (%d)\n", type));
        v.push_back(strpr("sizeState       = %zu\n", sizeState));
        v.push_back(strpr("eta             = %12.4E\n", eta));
        v.push_back(strpr("beta1           = %12.4E\n", beta1));
        v.push_back(strpr("beta2           = %12.4E\n", beta2));
        v.push_back(strpr("epsilon         = %12.4E\n", epsilon));
    }

    return v;
}
