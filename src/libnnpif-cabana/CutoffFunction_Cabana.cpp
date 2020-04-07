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

#include <nnp_cutoff.h>

#include <cmath> // cos, sin, tanh, exp, pow
#include <iostream>
#include <limits> // std::numeric_limits
#include <stdexcept>

using namespace std;
using namespace nnpCbn;

double const CutoffFunction::PI = 4.0 * atan( 1.0 );
double const CutoffFunction::PI_2 = 2.0 * atan( 1.0 );
double const CutoffFunction::E = exp( 1.0 );
double const CutoffFunction::TANH_PRE = pow( ( E + 1 / E ) / ( E - 1 / E ), 3 );

CutoffFunction::CutoffFunction()
    : cutoffType( CT_HARD )
    , rc( 0.0 )
    , rcinv( 0.0 )
    , rci( 0.0 )
    , alpha( 0.0 )
    , iw( 0.0 )
{
}

CutoffFunction::CutoffFunction( double _rc )
{
    cutoffType = CT_TANHU;
    rc = _rc;
    rcinv = 1 / rc;
    rci = alpha = iw = 0.0;
}
