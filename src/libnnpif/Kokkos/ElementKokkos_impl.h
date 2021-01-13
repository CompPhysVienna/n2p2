// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
// Copyright (C) 2020 Saaketh Desai and Sam Reeve
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

#include <algorithm> // std::sort, std::min, std::max
#include <cstdlib>   // atoi
#include <iostream>  // std::cerr
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error

using namespace std;

namespace nnp
{

ElementCabana::ElementCabana( size_t const _index )
    : Element()
{
    index = _index;
    atomicEnergyOffset = 0.0;
    neuralNetwork = NULL;
}

ElementCabana::~ElementCabana() {}

template <class t_SF, class h_t_int>
void ElementCabana::addSymmetryFunction( string const &parameters,
                                         vector<string> elementStrings, int attype,
                                         t_SF SF, double convLength,
                                         h_t_int h_numSFperElem )
{
    vector<string> args = split( reduce( parameters ) );
    size_t type = (size_t)atoi( args.at( 1 ).c_str() );
    const char *estring;
    int el = 0;

    vector<string> splitLine = split( reduce( parameters ) );
    if ( type == 2 )
    {
        estring = splitLine.at( 0 ).c_str();
        for ( size_t i = 0; i < elementStrings.size(); ++i )
        {
            if ( strcmp( elementStrings[i].c_str(), estring ) == 0 )
                el = i;
        }
        SF( attype, h_numSFperElem( attype ), 0 ) = el;   // ec
        SF( attype, h_numSFperElem( attype ), 1 ) = type; // type

        estring = splitLine.at( 2 ).c_str();
        for ( size_t i = 0; i < elementStrings.size(); ++i )
        {
            if ( strcmp( elementStrings[i].c_str(), estring ) == 0 )
                el = i;
        }
        SF( attype, h_numSFperElem( attype ), 2 ) = el; // e1
        // set e2 to arbit high number for ease in creating groups
        SF( attype, h_numSFperElem( attype ), 3 ) = 100000; // e2

        SF( attype, h_numSFperElem( attype ), 4 ) =
            atof( splitLine.at( 3 ).c_str() ) /
            ( convLength * convLength ); // eta
        SF( attype, h_numSFperElem( attype ), 8 ) =
            atof( splitLine.at( 4 ).c_str() ) * convLength; // rs
        SF( attype, h_numSFperElem( attype ), 7 ) =
            atof( splitLine.at( 5 ).c_str() ) * convLength; // rc

        SF( attype, h_numSFperElem( attype ), 13 ) = h_numSFperElem( attype );
        h_numSFperElem( attype )++;
    }

    else if ( type == 3 )
    {
        if ( type != (size_t)atoi( splitLine.at( 1 ).c_str() ) )
            throw runtime_error( "ERROR: Incorrect symmetry function type.\n" );
        estring = splitLine.at( 0 ).c_str();
        for ( size_t i = 0; i < elementStrings.size(); ++i )
        {
            if ( strcmp( elementStrings[i].c_str(), estring ) == 0 )
                el = i;
        }
        SF( attype, h_numSFperElem( attype ), 0 ) = el;   // ec
        SF( attype, h_numSFperElem( attype ), 1 ) = type; // type

        estring = splitLine.at( 2 ).c_str();
        for ( size_t i = 0; i < elementStrings.size(); ++i )
        {
            if ( strcmp( elementStrings[i].c_str(), estring ) == 0 )
                el = i;
        }
        SF( attype, h_numSFperElem( attype ), 2 ) = el; // e1

        estring = splitLine.at( 3 ).c_str();
        for ( size_t i = 0; i < elementStrings.size(); ++i )
        {
            if ( strcmp( elementStrings[i].c_str(), estring ) == 0 )
                el = i;
        }

        SF( attype, h_numSFperElem( attype ), 3 ) = el; // e2
        SF( attype, h_numSFperElem( attype ), 4 ) =
            atof( splitLine.at( 4 ).c_str() ) /
            ( convLength * convLength ); // eta
        SF( attype, h_numSFperElem( attype ), 5 ) =
            atof( splitLine.at( 5 ).c_str() ); // lambda
        SF( attype, h_numSFperElem( attype ), 6 ) =
            atof( splitLine.at( 6 ).c_str() ); // zeta
        SF( attype, h_numSFperElem( attype ), 7 ) =
            atof( splitLine.at( 7 ).c_str() ) * convLength; // rc
        // Shift parameter is optional.
        if ( splitLine.size() > 8 )
            SF( attype, h_numSFperElem( attype ), 8 ) =
                atof( splitLine.at( 8 ).c_str() ) * convLength; // rs

        T_INT e1 = SF( attype, h_numSFperElem( attype ), 2 );
        T_INT e2 = SF( attype, h_numSFperElem( attype ), 3 );
        if ( e1 > e2 )
        {
            size_t tmp = e1;
            e1 = e2;
            e2 = tmp;
        }
        SF( attype, h_numSFperElem( attype ), 2 ) = e1;
        SF( attype, h_numSFperElem( attype ), 3 ) = e2;

        T_FLOAT zeta = SF( attype, h_numSFperElem( attype ), 6 );
        T_INT zetaInt = round( zeta );
        if ( fabs( zeta - zetaInt ) <= numeric_limits<double>::min() )
            SF( attype, h_numSFperElem( attype ), 9 ) = 1;
        else
            SF( attype, h_numSFperElem( attype ), 9 ) = 0;

        SF( attype, h_numSFperElem( attype ), 13 ) = h_numSFperElem( attype );
        h_numSFperElem( attype )++;
    }
    // TODO: Add this later
    else if ( type == 9 )
    {
    }
    else if ( type == 12 )
    {
    }
    else if ( type == 13 )
    {
    }
    else
    {
        throw runtime_error( "ERROR: Unknown symmetry function type.\n" );
    }

    return;
}

template <class t_SF, class h_t_int>
void ElementCabana::sortSymmetryFunctions( t_SF SF, h_t_int h_numSFperElem,
                                           int attype )
{
    int size = h_numSFperElem( attype );
    h_t_int h_SFsort( "SortSort", size );
    for ( int i = 0; i < size; ++i )
        h_SFsort( i ) = i;

    // naive insertion sort
    int i, j, tmp;
    for ( i = 1; i < size; ++i )
    {
        j = i;
        // explicit condition for sort
        while ( j > 0 &&
                compareSF( SF, attype, h_SFsort( j - 1 ), h_SFsort( j ) ) )
        {
            tmp = h_SFsort( j );
            h_SFsort( j ) = h_SFsort( j - 1 );
            h_SFsort( j - 1 ) = tmp;
            --j;
        }
    }

    int tmpindex;
    for ( int i = 0; i < size; ++i )
    {
        SF( attype, i, 13 ) = h_SFsort( i );
        tmpindex = SF( attype, i, 13 );
        SF( attype, tmpindex, 14 ) = i;
    }

    return;
}

template <class t_SF>
bool ElementCabana::compareSF( t_SF SF, int attype, int index1, int index2 )
{
    if ( SF( attype, index2, 0 ) < SF( attype, index1, 0 ) )
        return true; // ec
    else if ( SF( attype, index2, 0 ) > SF( attype, index1, 0 ) )
        return false;

    if ( SF( attype, index2, 1 ) < SF( attype, index1, 1 ) )
        return true; // type
    else if ( SF( attype, index2, 1 ) > SF( attype, index1, 1 ) )
        return false;

    if ( SF( attype, index2, 11 ) < SF( attype, index1, 11 ) )
        return true; // cutofftype
    else if ( SF( attype, index2, 11 ) > SF( attype, index1, 11 ) )
        return false;

    if ( SF( attype, index2, 12 ) < SF( attype, index1, 12 ) )
        return true; // cutoffalpha
    else if ( SF( attype, index2, 12 ) > SF( attype, index1, 12 ) )
        return false;

    if ( SF( attype, index2, 7 ) < SF( attype, index1, 7 ) )
        return true; // rc
    else if ( SF( attype, index2, 7 ) > SF( attype, index1, 7 ) )
        return false;

    if ( SF( attype, index2, 4 ) < SF( attype, index1, 4 ) )
        return true; // eta
    else if ( SF( attype, index2, 4 ) > SF( attype, index1, 4 ) )
        return false;

    if ( SF( attype, index2, 8 ) < SF( attype, index1, 8 ) )
        return true; // rs
    else if ( SF( attype, index2, 8 ) > SF( attype, index1, 8 ) )
        return false;

    if ( SF( attype, index2, 6 ) < SF( attype, index1, 6 ) )
        return true; // zeta
    else if ( SF( attype, index2, 6 ) > SF( attype, index1, 6 ) )
        return false;

    if ( SF( attype, index2, 5 ) < SF( attype, index1, 5 ) )
        return true; // lambda
    else if ( SF( attype, index2, 5 ) > SF( attype, index1, 5 ) )
        return false;

    if ( SF( attype, index2, 2 ) < SF( attype, index1, 2 ) )
        return true; // e1
    else if ( SF( attype, index2, 2 ) > SF( attype, index1, 2 ) )
        return false;

    if ( SF( attype, index2, 3 ) < SF( attype, index1, 3 ) )
        return true; // e2
    else if ( SF( attype, index2, 3 ) > SF( attype, index1, 3 ) )
        return false;

    else
        return false;
}

template <class t_SF, class h_t_int>
vector<string>
ElementCabana::infoSymmetryFunctionParameters( t_SF SF, int attype,
                                               h_t_int h_numSFperElem ) const
{
    vector<string> v;
    string pushstring = "";
    int index;
    float writestring;
    for ( int i = 0; i < h_numSFperElem( attype ); ++i )
    {
        index = SF( attype, i, 13 );
        // TODO: improve function
        for ( int j = 1; j < 12; ++j )
        {
            writestring = SF( attype, index, j );
            pushstring += to_string( writestring ) + " ";
        }
        pushstring += "\n";
    }
    v.push_back( pushstring );

    return v;
}

template <class t_SF, class t_SFscaling, class h_t_int>
vector<string>
ElementCabana::infoSymmetryFunctionScaling( ScalingType scalingType, t_SF SF,
                                      t_SFscaling SFscaling, int attype,
                                      h_t_int h_numSFperElem ) const
{
    vector<string> v;
    int index;
    for ( int k = 0; k < h_numSFperElem( attype ); ++k )
    {
        index = SF( attype, k, 13 );
        v.push_back( scalingLine( scalingType, SFscaling, attype, index ) );
    }
    return v;
}

template <class t_SF, class t_SFGmemberlist, class h_t_int>
void ElementCabana::setupSymmetryFunctionGroups( t_SF SF,
                                                 t_SFGmemberlist SFGmemberlist,
                                                 int attype, h_t_int h_numSFperElem,
                                                 h_t_int h_numSFGperElem,
                                                 int maxSFperElem )
{
    int num_group = h_numSFperElem.extent( 0 );
    h_t_int h_numGR( "RadialCounter", num_group );
    h_t_int h_numGA( "AngularCounter", num_group );
    int SFindex;
    for ( int k = 0; k < h_numSFperElem( attype ); ++k )
    {
        bool createNewGroup = true;
        SFindex = SF( attype, k, 13 );
        for ( int l = 0; l < h_numSFGperElem( attype ); ++l )
        {
            if ( ( SF( attype, SFindex, 0 ) ==
                   SF( attype, SFGmemberlist( attype, l, 0 ),
                       0 ) ) && // same ec
                 ( SF( attype, SFindex, 2 ) ==
                   SF( attype, SFGmemberlist( attype, l, 0 ),
                       2 ) ) && // same e1
                 ( SF( attype, SFindex, 3 ) ==
                   SF( attype, SFGmemberlist( attype, l, 0 ),
                       3 ) ) && // same e2
                 ( SF( attype, SFindex, 7 ) ==
                   SF( attype, SFGmemberlist( attype, l, 0 ),
                       7 ) ) && // same rc
                 ( SF( attype, SFindex, 11 ) ==
                   SF( attype, SFGmemberlist( attype, l, 0 ),
                       11 ) ) && // same cutoffType
                 ( SF( attype, SFindex, 12 ) ==
                   SF( attype, SFGmemberlist( attype, l, 0 ),
                       12 ) ) ) // same cutoffAlpha
            {
                createNewGroup = false;
                if ( SF( attype, SFindex, 1 ) == 2 )
                {
                    SFGmemberlist( attype, l, h_numGR( l ) ) = SFindex;
                    h_numGR( l )++;
                    SFGmemberlist( attype, l, maxSFperElem )++;
                    break;
                }

                else if ( SF( attype, SFindex, 1 ) == 3 )
                {
                    SFGmemberlist( attype, l, h_numGA( l ) ) = SFindex;
                    h_numGA( l )++;
                    SFGmemberlist( attype, l, maxSFperElem )++;
                    break;
                }
            }
        }

        if ( createNewGroup )
        {
            int l = h_numSFGperElem( attype );
            h_numSFGperElem( attype )++;
            if ( SF( attype, SFindex, 1 ) == 2 )
            {
                SFGmemberlist( attype, l, 0 ) = SFindex;
                if ( l >= (int)h_numGR.extent( 0 ) )
                    Kokkos::resize( h_numGR, l+1 );
                h_numGR( l ) = 1;
                SFGmemberlist( attype, l, maxSFperElem )++;
            }
            else if ( SF( attype, SFindex, 1 ) == 3 )
            {
                SFGmemberlist( attype, l, 0 ) = SFindex;
                if ( l >= (int)h_numGA.extent( 0 ) )
                    Kokkos::resize( h_numGA, l+1 );
                h_numGA( l ) = 1;
                SFGmemberlist( attype, l, maxSFperElem )++;
            }
        }
    }

    return;
}

template <class t_SF, class t_SFGmemberlist, class h_t_int>
vector<string>
ElementCabana::infoSymmetryFunctionGroups( t_SF SF, t_SFGmemberlist SFGmemberlist,
                                           int attype, h_t_int h_numSFGperElem ) const
{
    vector<string> v;
    string pushstring = "";
    for ( int groupIndex = 0; groupIndex < h_numSFGperElem( attype );
          ++groupIndex )
    {
        // TODO: improve function
        for ( int j = 0; j < 8; ++j )
            pushstring +=
                to_string(
                    SF( attype, SFGmemberlist( attype, groupIndex, 0 ), j ) ) +
                " ";
        pushstring += "\n";
    }
    v.push_back( pushstring );

    return v;
}

template <class t_SF, class h_t_int>
void ElementCabana::setCutoffFunction( CutoffFunction::CutoffType const cutoffType,
                                       double const cutoffAlpha, t_SF SF, int attype,
                                       h_t_int h_numSFperElem )
{
    for ( int k = 0; k < h_numSFperElem( attype ); ++k )
    {
        SF( attype, k, 10 ) = cutoffType;
        SF( attype, k, 11 ) = cutoffAlpha;
    }
    return;
}

template <class t_SF, class t_SFscaling, class h_t_int>
void ElementCabana::setScaling( ScalingType scalingType,
                                vector<string> const &statisticsLine, double Smin,
                                double Smax, t_SF SF, t_SFscaling SFscaling,
                                int attype, h_t_int h_numSFperElem ) const
{
    int index;
    for ( int k = 0; k < h_numSFperElem( attype ); ++k )
    {
        index = SF( attype, k, 13 );
        setScalingType( scalingType, statisticsLine.at( k ), Smin, Smax,
                        SFscaling, attype, index );
    }
    // TODO: groups
    // for (int k = 0; k < h_numSFperElem(attype); ++k)
    //    setScalingFactors(SF,attype,k);

    return;
}

template <class t_SF>
size_t ElementCabana::getMinNeighbors( int attype, t_SF SF, int nSF ) const
{
    // get max number of minNeighbors
    size_t global_minNeighbors = 0;
    size_t minNeighbors = 0;
    int SFtype;
    for ( int k = 0; k < nSF; ++k )
    {
        SFtype = SF( attype, k, 1 );
        if ( SFtype == 2 )
            minNeighbors = 1;
        else if ( SFtype == 3 )
            minNeighbors = 2;
        global_minNeighbors = max( minNeighbors, global_minNeighbors );
    }

    return global_minNeighbors;
}

template <class t_SF, class h_t_int>
double ElementCabana::getMinCutoffRadius( t_SF SF, int attype,
                                          h_t_int h_numSFperElem ) const
{
    double minCutoffRadius = numeric_limits<double>::max();

    for ( int k = 0; k < h_numSFperElem( attype ); ++k )
        minCutoffRadius = min( SF( attype, k, 7 ), minCutoffRadius );

    return minCutoffRadius;
}

template <class t_SF, class h_t_int>
double ElementCabana::getMaxCutoffRadius( t_SF SF, int attype,
                                          h_t_int h_numSFperElem ) const
{
    double maxCutoffRadius = 0.0;

    for ( int k = 0; k < h_numSFperElem( attype ); ++k )
        maxCutoffRadius = max( SF( attype, k, 7 ), maxCutoffRadius );

    return maxCutoffRadius;
}

// TODO:add functionality
/*
void ElementCabana::updateSymmetryFunctionStatistics
*/

}
