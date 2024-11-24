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

#include <algorithm> // std::min, std::max
#include <cstdlib>   // atoi, atof
#include <fstream>   // std::ifstream
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <string>

using namespace std;

namespace nnp
{

template <class t_device>
template <class t_slice_x, class t_slice_type, class t_slice_G,
          class t_neigh_list, class t_neigh_parallel, class t_angle_parallel>
void ModeCabana<t_device>::calculateSymmetryFunctionGroups(
    t_slice_x x, t_slice_type type, t_slice_G G, t_neigh_list neigh_list,
    int N_local, t_neigh_parallel neigh_op_tag, t_angle_parallel angle_op_tag )
{
    Cabana::deep_copy( G, 0.0 );

    Kokkos::RangePolicy<exe_space> policy( 0, N_local );

    // Create local copies for lambda
    auto numSFGperElem_ = numSFGperElem;
    auto SFGmemberlist_ = d_SFGmemberlist;
    auto SF_ = d_SF;
    auto SFscaling_ = d_SFscaling;
    auto maxSFperElem_ = maxSFperElem;
    auto convLength_ = convLength;
    auto cutoffType_ = cutoffType;
    auto cutoffAlpha_ = cutoffAlpha;

    auto calc_radial_symm_op = KOKKOS_LAMBDA( const int i, const int j )
    {
        double pfcij = 0.0;
        double pdfcij = 0.0;
        double eta, rs;
        size_t nej;
        int memberindex, globalIndex;
        double rij, r2ij;
        T_F_FLOAT dxij, dyij, dzij;

        int attype = type( i );
        for ( int groupIndex = 0; groupIndex < numSFGperElem_( attype );
              ++groupIndex )
        {
            if ( SF_( attype, SFGmemberlist_( attype, groupIndex, 0 ), 1 ) ==
                 2 )
            {
                size_t memberindex0 = SFGmemberlist_( attype, groupIndex, 0 );
                size_t e1 = SF_( attype, memberindex0, 2 );
                double rc = SF_( attype, memberindex0, 7 );
                size_t size =
                    SFGmemberlist_( attype, groupIndex, maxSFperElem_ );

                nej = type( j );
                dxij = ( x( i, 0 ) - x( j, 0 ) ) * CFLENGTH * convLength_;
                dyij = ( x( i, 1 ) - x( j, 1 ) ) * CFLENGTH * convLength_;
                dzij = ( x( i, 2 ) - x( j, 2 ) ) * CFLENGTH * convLength_;
                r2ij = dxij * dxij + dyij * dyij + dzij * dzij;
                rij = sqrt( r2ij );
                if ( e1 == nej && rij < rc )
                {
                    this->compute_cutoff( cutoffType_, cutoffAlpha_, pfcij, pdfcij,
                                          rij, rc, false );
                    for ( size_t k = 0; k < size; ++k )
                    {
                        memberindex = SFGmemberlist_( attype, groupIndex, k );
                        globalIndex = SF_( attype, memberindex, 14 );
                        eta = SF_( attype, memberindex, 4 );
                        rs = SF_( attype, memberindex, 8 );
                        G( i, globalIndex ) +=
                            exp( -eta * ( rij - rs ) * ( rij - rs ) ) * pfcij;
                    }
                }
            }
        }
    };
    Cabana::neighbor_parallel_for(
        policy, calc_radial_symm_op, neigh_list, Cabana::FirstNeighborsTag(),
        neigh_op_tag, "Mode::calculateRadialSymmetryFunctionGroups" );
    Kokkos::fence();

    auto calc_angular_symm_op =
        KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        double pfcij = 0.0, pdfcij = 0.0;
        double pfcik = 0.0, pdfcik = 0.0;
        double pfcjk = 0.0, pdfcjk = 0.0;
        size_t nej, nek;
        int memberindex, globalIndex;
        double rij, r2ij, rik, r2ik, rjk, r2jk;
        T_F_FLOAT dxij, dyij, dzij, dxik, dyik, dzik, dxjk, dyjk, dzjk;
        double eta, rs, lambda, zeta;

        int attype = type( i );
        for ( int groupIndex = 0; groupIndex < numSFGperElem_( attype );
              ++groupIndex )
        {
            if ( SF_( attype, SFGmemberlist_( attype, groupIndex, 0 ), 1 ) ==
                 3 )
            {
                size_t memberindex0 = SFGmemberlist_( attype, groupIndex, 0 );
                size_t e1 = SF_( attype, memberindex0, 2 );
                size_t e2 = SF_( attype, memberindex0, 3 );
                double rc = SF_( attype, memberindex0, 7 );
                size_t size =
                    SFGmemberlist_( attype, groupIndex, maxSFperElem_ );

                nej = type( j );
                dxij = ( x( i, 0 ) - x( j, 0 ) ) * CFLENGTH * convLength_;
                dyij = ( x( i, 1 ) - x( j, 1 ) ) * CFLENGTH * convLength_;
                dzij = ( x( i, 2 ) - x( j, 2 ) ) * CFLENGTH * convLength_;
                r2ij = dxij * dxij + dyij * dyij + dzij * dzij;
                rij = sqrt( r2ij );
                if ( ( e1 == nej || e2 == nej ) && rij < rc )
                {
                    // Calculate cutoff function and derivative.
                    this->compute_cutoff( cutoffType_, cutoffAlpha_, pfcij, pdfcij,
                                          rij, rc, false );

                    nek = type( k );

                    if ( ( e1 == nej && e2 == nek ) ||
                         ( e2 == nej && e1 == nek ) )
                    {
                        dxik =
                            ( x( i, 0 ) - x( k, 0 ) ) * CFLENGTH * convLength_;
                        dyik =
                            ( x( i, 1 ) - x( k, 1 ) ) * CFLENGTH * convLength_;
                        dzik =
                            ( x( i, 2 ) - x( k, 2 ) ) * CFLENGTH * convLength_;
                        r2ik = dxik * dxik + dyik * dyik + dzik * dzik;
                        rik = sqrt( r2ik );
                        if ( rik < rc )
                        {
                            dxjk = dxik - dxij;
                            dyjk = dyik - dyij;
                            dzjk = dzik - dzij;
                            r2jk = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
                            if ( r2jk < rc * rc )
                            {
                                // Energy calculation.
                                this->compute_cutoff( cutoffType_, cutoffAlpha_,
                                                      pfcik, pdfcik, rik, rc, false );

                                rjk = sqrt( r2jk );
                                this->compute_cutoff( cutoffType_, cutoffAlpha_,
                                                      pfcjk, pdfcjk, rjk, rc, false );

                                double const rinvijik = 1.0 / rij / rik;
                                double const costijk =
                                    ( dxij * dxik + dyij * dyik +
                                      dzij * dzik ) *
                                    rinvijik;
                                double vexp = 0.0, rijs = 0.0, riks = 0.0,
                                       rjks = 0.0;
                                for ( size_t l = 0; l < size; ++l )
                                {
                                    globalIndex =
                                        SF_( attype,
                                              SFGmemberlist_( attype,
                                                               groupIndex, l ),
                                              14 );
                                    memberindex = SFGmemberlist_(
                                        attype, groupIndex, l );
                                    eta = SF_( attype, memberindex, 4 );
                                    lambda = SF_( attype, memberindex, 5 );
                                    zeta = SF_( attype, memberindex, 6 );
                                    rs = SF_( attype, memberindex, 8 );
                                    if ( rs > 0.0 )
                                    {
                                        rijs = rij - rs;
                                        riks = rik - rs;
                                        rjks = rjk - rs;
                                        vexp = exp( -eta * ( rijs * rijs +
                                                             riks * riks +
                                                             rjks * rjks ) );
                                    }
                                    else
                                        vexp = exp( -eta *
                                                    ( r2ij + r2ik + r2jk ) );
                                    double const plambda =
                                        1.0 + lambda * costijk;
                                    double fg = vexp;
                                    if ( plambda <= 0.0 )
                                        fg = 0.0;
                                    else
                                        fg *= pow( plambda, ( zeta - 1.0 ) );
                                    G( i, globalIndex ) +=
                                        fg * plambda * pfcij * pfcik * pfcjk;
                                } // l
                            }     // rjk <= rc
                        }         // rik <= rc
                    }             // elem
                }                 // rij <= rc
            }
        }
    };
    Cabana::neighbor_parallel_for(
        policy, calc_angular_symm_op, neigh_list, Cabana::SecondNeighborsTag(),
        angle_op_tag, "Mode::calculateAngularSymmetryFunctionGroups" );
    Kokkos::fence();

    auto scale_symm_op = KOKKOS_LAMBDA( const int i )
    {
        int attype = type( i );

        int memberindex0;
        int memberindex, globalIndex;
        double raw_value = 0.0;
        for ( int groupIndex = 0; groupIndex < numSFGperElem_( attype );
              ++groupIndex )
        {
            memberindex0 = SFGmemberlist_( attype, groupIndex, 0 );

            size_t size = SFGmemberlist_( attype, groupIndex, maxSFperElem_ );
            for ( size_t k = 0; k < size; ++k )
            {
                globalIndex = SF_(
                    attype, SFGmemberlist_( attype, groupIndex, k ), 14 );
                memberindex = SFGmemberlist_( attype, groupIndex, k );

                if ( SF_( attype, memberindex0, 1 ) == 2 )
                    raw_value = G( i, globalIndex );
                else if ( SF_( attype, memberindex0, 1 ) == 3 )
                    raw_value =
                        G( i, globalIndex ) *
                        pow( 2, ( 1 - SF_( attype, memberindex, 6 ) ) );

                G( i, globalIndex ) =
                    this->scale( attype, raw_value, memberindex, SFscaling_ );
            }
        }
    };
    Kokkos::parallel_for( "Mode::scaleSymmetryFunctionGroups", policy,
                          scale_symm_op );
    Kokkos::fence();
}

template <class t_device>
template <class t_slice_x, class t_slice_f, class t_slice_type,
          class t_slice_dEdG, class t_neigh_list, class t_neigh_parallel,
          class t_angle_parallel>
void ModeCabana<t_device>::calculateForces( 
    t_slice_x x, t_slice_f f_a, t_slice_type type, t_slice_dEdG dEdG,
    t_neigh_list neigh_list, int N_local, t_neigh_parallel neigh_op_tag,
    t_angle_parallel angle_op_tag )
{
    double convForce_ = convLength / convEnergy;

    Kokkos::RangePolicy<exe_space> policy( 0, N_local );

    // Create local copies for lambda
    auto numSFGperElem_ = numSFGperElem;
    auto SFGmemberlist_ = d_SFGmemberlist;
    auto SF_ = d_SF;
    auto SFscaling_ = d_SFscaling;
    auto maxSFperElem_ = maxSFperElem;
    auto convLength_ = convLength;
    auto cutoffType_ = cutoffType;
    auto cutoffAlpha_ = cutoffAlpha;

    auto calc_radial_force_op = KOKKOS_LAMBDA( const int i, const int j )
    {
        double pfcij = 0.0;
        double pdfcij = 0.0;
        double rij, r2ij;
        T_F_FLOAT dxij, dyij, dzij;
        double eta, rs;
        int memberindex, globalIndex;

        int attype = type( i );

        for ( int groupIndex = 0; groupIndex < numSFGperElem_( attype );
              ++groupIndex )
        {
            if ( SF_( attype, SFGmemberlist_( attype, groupIndex, 0 ), 1 ) ==
                 2 )
            {
                size_t memberindex0 = SFGmemberlist_( attype, groupIndex, 0 );
                size_t e1 = SF_( attype, memberindex0, 2 );
                double rc = SF_( attype, memberindex0, 7 );
                size_t size =
                    SFGmemberlist_( attype, groupIndex, maxSFperElem_ );

                size_t nej = type( j );
                dxij = ( x( i, 0 ) - x( j, 0 ) ) * CFLENGTH * convLength_;
                dyij = ( x( i, 1 ) - x( j, 1 ) ) * CFLENGTH * convLength_;
                dzij = ( x( i, 2 ) - x( j, 2 ) ) * CFLENGTH * convLength_;
                r2ij = dxij * dxij + dyij * dyij + dzij * dzij;
                rij = sqrt( r2ij );
                if ( e1 == nej && rij < rc )
                {
                    // Energy calculation.
                    // Calculate cutoff function and derivative.
                    this->compute_cutoff( cutoffType_, cutoffAlpha_, pfcij, pdfcij,
                                          rij, rc, true );
                    for ( size_t k = 0; k < size; ++k )
                    {
                        globalIndex = SF_(
                            attype, SFGmemberlist_( attype, groupIndex, k ),
                            14 );
                        memberindex = SFGmemberlist_( attype, groupIndex, k );
                        eta = SF_( attype, memberindex, 4 );
                        rs = SF_( attype, memberindex, 8 );
                        double pexp = exp( -eta * ( rij - rs ) * ( rij - rs ) );
                        // Force calculation.
                        double const p1 =
                            SFscaling_( attype, memberindex, 6 ) *
                            ( pdfcij - 2.0 * eta * ( rij - rs ) * pfcij ) *
                            pexp / rij;
                        f_a( i, 0 ) -= ( dEdG( i, globalIndex ) *
                                         ( p1 * dxij ) * CFFORCE * convForce_ );
                        f_a( i, 1 ) -= ( dEdG( i, globalIndex ) *
                                         ( p1 * dyij ) * CFFORCE * convForce_ );
                        f_a( i, 2 ) -= ( dEdG( i, globalIndex ) *
                                         ( p1 * dzij ) * CFFORCE * convForce_ );

                        f_a( j, 0 ) += ( dEdG( i, globalIndex ) *
                                         ( p1 * dxij ) * CFFORCE * convForce_ );
                        f_a( j, 1 ) += ( dEdG( i, globalIndex ) *
                                         ( p1 * dyij ) * CFFORCE * convForce_ );
                        f_a( j, 2 ) += ( dEdG( i, globalIndex ) *
                                         ( p1 * dzij ) * CFFORCE * convForce_ );
                    }
                }
            }
        }
    };
    Cabana::neighbor_parallel_for( policy, calc_radial_force_op, neigh_list,
                                   Cabana::FirstNeighborsTag(), neigh_op_tag,
                                   "Mode::calculateRadialForces" );
    Kokkos::fence();

    auto calc_angular_force_op =
        KOKKOS_LAMBDA( const int i, const int j, const int k )
    {
        double pfcij = 0.0;
        double pdfcij = 0.0;
        double pfcik, pdfcik, pfcjk, pdfcjk;
        size_t nej, nek;
        double rij, r2ij, rik, r2ik, rjk, r2jk;
        T_F_FLOAT dxij, dyij, dzij, dxik, dyik, dzik, dxjk, dyjk, dzjk;
        double eta, rs, lambda, zeta;
        int memberindex, globalIndex;

        int attype = type( i );
        for ( int groupIndex = 0; groupIndex < numSFGperElem_( attype );
              ++groupIndex )
        {
            if ( SF_( attype, SFGmemberlist_( attype, groupIndex, 0 ), 1 ) ==
                 3 )
            {
                size_t memberindex0 = SFGmemberlist_( attype, groupIndex, 0 );
                size_t e1 = SF_( attype, memberindex0, 2 );
                size_t e2 = SF_( attype, memberindex0, 3 );
                double rc = SF_( attype, memberindex0, 7 );
                size_t size =
                    SFGmemberlist_( attype, groupIndex, maxSFperElem_ );

                nej = type( j );
                dxij = ( x( i, 0 ) - x( j, 0 ) ) * CFLENGTH * convLength_;
                dyij = ( x( i, 1 ) - x( j, 1 ) ) * CFLENGTH * convLength_;
                dzij = ( x( i, 2 ) - x( j, 2 ) ) * CFLENGTH * convLength_;
                r2ij = dxij * dxij + dyij * dyij + dzij * dzij;
                rij = sqrt( r2ij );
                if ( ( e1 == nej || e2 == nej ) && rij < rc )
                {
                    // Calculate cutoff function and derivative.
                    this->compute_cutoff( cutoffType_, cutoffAlpha_, pfcij, pdfcij,
                                          rij, rc, true );

                    nek = type( k );
                    if ( ( e1 == nej && e2 == nek ) ||
                         ( e2 == nej && e1 == nek ) )
                    {
                        dxik =
                            ( x( i, 0 ) - x( k, 0 ) ) * CFLENGTH * convLength_;
                        dyik =
                            ( x( i, 1 ) - x( k, 1 ) ) * CFLENGTH * convLength_;
                        dzik =
                            ( x( i, 2 ) - x( k, 2 ) ) * CFLENGTH * convLength_;
                        r2ik = dxik * dxik + dyik * dyik + dzik * dzik;
                        rik = sqrt( r2ik );
                        if ( rik < rc )
                        {
                            dxjk = dxik - dxij;
                            dyjk = dyik - dyij;
                            dzjk = dzik - dzij;
                            r2jk = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
                            if ( r2jk < rc * rc )
                            {
                                // Energy calculation.
                                this->compute_cutoff( cutoffType_, cutoffAlpha_,
                                                      pfcik, pdfcik, rik, rc, true );
                                rjk = sqrt( r2jk );

                                this->compute_cutoff( cutoffType_, cutoffAlpha_,
                                                      pfcjk, pdfcjk, rjk, rc, true );

                                double const rinvijik = 1.0 / rij / rik;
                                double const costijk =
                                    ( dxij * dxik + dyij * dyik +
                                      dzij * dzik ) *
                                    rinvijik;
                                double const pfc = pfcij * pfcik * pfcjk;
                                double const r2sum = r2ij + r2ik + r2jk;
                                double const pr1 = pfcik * pfcjk * pdfcij / rij;
                                double const pr2 = pfcij * pfcjk * pdfcik / rik;
                                double const pr3 = pfcij * pfcik * pdfcjk / rjk;
                                double vexp = 0.0, rijs = 0.0, riks = 0.0,
                                       rjks = 0.0;
                                for ( size_t l = 0; l < size; ++l )
                                {
                                    globalIndex =
                                        SF_( attype,
                                              SFGmemberlist_( attype,
                                                               groupIndex, l ),
                                              14 );
                                    memberindex = SFGmemberlist_(
                                        attype, groupIndex, l );
                                    rs = SF_( attype, memberindex, 8 );
                                    eta = SF_( attype, memberindex, 4 );
                                    lambda = SF_( attype, memberindex, 5 );
                                    zeta = SF_( attype, memberindex, 6 );
                                    if ( rs > 0.0 )
                                    {
                                        rijs = rij - rs;
                                        riks = rik - rs;
                                        rjks = rjk - rs;
                                        vexp = exp( -eta * ( rijs * rijs +
                                                             riks * riks +
                                                             rjks * rjks ) );
                                    }
                                    else
                                        vexp = exp( -eta * r2sum );

                                    double const plambda =
                                        1.0 + lambda * costijk;
                                    double fg = vexp;
                                    if ( plambda <= 0.0 )
                                        fg = 0.0;
                                    else
                                        fg *= pow( plambda, ( zeta - 1.0 ) );

                                    fg *= pow( 2, ( 1 - zeta ) ) *
                                          SFscaling_( attype, memberindex, 6 );
                                    double const pfczl = pfc * zeta * lambda;
                                    double factorDeriv =
                                        2.0 * eta / zeta / lambda;
                                    double const p2etapl =
                                        plambda * factorDeriv;
                                    double p1, p2, p3;
                                    if ( rs > 0.0 )
                                    {
                                        p1 = fg *
                                             ( pfczl *
                                                   ( rinvijik - costijk / r2ij -
                                                     p2etapl * rijs / rij ) +
                                               pr1 * plambda );
                                        p2 = fg *
                                             ( pfczl *
                                                   ( rinvijik - costijk / r2ik -
                                                     p2etapl * riks / rik ) +
                                               pr2 * plambda );
                                        p3 =
                                            fg *
                                            ( pfczl * ( rinvijik +
                                                        p2etapl * rjks / rjk ) -
                                              pr3 * plambda );
                                    }
                                    else
                                    {
                                        p1 = fg * ( pfczl * ( rinvijik -
                                                              costijk / r2ij -
                                                              p2etapl ) +
                                                    pr1 * plambda );
                                        p2 = fg * ( pfczl * ( rinvijik -
                                                              costijk / r2ik -
                                                              p2etapl ) +
                                                    pr2 * plambda );
                                        p3 = fg *
                                             ( pfczl * ( rinvijik + p2etapl ) -
                                               pr3 * plambda );
                                    }
                                    f_a( i, 0 ) -= ( dEdG( i, globalIndex ) *
                                                     ( p1 * dxij + p2 * dxik ) *
                                                     CFFORCE * convForce_ );
                                    f_a( i, 1 ) -= ( dEdG( i, globalIndex ) *
                                                     ( p1 * dyij + p2 * dyik ) *
                                                     CFFORCE * convForce_ );
                                    f_a( i, 2 ) -= ( dEdG( i, globalIndex ) *
                                                     ( p1 * dzij + p2 * dzik ) *
                                                     CFFORCE * convForce_ );

                                    f_a( j, 0 ) += ( dEdG( i, globalIndex ) *
                                                     ( p1 * dxij + p3 * dxjk ) *
                                                     CFFORCE * convForce_ );
                                    f_a( j, 1 ) += ( dEdG( i, globalIndex ) *
                                                     ( p1 * dyij + p3 * dyjk ) *
                                                     CFFORCE * convForce_ );
                                    f_a( j, 2 ) += ( dEdG( i, globalIndex ) *
                                                     ( p1 * dzij + p3 * dzjk ) *
                                                     CFFORCE * convForce_ );

                                    f_a( k, 0 ) += ( dEdG( i, globalIndex ) *
                                                     ( p2 * dxik - p3 * dxjk ) *
                                                     CFFORCE * convForce_ );
                                    f_a( k, 1 ) += ( dEdG( i, globalIndex ) *
                                                     ( p2 * dyik - p3 * dyjk ) *
                                                     CFFORCE * convForce_ );
                                    f_a( k, 2 ) += ( dEdG( i, globalIndex ) *
                                                     ( p2 * dzik - p3 * dzjk ) *
                                                     CFFORCE * convForce_ );
                                } // l
                            }     // rjk <= rc
                        }         // rik <= rc
                    }             // elem
                }                 // rij <= rc
            }
        }
    };
    Cabana::neighbor_parallel_for( policy, calc_angular_force_op, neigh_list,
                                   Cabana::SecondNeighborsTag(), angle_op_tag,
                                   "Mode::calculateAngularForces" );
    Kokkos::fence();

    return;
}

}
