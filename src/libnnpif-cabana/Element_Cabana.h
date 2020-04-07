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

#ifndef CBN_ELEMENT_H
#define CBN_ELEMENT_H

#include <nnp_cutoff.h>
#include <types_nnp.h>

#include <system.h>
#include <types.h>

#include <utility.h>

#include <cstddef> // size_t
#include <string>  // string
#include <vector>  // vector

using namespace std;
namespace nnpCbn
{

class SymmetryFunctionGroup;

/// Contains element-specific data.
class Element
{
  public:
    /** Constructor using index.
     */
    Element( size_t const index );
    /** Destructor.
     *
     * Necessary because of #symmetryFunctions vector of pointers.
     */
    ~Element();
    /** Set #atomicEnergyOffset.
     */
    void setAtomicEnergyOffset( double atomicEnergyOffset );
    /** Get #index.
     */
    size_t getIndex() const;
    /** Get #atomicNumber.
     */
    size_t getAtomicNumber() const;
    /** Get #atomicEnergyOffset.
     */
    double getAtomicEnergyOffset() const;
    /** Get #symbol.
     */
    string getSymbol() const;
    /** Add one symmetry function.
     *
     * @param[in] parameters String containing settings for symmetry function.
     * @param[in] lineNumber Line number of symmetry function in settings file.
     */
    template <class t_SF, class h_t_int>
    void addSymmetryFunction( string const &parameters,
                              vector<string> elementStrings, int attype,
                              t_SF SF, double convLength,
                              h_t_int h_numSFperElem );
    /** Change length unit for all symmetry functions.
     *
     * @param[in] convLength Length unit conversion factor.
     */
    void changeLengthUnitSymmetryFunctions( double convLength );
    /** Sort all symmetry function.
     */
    template <class t_SF, class h_t_int>
    void sortSymmetryFunctions( t_SF SF, h_t_int h_numSFperElem, int attype );
    /** Print symmetry function parameter value information.
     */
    template <class t_SF>
    bool compareSF( t_SF SF, int attype, int index1, int index2 );
    template <class t_SF, class h_t_int>
    vector<string>
    infoSymmetryFunctionParameters( t_SF SF, int attype,
                                    h_t_int h_numSFperElem ) const;
    template <class t_SF, class t_SFscaling, class h_t_int>
    vector<string> infoSymmetryFunctionScaling( ScalingType scalingType,
                                                t_SF SF, t_SFscaling SFscaling,
                                                int attype,
                                                h_t_int h_numSFperElem ) const;
    /** Set up symmetry function groups.
     */
    template <class t_SF, class t_SFGmemberlist, class h_t_int>
    void setupSymmetryFunctionGroups( t_SF SF, t_SFGmemberlist SFGmemberlist,
                                      int attype, h_t_int h_numSFperElem,
                                      h_t_int h_numSFGperElem,
                                      int maxSFperElem );
    /** Print symmetry function group info.
     */
    template <class t_SF, class t_SFGmemberlist, class h_t_int>
    vector<string>
    infoSymmetryFunctionGroups( t_SF SF, t_SFGmemberlist SFGmemberlist,
                                int attype, h_t_int h_numSFGperElem ) const;
    /** Set cutoff function for all symmetry functions.
     *
     * @param[in] cutoffType Type of cutoff function.
     * @param[in] cutoffAlpha Cutoff parameter for all functions.
     */
    template <class t_SF, class h_t_int>
    void setCutoffFunction( CutoffFunction::CutoffType const cutoffType,
                            double const cutoffAlpha, t_SF SF, int attype,
                            h_t_int h_numSFperElem );
    /** Set scaling of all symmetry functions.
     *
     * @param[in] scalingType Type of scaling, see
     *                        SymmetryFunction::ScalingType.
     * @param[in] statisticsLine Vector of strings containing statistics for
     *                           all symmetry functions.
     * @param[in] minS Minimum for scaling range.
     * @param[in] maxS Minimum for scaling range.
     */
    template <class t_SF, class t_SFscaling, class h_t_int>
    void setScaling( ScalingType scalingType,
                     vector<string> const &statisticsLine, double minS,
                     double maxS, t_SF SF, t_SFscaling SFscaling, int attype,
                     h_t_int h_numSFperElem ) const;
    /** Get number of symmetry functions.
     *
     * @return Number of symmetry functions.
     */
    template <class h_t_int>
    size_t numSymmetryFunctions( int attype, h_t_int h_numSFperElem ) const;
    /** Get maximum of required minimum number of neighbors for all symmetry
     * functions for this element.
     *
     * @return Minimum number of neighbors required.
     */
    template <class t_SF>
    size_t getMinNeighbors( int attype, t_SF SF, int nSF ) const;
    /** Get minimum cutoff radius of all symmetry functions.
     *
     * @return Minimum cutoff radius.
     */
    template <class t_SF, class h_t_int>
    double getMinCutoffRadius( t_SF SF, int attype,
                               h_t_int h_numSFperElem ) const;
    /** Get maximum cutoff radius of all symmetry functions.
     *
     * @return Maximum cutoff radius.
     */
    template <class t_SF, class h_t_int>
    double getMaxCutoffRadius( t_SF SF, int attype,
                               h_t_int h_numSFperElem ) const;
    /** Update symmetry function statistics.
     *
     * @param[in] atom Atom with symmetry function values.
     *
     * This function checks also for extrapolation warnings.
     */
    // void                     updateSymmetryFunctionStatistics(
    //                                                         System* s,
    //                                                         AoSoA_NNP_all
    //                                                         nnp_data,...);

    /** Get symmetry function instance.
     *
     * @param[in] index Symmetry function index.
     *
     * @return Symmetry function object.
     */
    // SymmetryFunction const&  getSymmetryFunction(size_t index) const;

    /// Symmetry function statistics.
    // SymmetryFunctionStatistics statistics;

    template <class t_SFscaling>
    inline void setScalingType( ScalingType scalingType, string statisticsLine,
                                double Smin, double Smax, t_SFscaling SFscaling,
                                int attype, int k ) const;
    template <class t_SFscaling>
    inline string scalingLine( ScalingType scalingType, t_SFscaling SFscaling,
                               int attype, int k ) const;
    template <class t_SFscaling>
    inline double unscale( int attype, double value, int k,
                           t_SFscaling SFscaling );

  private:
    /// Copy of element map.
    /// Global index of this element.
    size_t index;
    /// Atomic number of this element.
    size_t atomicNumber;
    /// Offset energy for every atom of this element.
    double atomicEnergyOffset;
    /// Element symbol.
    string symbol;
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline void Element::setAtomicEnergyOffset( double atomicEnergyOffset )
{
    this->atomicEnergyOffset = atomicEnergyOffset;

    return;
}

inline size_t Element::getIndex() const { return index; }

inline size_t Element::getAtomicNumber() const { return atomicNumber; }

inline double Element::getAtomicEnergyOffset() const
{
    return atomicEnergyOffset;
}

inline string Element::getSymbol() const { return symbol; }

template <class h_t_int>
inline size_t Element::numSymmetryFunctions( int attype,
                                             h_t_int h_numSFperElem ) const
{
    return h_numSFperElem( attype );
}

template <class t_SFscaling>
inline void Element::setScalingType( ScalingType scalingType,
                                     string statisticsLine, double Smin,
                                     double Smax, t_SFscaling SFscaling,
                                     int attype, int k ) const
{
    double Gmin, Gmax, Gmean, Gsigma = 0, scalingFactor = 0;
    vector<string> s = nnp::split( nnp::reduce( statisticsLine ) );

    Gmin = atof( s.at( 2 ).c_str() );
    Gmax = atof( s.at( 3 ).c_str() );
    Gmean = atof( s.at( 4 ).c_str() );
    SFscaling( attype, k, 0 ) = Gmin;
    SFscaling( attype, k, 1 ) = Gmax;
    SFscaling( attype, k, 2 ) = Gmean;

    // Older versions may not supply sigma.
    if ( s.size() > 5 )
        Gsigma = atof( s.at( 5 ).c_str() );
    SFscaling( attype, k, 3 ) = Gsigma;

    SFscaling( attype, k, 4 ) = Smin;
    SFscaling( attype, k, 5 ) = Smax;
    SFscaling( attype, k, 7 ) = scalingType;

    if ( scalingType == ST_NONE )
        scalingFactor = 1.0;
    else if ( scalingType == ST_SCALE )
        scalingFactor = ( Smax - Smin ) / ( Gmax - Gmin );
    else if ( scalingType == ST_CENTER )
        scalingFactor = 1.0;
    else if ( scalingType == ST_SCALECENTER )
        scalingFactor = ( Smax - Smin ) / ( Gmax - Gmin );
    else if ( scalingType == ST_SCALESIGMA )
        scalingFactor = ( Smax - Smin ) / Gsigma;
    SFscaling( attype, k, 6 ) = scalingFactor;

    return;
}

template <class t_SFscaling>
inline string Element::scalingLine( ScalingType scalingType,
                                    t_SFscaling SFscaling, int attype,
                                    int k ) const
{
    return nnp::strpr( "%4zu %9.2E %9.2E %9.2E %9.2E %9.2E %5.2f %5.2f %d\n",
                       k + 1, SFscaling( attype, k, 0 ),
                       SFscaling( attype, k, 1 ), SFscaling( attype, k, 2 ),
                       SFscaling( attype, k, 3 ), SFscaling( attype, k, 6 ),
                       SFscaling( attype, k, 4 ), SFscaling( attype, k, 5 ),
                       scalingType );
}

template <class t_SFscaling>
inline double Element::unscale( int attype, double value, int k,
                                t_SFscaling SFscaling )
{
    double scalingType = SFscaling( attype, k, 7 );
    double scalingFactor = SFscaling( attype, k, 6 );
    double Gmin = SFscaling( attype, k, 0 );
    // double Gmax = SFscaling(attype,k,1);
    double Gmean = SFscaling( attype, k, 2 );
    // double Gsigma = SFscaling(attype,k,3);
    double Smin = SFscaling( attype, k, 4 );
    // double Smax = SFscaling(attype,k,5);

    if ( scalingType == 0.0 )
    {
        return value;
    }
    else if ( scalingType == 1.0 )
    {
        return ( value - Smin ) / scalingFactor + Gmin;
    }
    else if ( scalingType == 2.0 )
    {
        return value + Gmean;
    }
    else if ( scalingType == 3.0 )
    {
        return ( value - Smin ) / scalingFactor + Gmean;
    }
    else if ( scalingType == 4.0 )
    {
        return ( value - Smin ) / scalingFactor + Gmean;
    }
    else
    {
        return 0.0;
    }
}

} // namespace nnpCbn

#include <nnp_element_impl.h>

#endif
