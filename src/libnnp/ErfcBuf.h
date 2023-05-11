//
// Created by philipp on 5/23/22.
//

#ifndef N2P2_ERFCBUF_H
#define N2P2_ERFCBUF_H

#include "Atom.h"
#include <vector>


namespace nnp
{
    /// Helper class to store previously calculated values of erfc() that are
    /// needed during the charge equilibration.
    struct ErfcBuf
    {
        /** Resizes and resets the storage array to fit the current structure.
         *
         * @param[in] atoms Vector of atoms of the structure.
         * @param[in] valuesPerPair Number of values that are stored for each
         * neighbor pair (intended use is for erfc(rij...)).
         */
        void reset(std::vector<Atom> const& atoms, size_t const valuesPerPair);
        /** Either returns already stored value erfc(x) or calculates it if not
         *  yet stored.
         * @param[in] atomIndex Index of atom.
         * @param[in] neighIndex Index of neighbor of this atom.
         * @param[in] valIndex Index corresponds to the value that is requested,
         *                      see numValuesPerPair.
         * @param[in] x Corresponds to erfc(x). Only used when result is not
         *              stored yet.
         */
        double getf(size_t const atomIndex,
                    size_t const neighIndex,
                    size_t const valIndex,
                    double const x);

        /// 2d vector to store already calculated results. Elements set to -1
        /// indicate that this value has not been calculated yet.
        std::vector<std::vector<double>> f;
        /// Typically one needs erfc(a_i * rij), where the number of a_i's
        /// correspond to numValuesPerPair.
        size_t numValuesPerPair = 1;
    };

} // nnp

#endif //N2P2_ERFCBUF_H
