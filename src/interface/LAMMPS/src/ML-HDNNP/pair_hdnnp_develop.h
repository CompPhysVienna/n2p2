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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(hdnnp/develop,PairHDNNPDevelop)
// clang-format on
#else

#ifndef LMP_PAIR_HDNNP_DEVELOP_H
#define LMP_PAIR_HDNNP_DEVELOP_H

#include "pair_hdnnp.h"

namespace LAMMPS_NS {

class PairHDNNPDevelop : public PairHDNNP {

 public:
  PairHDNNPDevelop(class LAMMPS *);
  void compute(int, int) override;
  void init_style() override;
  double init_one(int i, int j) override;

 protected:
  /// Keeps track of the maximum cutoff radius that was used for the neighbor
  /// list.
  double maxCutoffRadiusNeighborList;

  void transferNeighborList(double const cutoffRadius);
  /// Update neighborlist if maxCutoffRadiusNeighborList has changed.
  void updateNeighborlistCutoff();

};

}    // namespace LAMMPS_NS

#endif
#endif
