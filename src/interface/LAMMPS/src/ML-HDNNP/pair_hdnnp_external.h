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
PairStyle(hdnnp/external,PairHDNNPExternal);
// clang-format on
#else

#ifndef LMP_PAIR_HDNNP_EXTERNAL_H
#define LMP_PAIR_HDNNP_EXTERNAL_H

#include "pair.h"
#include "ElementMap.h"
#include "Structure.h"

namespace LAMMPS_NS {

class PairHDNNPExternal : public Pair {

 public:
  PairHDNNPExternal(class LAMMPS *);
  ~PairHDNNPExternal() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  virtual void allocate();

  double cflength;
  double cfenergy;
  double maxCutoffRadius;
  char* directory;
  char* elements;
  char* command;
  nnp::ElementMap em;
  nnp::Structure structure;
};

}    // namespace LAMMPS_NS

#endif
#endif
