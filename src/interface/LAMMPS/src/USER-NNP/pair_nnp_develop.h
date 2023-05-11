// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifdef PAIR_CLASS

PairStyle(nnp/develop,PairNNPDevelop)

#else

#ifndef LMP_PAIR_NNP_DEVELOP_H
#define LMP_PAIR_NNP_DEVELOP_H

#include "pair_nnp.h"

namespace LAMMPS_NS {

class PairNNPDevelop : public PairNNP {

 public:

  PairNNPDevelop(class LAMMPS *);
  virtual ~PairNNPDevelop() {}
  virtual void compute(int, int);
  virtual void init_style();
  virtual double init_one(int i, int j);

 protected:
  /// Keeps track of the maximum cutoff radius that was used for the neighbor
  /// list.
  double maxCutoffRadiusNeighborList;

  /// Update neighborlist if maxCutoffRadiusNeighborList has changed.
  void updateNeighborlistCutoff();

};

}

#endif
#endif
