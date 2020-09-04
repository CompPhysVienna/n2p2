// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifdef PAIR_CLASS

PairStyle(nnp,PairNNP)

#else

#ifndef LMP_PAIR_NNP_H
#define LMP_PAIR_NNP_H

#include "pair.h"
#include "InterfaceLammps.h"

namespace LAMMPS_NS {

class PairNNP : public Pair {

 public:

  PairNNP(class LAMMPS *);
  virtual ~PairNNP();
  virtual void compute(int, int);
  virtual void settings(int, char **);
  virtual void coeff(int, char **);
  virtual void init_style();
  virtual double init_one(int, int);
  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);

 protected:

  virtual void allocate();
  void transferNeighborList();
  void handleExtrapolationWarnings();

  bool showew;
  bool resetew;
  int showewsum;
  int maxew;
  long numExtrapolationWarningsTotal;
  long numExtrapolationWarningsSummary;
  double cflength;
  double cfenergy;
  double maxCutoffRadius;
  char* directory;
  char* emap;
  nnp::InterfaceLammps interface;
};

}

#endif
#endif
