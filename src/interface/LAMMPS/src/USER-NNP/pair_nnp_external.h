// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifdef PAIR_CLASS

PairStyle(nnp/external,PairNNPExternal)

#else

#ifndef LMP_PAIR_NNP_EXTERNAL_H
#define LMP_PAIR_NNP_EXTERNAL_H

#include "pair.h"
#include "ElementMap.h"
#include "Structure.h"

namespace LAMMPS_NS {

class PairNNPExternal : public Pair {

 public:

  PairNNPExternal(class LAMMPS *);
  virtual ~PairNNPExternal();
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

  double cflength;
  double cfenergy;
  double maxCutoffRadius;
  char* directory;
  char* elements;
  char* command;
  nnp::ElementMap em;
  nnp::Structure structure;
};

}

#endif
#endif
