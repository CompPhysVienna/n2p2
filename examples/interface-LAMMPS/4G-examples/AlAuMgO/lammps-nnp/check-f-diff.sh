#!/bin/bash

../../../../../src/interface/lammps-nnp/src/lmp_mpi -in md.lmp 2>err.out
grep ") q" log.lammps | awk '{print $NF}' | sort -n > q.dat
paste q.dat ../nnp-predict/q.dat | \
awk 'function abs(x){return ((x < 0.0) ? -x : x)} {err=$1-$2; max=abs(err) < max ? max: abs(err); print $1-$2} END {print "MAX = ", max}'
echo "NIT = $(cat err.out | grep "eqeq-iter" | wc -l)"
