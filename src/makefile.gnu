#!/bin/make -f

###############################################################################
# COMPILERS AND FLAGS
###############################################################################
PROJECT_CC=g++
PROJECT_MPICC=mpic++
PROJECT_CFLAGS=-O3 -march=native -std=c++98
PROJECT_CFLAGS_MPI=-Wno-long-long
PROJECT_DEBUG=-g -pedantic-errors -Wall -Wextra
PROJECT_AR=ar
PROJECT_ARFLAGS=-rcsv

###############################################################################
# COMPILE-TIME OPTIONS
###############################################################################

# Do not use symmetry function groups.
#PROJECT_OPTIONS+= -DNOSFGROUPS

# Do not use cutoff function cache.
#PROJECT_OPTIONS+= -DNOCFCACHE

# Build with dummy Stopwatch class.
#PROJECT_OPTIONS+= -DNOTIME

# Disable check for low number of neighbors.
#PROJECT_OPTIONS+= -DNONEIGHCHECK

# Compile without MPI support.
#PROJECT_OPTIONS+= -DNOMPI
