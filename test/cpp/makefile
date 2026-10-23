#!/bin/make -f

###############################################################################
# PROJECT PATHS
###############################################################################
PROJECT_DIR=../..
PROJECT_BIN=$(PROJECT_DIR)/bin
PROJECT_INCLUDE=$(PROJECT_DIR)/include
PROJECT_LIB=$(PROJECT_DIR)/lib


###############################################################################
# GENERAL SETTINGS
###############################################################################
# Default compiler, may be overridden by command line.
COMP=gnu # WARNING: Testing so far only works with "gnu" compiler.
# Get from main makefile in src directory:
#COMP=$(shell grep "^COMP=" ${PROJECT_DIR}/src/makefile | cut -d"=" -f2)

# Log level for boost tests, may be overriden by command line.
LOG_LEVEL=error

# Include global (project-wide) settings.
include $(PROJECT_DIR)/src/makefile.$(COMP)


###############################################################################
# PROJECT COMPILER AND FLAGS
###############################################################################
CC=$(PROJECT_CC)
MPICC=$(PROJECT_MPICC)
CFLAGS=$(PROJECT_CFLAGS)
OPTIONS+=$(PROJECT_OPTIONS)
DEBUG=$(PROJECT_DEBUG) $(PROJECT_TEST)

# Extra include paths for compiling.
INCLUDES=-I./ -I$(PROJECT_DIR)/src/libnnp -I$(PROJECT_DIR)/src/libnnptrain -I$(PROJECT_EIGEN) -I$(PROJECT_BOOST)/include

# Extra flags for linking.
# NOTE: libnnp.a and libnnptrain.a are wrapped in --start-group/--end-group
# because some libnnptrain object files (e.g. GradientDescent.o) reference
# symbols defined in libnnp.a (e.g. nnp::strpr, nnp::createFileHeader). GNU
# ld only makes one pass per static archive, so if a test translation unit
# does not itself reference any libnnp symbol directly (pulling libnnp.a in
# early), libnnp.a is already closed by the time libnnptrain.a's undefined
# refs surface, and linking fails. --start-group/--end-group makes ld retry
# the archives until all symbols resolve.
LDFLAGS=-Wl,--start-group $(PROJECT_LIB)/libnnp.a $(PROJECT_LIB)/libnnptrain.a -Wl,--end-group $(PROJECT_LDFLAGS_BLAS) -L$(PROJECT_BOOST)/lib -lboost_unit_test_framework -lboost_filesystem -lpthread


###############################################################################
# DEFINE SOURCES
###############################################################################
SRC=$(wildcard test_*.cpp)
SRC_NOIF=$(filter-out test_lammps_hdnnp.cpp, $(SRC))
EXE=$(SRC:%.cpp=%)
EXE_NOIF=$(SRC_NOIF:%.cpp=%)
# Groups for running the suite in parts (e.g. separate CI steps): the LAMMPS
# interface test, the application (tool) regression tests, and all others.
EXE_INTERFACE=$(filter test_lammps_hdnnp, $(EXE))
EXE_TOOLS=$(filter test_nnp-%, $(EXE))
EXE_UNIT=$(filter-out $(EXE_INTERFACE) $(EXE_TOOLS), $(EXE))
# Quick suite for development: everything except the interface test and the
# tests taking several minutes each. Run the full suite before pushing.
EXE_SLOW=$(filter test_nnp-train test_ForcesFiniteDifference test_Training_mpi, $(EXE))
EXE_QUICK=$(filter-out $(EXE_INTERFACE) $(EXE_SLOW), $(EXE))


###############################################################################
# RULES
###############################################################################
# Test targets are phony: each one is always rebuilt and run. Otherwise an
# existing test binary would be considered up to date, skipping the test
# after library changes or even after the test failed.
.PHONY: all no-interface unit tools interface quick coverage clean clean-coverage $(EXE)
.NOTPARALLEL:

all: $(EXE)

no-interface: $(EXE_NOIF)

unit: $(EXE_UNIT)

tools: $(EXE_TOOLS)

interface: $(EXE_INTERFACE)

quick: $(EXE_QUICK)

# test_Dataset and test_Training link Dataset.o/Training.o directly (call
# their methods in-process, unlike every other test_nnp-* regression test
# which spawns the real application binary via mpirun instead), pulling in
# real MPI symbol references that the plain compiler cannot resolve -- use
# the MPI compiler wrapper for these targets only. test_KalmanFilter needs it
# too: KalmanFilter.h includes mpi.h, which is not on the default include
# path everywhere (e.g. Ubuntu's OpenMPI, used in CI).
test_Dataset: CC=$(MPICC)
test_Training: CC=$(MPICC)
test_Training_mpi: CC=$(MPICC)
test_KalmanFilter: CC=$(MPICC)

# MPI launcher for tests that need several processes: test_Training_mpi is
# run once more on MPI_TEST_RANKS ranks after its single-rank run, and the
# nnp-comp2 "compare" regression test reads the launcher from the exported
# N2P2_TEST_MPIRUN. The launcher must accept "-np N". Where fewer slots than
# ranks are available, use e.g. MPIRUN="mpirun --oversubscribe" (OpenMPI).
MPIRUN=mpirun
MPI_TEST_RANKS=4
export N2P2_TEST_MPIRUN=$(MPIRUN)

test_Training_mpi: test_Training_mpi.cpp
	$(CC) $(CFLAGS) $(DEBUG) $(INCLUDES) $(OPTIONS) -o $@ $< $(LDFLAGS)
	./$@ --color_output=yes --log_level=$(LOG_LEVEL)
	$(MPIRUN) -np $(MPI_TEST_RANKS) ./$@ --color_output=yes --log_level=$(LOG_LEVEL)

# Static pattern rule, implicit rules are not applied to phony targets.
$(filter-out test_Training_mpi, $(EXE)): %: %.cpp
	$(CC) $(CFLAGS) $(DEBUG) $(INCLUDES) $(OPTIONS) -o $@ $< $(LDFLAGS)
	./$@ --color_output=yes --log_level=$(LOG_LEVEL)

coverage:
	lcov -c -i --no-external        -d $(PROJECT_DIR)/src/libnnp -d $(PROJECT_DIR)/src/libnnptrain -d $(PROJECT_DIR)/src/libnnpif -d $(PROJECT_DIR)/src/application -o tests.base
	lcov -c    --no-external        -d $(PROJECT_DIR)/src/libnnp -d $(PROJECT_DIR)/src/libnnptrain -d $(PROJECT_DIR)/src/libnnpif -d $(PROJECT_DIR)/src/application -o tests.run
	lcov -a tests.base -a tests.run -d $(PROJECT_DIR)/src/libnnp -d $(PROJECT_DIR)/src/libnnptrain -d $(PROJECT_DIR)/src/libnnpif -d $(PROJECT_DIR)/src/application -o tests.info
	genhtml -o htmlcov tests.info

clean: clean-coverage
	$(RM) *.o *.gcno $(EXE)

clean-coverage:
	$(RM) $(PROJECT_DIR)/src/libnnp/*.gcda
	$(RM) $(PROJECT_DIR)/src/libnnptrain/*.gcda
	$(RM) $(PROJECT_DIR)/src/libnnpif/LAMMPS/*.gcda
	$(RM) $(PROJECT_DIR)/src/application/*.gcda
	$(RM) *.gcda *.base *.run *.info
	$(RM) -r htmlcov

# Grep for "CI-CHECK" for code changes if you want to review output files from non-interactive CI run.
#test_nnp-train: test_nnp-train.cpp
#	$(CC) $(CFLAGS) $(DEBUG) $(INCLUDES) $(OPTIONS) -o $@ $< $(LDFLAGS)
#	-./$@ --color_output=yes --log_level=$(LOG_LEVEL)
#	cat nnp-train.log.0000.*
#	cat learning-curve.out.*
