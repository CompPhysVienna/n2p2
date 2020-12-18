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
INCLUDES=-I./ -I$(PROJECT_DIR)/src/libnnp -I$(PROJECT_DIR)/src/libnnptrain

# Extra flags for linking.
LDFLAGS=$(PROJECT_LIB)/libnnp.a $(PROJECT_LIB)/libnnptrain.a -lboost_system -lboost_unit_test_framework -lboost_filesystem -lpthread


###############################################################################
# DEFINE SOURCES
###############################################################################
SRC=$(wildcard test_*.cpp)
SRC_NOIF=$(filter-out test_lammps_nnp.cpp, $(SRC))
EXE=$(SRC:%.cpp=%)
EXE_NOIF=$(SRC_NOIF:%.cpp=%)


###############################################################################
# RULES
###############################################################################
.PHONY: all no-interface coverage clean clean-coverage

all: $(EXE)

no-interface: $(EXE_NOIF)

%: %.cpp
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
