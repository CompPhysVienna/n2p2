#!/bin/make -f

###############################################################################
# PROJECT
###############################################################################
PROJECT_DIR=../
PROJECT_BIN=${PROJECT_DIR}/bin
PROJECT_INCLUDE=${PROJECT_DIR}/include
PROJECT_LIB=${PROJECT_DIR}/lib
PROJECT_BUILD=${PROJECT_DIR}/build


###############################################################################
# GENERAL SETTINGS
###############################################################################
# Compiler (gnu/intel).
# Get from main makefile in src directory:
COMP=$(shell grep "^COMP=" ${PROJECT_DIR}/src/makefile | cut -d"=" -f2)
# or set manually here:
#COMP=gnu


###############################################################################
# PHONY TARGETS
###############################################################################
.PHONY: libnnp \
	pynnp \
	clean-libnnp \
	clean-pynnp


###############################################################################
# RULES
###############################################################################

libnnp:
	cd libnnp && $(MAKE) COMP=${COMP}

clean-libnnp:
	cd libnnp && $(MAKE) clean

pynnp:
	cd pynnp && $(MAKE)

clean-pynnp:
	cd pynnp && $(MAKE) clean
