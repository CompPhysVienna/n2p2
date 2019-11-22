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
# Default build mode, may be overridden command line.
# Possible modes are "shared" and "test".
MODE=shared
# Override "static" mode from main makefile.
ifeq ($(MODE), static)
override MODE=shared
endif


###############################################################################
# RULES
###############################################################################
.PHONY: all clean

ifeq ($(MODE), shared)
all:
	python3 setup.py build_ext --inplace
	cp pynnp*.so $(PROJECT_LIB)
else
all:
	python3 setup-test.py build_ext --inplace --define CYTHON_TRACE
	cp pynnp*.so $(PROJECT_LIB)
endif

clean:
	$(RM) pynnp.cpp pynnp*.so
	$(RM) -r build
	$(RM) $(PROJECT_LIB)/pynnp*.so
	$(RM) ./*.html
