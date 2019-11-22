#!/bin/make -f

###############################################################################
# PROJECT PATHS
###############################################################################
PROJECT_DIR=..
PROJECT_BIN=$(PROJECT_DIR)/bin
PROJECT_INCLUDE=$(PROJECT_DIR)/include
PROJECT_LIB=$(PROJECT_DIR)/lib


###############################################################################
# PHONY TARGETS
###############################################################################
.PHONY: all clean cpp python clean-cpp clean-python


###############################################################################
# RULES
###############################################################################

all: cpp cpp-coverage python

clean: clean-cpp clean-python

cpp:
	cd cpp && $(MAKE)

cpp-coverage:
	cd cpp && $(MAKE) coverage

clean-cpp:
	cd cpp && $(MAKE) clean

python:
	cd python && $(MAKE)

clean-python:
	cd python && $(MAKE) clean
