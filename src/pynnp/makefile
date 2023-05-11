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
# Default compiler, may be overridden by master makefile or command line.
COMP=gnu

# Default build mode, may be overridden command line.
# Possible modes are "shared" and "test".
MODE=shared
# Override "static" mode from main makefile.
ifeq ($(MODE), static)
override MODE=shared
endif

# Include global (project-wide) settings.
include $(PROJECT_DIR)/src/makefile.$(COMP)

###############################################################################
# RULES
###############################################################################
.PHONY: all setup-py clean-setup-py clean

ifeq ($(MODE), shared)
all: setup-py
	python3 setup.py build_ext --inplace
	cp pynnp*.so $(PROJECT_LIB)
else
all: setup-py
	python3 setup-test.py build_ext --inplace --define CYTHON_TRACE
	cp pynnp*.so $(PROJECT_LIB)
endif

setup-py:
	@sed -i.bak -E "s,(includes =).*,\1 [\"$(PROJECT_EIGEN)\"]," setup.py
	@rm setup.py.bak
	@sed -i.bak -E "s,(includes =).*,\1 [\"$(PROJECT_EIGEN)\"]," setup-test.py
	@rm setup-test.py.bak

clean: clean-setup-py
	$(RM) pynnp.cpp pynnp*.so
	$(RM) -r build
	$(RM) $(PROJECT_LIB)/pynnp*.so
	$(RM) ./*.html

clean-setup-py:
	@sed -i.bak -E "s,(includes =).*,\1 []," setup.py
	@rm setup.py.bak
	@sed -i.bak -E "s,(includes =).*,\1 []," setup-test.py
	@rm setup-test.py.bak
