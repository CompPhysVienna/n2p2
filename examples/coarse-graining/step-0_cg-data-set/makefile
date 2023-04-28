#!/bin/make -f

###############################################################################
# PROJECT PATHS
###############################################################################
PROJECT_DIR=../../../
PROJECT_BIN=$(PROJECT_DIR)/bin
PROJECT_INCLUDE=$(PROJECT_DIR)/include
PROJECT_LIB=$(PROJECT_DIR)/lib


###############################################################################
# GENERAL SETTINGS
###############################################################################
# Default compiler, may be overridden by master makefile or command line.
COMP=gnu

# Default build mode, may be overridden by master makefile or command line.
# Possible modes are "static", "shared" and "test".
MODE=static

# Include global (project-wide) settings.
include $(PROJECT_DIR)/src/makefile.$(COMP)


###############################################################################
# PROJECT COMPILER AND FLAGS
###############################################################################
# Include directories and linking flags are target specific (see below).
CC=$(PROJECT_CC)
MPICC=$(PROJECT_MPICC)
CFLAGS=$(PROJECT_CFLAGS)
OPTIONS+=$(PROJECT_OPTIONS)
DEBUG=$(PROJECT_DEBUG)
ifeq ($(MODE), test)
	DEBUG+= $(PROJECT_TEST)
endif


###############################################################################
# APPLICATION TYPES
###############################################################################
# Applications which require only libnnp:
APP_CORE=nnp-cgdata

# Targets for cleaning.
CLEAN_APP=$(patsubst %, clean-%, $(APP_CORE))


###############################################################################
# RULES
###############################################################################
.PHONY: all $(CLEAN_APP) list-core

all: $(APP_CORE)

# Applications which require only libnnp:
$(APP_CORE): INCLUDES=-I./ -I$(PROJECT_INCLUDE)/
ifeq ($(MODE), shared)
$(APP_CORE): LDFLAGS=-L$(PROJECT_LIB) -lnnp
else
$(APP_CORE): LDFLAGS=$(PROJECT_LIB)/libnnp.a
endif
$(APP_CORE):
	$(CC) $(CFLAGS) $(DEBUG) $(INCLUDES) $(OPTIONS) -o $@.o -c $@.cpp
	$(CC) $(CFLAGS) $(DEBUG) $(OPTIONS) -o $@ $@.o $(LDFLAGS)
	#cp $@ $(PROJECT_BIN)/

# Clean everything.
clean: $(CLEAN_APP)

# Clean rules for all applications.
$(CLEAN_APP): EXE=$(patsubst clean-%,%,$@)
$(CLEAN_APP):
	$(RM) $(EXE).gcno $(EXE).gcda $(EXE).o $(EXE)
	#$(RM) $(PROJECT_BIN)/$(EXE)

list-core:
	@echo $(APP_CORE)
