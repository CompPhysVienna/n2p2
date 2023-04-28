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
APP_CORE=nnp-convert \
         nnp-cutoff \
         nnp-dist \
         nnp-predict \
         nnp-prune \
         nnp-select \
         nnp-symfunc

# Applications which require libnnp and libnnptrain
# but do not use class Training (only Dataset):
APP_DATASET=nnp-atomenv \
            nnp-checkf \
            nnp-comp2 \
            nnp-dataset \
            nnp-norm \
            nnp-scaling

# Applications which require libnnp and libnnptrain:
APP_TRAINING=nnp-checkdw \
             nnp-norm2 \
             nnp-train

# All applications together.
APP=$(APP_CORE) $(APP_DATASET) $(APP_TRAINING)

# Targets for cleaning.
CLEAN_APP=$(patsubst %, clean-%, $(APP))


###############################################################################
# RULES
###############################################################################
.PHONY: all $(CLEAN_APP) list-libnnp list-libnnptrain list-all

all: $(APP_CORE) $(APP_DATASET) $(APP_TRAINING)

# Applications which require only libnnp:
$(APP_CORE): INCLUDES=-I./ -I$(PROJECT_INCLUDE)/ -I$(PROJECT_EIGEN)
ifeq ($(MODE), shared)
$(APP_CORE): LDFLAGS=-L$(PROJECT_LIB) -lnnp
else
$(APP_CORE): LDFLAGS=$(PROJECT_LIB)/libnnp.a
endif
$(APP_CORE):
	$(CC) $(CFLAGS) $(DEBUG) $(INCLUDES) $(OPTIONS) -o $@.o -c $@.cpp
	$(CC) $(CFLAGS) $(DEBUG) $(OPTIONS) -o $@ $@.o $(LDFLAGS)
	cp $@ $(PROJECT_BIN)/

# Applications which require libnnp and libnnptrain
# but do not use class Training (only Dataset):
$(APP_DATASET): INCLUDES=-I./ -I$(PROJECT_INCLUDE)/ -I$(PROJECT_GSL) -I$(PROJECT_EIGEN)
ifeq ($(MODE), shared)
$(APP_DATASET): LDFLAGS=-L$(PROJECT_LIB) -lnnptrain -lnnp -lgsl $(PROJECT_LDFLAGS_BLAS)
else
$(APP_DATASET): LDFLAGS=$(PROJECT_LIB)/libnnptrain.a $(PROJECT_LIB)/libnnp.a -lgsl $(PROJECT_LDFLAGS_BLAS)
endif
$(APP_DATASET):
	$(MPICC) $(CFLAGS) $(DEBUG) $(INCLUDES) $(OPTIONS) -o $@.o -c $@.cpp
	$(MPICC) $(CFLAGS) $(DEBUG) $(OPTIONS) -o $@ $@.o $(LDFLAGS)
	cp $@ $(PROJECT_BIN)/

# Applications which require libnnp and libnnptrain, requires Eigen.
$(APP_TRAINING): INCLUDES=-I./ -I$(PROJECT_INCLUDE)/ -I$(PROJECT_GSL) -I$(PROJECT_EIGEN)
ifeq ($(MODE), shared)
$(APP_TRAINING): LDFLAGS=-L$(PROJECT_LIB) -lnnptrain -lnnp -lgsl $(PROJECT_LDFLAGS_BLAS)
else
$(APP_TRAINING): LDFLAGS=$(PROJECT_LIB)/libnnptrain.a $(PROJECT_LIB)/libnnp.a -lgsl $(PROJECT_LDFLAGS_BLAS)
endif
$(APP_TRAINING): nnp-train.cpp
	$(MPICC) $(CFLAGS) $(DEBUG) $(INCLUDES) $(OPTIONS) -o $@.o -c $@.cpp
	$(MPICC) $(CFLAGS) $(DEBUG) $(OPTIONS) -o $@ $@.o $(LDFLAGS)
	cp $@ $(PROJECT_BIN)/

# Clean everything.
clean: $(CLEAN_APP)

# Clean rules for all applications.
$(CLEAN_APP): EXE=$(patsubst clean-%,%,$@)
$(CLEAN_APP):
	$(RM) $(EXE).gcno $(EXE).gcda $(EXE).o $(EXE)
	$(RM) $(PROJECT_BIN)/$(EXE)

list-libnnp:
	@echo $(APP_CORE)

list-libnnptrain:
	@echo $(APP_DATASET) $(APP_TRAINING)

list-all:
	@echo $(APP_CORE) $(APP_DATASET) $(APP_TRAINING)
