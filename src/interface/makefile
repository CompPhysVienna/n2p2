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
# Possible modes are "static" and "test".
MODE=static

# LAMMPS version string
LAMMPS_VERSION=stable_23Jun2022
#LAMMPS_VERSION=stable_29Sep2021
#LAMMPS_VERSION=patch_27May2021
#LAMMPS_VERSION=patch_2Jul2021

# Include global (project-wide) settings.
include $(PROJECT_DIR)/src/makefile.$(COMP)


###############################################################################
# RULES
###############################################################################
.PHONY: all lammps-nnp clean-lammps-nnp clean-lammps-nnp-keep-file

all: lammps-nnp

clean: clean-lammps-nnp

##############
# lammps-nnp #
##############
LAMMPS_DEBUG := $(filter-out -pedantic-errors,$(PROJECT_DEBUG))
lammps-nnp:
	if [ ! -e $(LAMMPS_VERSION).tar.gz ]; then \
		wget https://github.com/lammps/lammps/archive/$(LAMMPS_VERSION).tar.gz; \
	fi
	tar -xzf $(LAMMPS_VERSION).tar.gz && mv lammps-$(LAMMPS_VERSION) lammps-nnp
	ln -s $(PROJECT_DIR)/../../../include/ lammps-nnp/lib/hdnnp/includelink
	ln -s $(PROJECT_DIR)/../../../lib/ lammps-nnp/lib/hdnnp/liblink
	ln -s $(PROJECT_DIR)/../../../lib/Makefile.lammps-extra lammps-nnp/lib/hdnnp/Makefile.lammps
	cp -r ./LAMMPS/src/ML-HDNNP/* lammps-nnp/src/ML-HDNNP/
	@sed -i.bak "s/^CC .*$$/CC      =       $(PROJECT_MPICC)/" lammps-nnp/src/MAKE/Makefile.mpi
	@sed -i.bak "s/^CCFLAGS .*$$/CCFLAGS =       $(PROJECT_CFLAGS) $(PROJECT_CFLAGS_MPI)/" lammps-nnp/src/MAKE/Makefile.mpi
	@sed -i.bak "s/^LINK .*$$/LINK =          $(PROJECT_MPICC)/" lammps-nnp/src/MAKE/Makefile.mpi
	@sed -i.bak "s/^LINKFLAGS .*$$/LINKFLAGS =     $(PROJECT_CFLAGS) $(PROJECT_CFLAGS_MPI)/" lammps-nnp/src/MAKE/Makefile.mpi
	if [ "$(MODE)" = "test" ]; then \
		sed -i.bak "/^CCFLAGS =/   s/$$/ $(LAMMPS_DEBUG) $(PROJECT_TEST)/" lammps-nnp/src/MAKE/Makefile.mpi; \
		sed -i.bak "/^LINKFLAGS =/ s/$$/ $(LAMMPS_DEBUG) $(PROJECT_TEST)/" lammps-nnp/src/MAKE/Makefile.mpi; \
	fi
	@rm lammps-nnp/src/MAKE/Makefile.mpi.bak
	cd lammps-nnp/src/ && $(MAKE) yes-ml-hdnnp && $(MAKE) yes-molecule && $(MAKE) mpi
	cp lammps-nnp/src/lmp_mpi $(PROJECT_BIN)/

clean-lammps-nnp:
	$(RM) -r lammps-nnp
	$(RM) $(LAMMPS_VERSION).tar.gz
	$(RM) $(PROJECT_BIN)/lmp_mpi

clean-lammps-nnp-keep-file:
	$(RM) -r lammps-nnp
	$(RM) $(PROJECT_BIN)/lmp_mpi
