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
LAMMPS_VERSION=stable_29Aug2024

# Include global (project-wide) settings.
include $(PROJECT_DIR)/src/makefile.$(COMP)


###############################################################################
# RULES
###############################################################################
.PHONY: all lammps-hdnnp clean-lammps-hdnnp clean-lammps-hdnnp-keep-file

all: lammps-hdnnp

clean: clean-lammps-hdnnp

################
# lammps-hdnnp #
################
LAMMPS_DEBUG := $(filter-out -pedantic-errors,$(PROJECT_DEBUG))
lammps-hdnnp:
	if [ ! -e $(LAMMPS_VERSION).tar.gz ]; then \
		wget https://github.com/lammps/lammps/archive/$(LAMMPS_VERSION).tar.gz; \
	fi
	tar -xzf $(LAMMPS_VERSION).tar.gz && mv lammps-$(LAMMPS_VERSION) lammps-hdnnp
	ln -s $(PROJECT_DIR)/../../../include/ lammps-hdnnp/lib/hdnnp/includelink
	ln -s $(PROJECT_DIR)/../../../lib/ lammps-hdnnp/lib/hdnnp/liblink
	ln -s $(PROJECT_DIR)/../../../lib/Makefile.lammps-extra lammps-hdnnp/lib/hdnnp/Makefile.lammps
	cp -r ./LAMMPS/src/ML-HDNNP/* lammps-hdnnp/src/ML-HDNNP/
	@sed -i.bak "s/^CC .*$$/CC      =       $(PROJECT_MPICC)/" lammps-hdnnp/src/MAKE/Makefile.mpi
	@sed -i.bak "s/^CCFLAGS .*$$/CCFLAGS =       $(PROJECT_CFLAGS) $(PROJECT_CFLAGS_MPI)/" lammps-hdnnp/src/MAKE/Makefile.mpi
	@sed -i.bak "s/^LINK .*$$/LINK =          $(PROJECT_MPICC)/" lammps-hdnnp/src/MAKE/Makefile.mpi
	@sed -i.bak "s/^LINKFLAGS .*$$/LINKFLAGS =     $(PROJECT_CFLAGS) $(PROJECT_CFLAGS_MPI) $(PROJECT_LDFLAGS_BLAS)/" lammps-hdnnp/src/MAKE/Makefile.mpi
	if [ "$(MODE)" = "test" ]; then \
		sed -i.bak "/^CCFLAGS =/   s/$$/ $(LAMMPS_DEBUG) $(PROJECT_TEST)/" lammps-hdnnp/src/MAKE/Makefile.mpi; \
		sed -i.bak "/^LINKFLAGS =/ s/$$/ $(LAMMPS_DEBUG) $(PROJECT_TEST)/" lammps-hdnnp/src/MAKE/Makefile.mpi; \
	fi
	@rm lammps-hdnnp/src/MAKE/Makefile.mpi.bak
	cd lammps-hdnnp/src/ && $(MAKE) yes-ml-hdnnp && $(MAKE) yes-molecule && $(MAKE) mpi
	cp lammps-hdnnp/src/lmp_mpi $(PROJECT_BIN)/

clean-lammps-hdnnp:
	$(RM) -r lammps-hdnnp
	$(RM) $(LAMMPS_VERSION).tar.gz
	$(RM) $(PROJECT_BIN)/lmp_mpi

clean-lammps-hdnnp-keep-file:
	$(RM) -r lammps-hdnnp
	$(RM) $(PROJECT_BIN)/lmp_mpi
