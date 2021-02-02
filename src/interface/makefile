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
LAMMPS_VERSION=stable_3Mar2020

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
lammps-nnp:
	if [ ! -e $(LAMMPS_VERSION).tar.gz ]; then \
		wget https://github.com/lammps/lammps/archive/$(LAMMPS_VERSION).tar.gz; \
	fi
	tar -xzf $(LAMMPS_VERSION).tar.gz && mv lammps-$(LAMMPS_VERSION) lammps-nnp
	ln -s $(PROJECT_DIR)/../../ lammps-nnp/lib/nnp
	cp -r ./LAMMPS/src/USER-NNP lammps-nnp/src/
	@sed -i.bak "s/^CC .*$$/CC      =       $(PROJECT_MPICC)/" lammps-nnp/src/MAKE/Makefile.mpi
	@sed -i.bak "s/^CCFLAGS .*$$/CCFLAGS =       $(PROJECT_CFLAGS) $(PROJECT_CFLAGS_MPI)/" lammps-nnp/src/MAKE/Makefile.mpi
	@sed -i.bak "s/^LINK .*$$/LINK =          $(PROJECT_MPICC)/" lammps-nnp/src/MAKE/Makefile.mpi
	@sed -i.bak "s/^LINKFLAGS .*$$/LINKFLAGS =     $(PROJECT_CFLAGS) $(PROJECT_CFLAGS_MPI)/" lammps-nnp/src/MAKE/Makefile.mpi
	if [ "$(MODE)" = "test" ]; then \
		sed -i.bak "/^CCFLAGS =/   s/$$/ $(PROJECT_DEBUG) $(PROJECT_TEST)/" lammps-nnp/src/MAKE/Makefile.mpi; \
		sed -i.bak "/^LINKFLAGS =/ s/$$/ $(PROJECT_DEBUG) $(PROJECT_TEST)/" lammps-nnp/src/MAKE/Makefile.mpi; \
	fi
	@rm lammps-nnp/src/MAKE/Makefile.mpi.bak
	cd lammps-nnp/src/ && $(MAKE) yes-user-nnp && $(MAKE) yes-molecule && $(MAKE) mpi
	cp lammps-nnp/src/lmp_mpi $(PROJECT_BIN)/

clean-lammps-nnp:
	$(RM) -r lammps-nnp
	$(RM) $(LAMMPS_VERSION).tar.gz
	$(RM) $(PROJECT_BIN)/lmp_mpi

clean-lammps-nnp-keep-file:
	$(RM) -r lammps-nnp
	$(RM) $(PROJECT_BIN)/lmp_mpi
