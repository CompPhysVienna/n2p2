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
.PHONY: all lammps-nnp clean-lammps-nnp

all: lammps-nnp

clean: clean-lammps-nnp

##############
# lammps-nnp #
##############
lammps-nnp:
	if [ ! -e $(LAMMPS_VERSION).tar.gz ]; then \
		wget https://github.com/lammps/lammps/archive/$(LAMMPS_VERSION).tar.gz; \
	fi
	tar -xzf $(LAMMPS_VERSION).tar.gz
	ln -s $(PROJECT_DIR)/../../ lammps-$(LAMMPS_VERSION)/lib/nnp
	cp -r ./LAMMPS/src/USER-NNP lammps-$(LAMMPS_VERSION)/src/
	if [ "$(MODE)" = "test" ]; then \
		sed -i "/^CCFLAGS =/   s/$$/ $(PROJECT_DEBUG) $(PROJECT_TEST)/" lammps-$(LAMMPS_VERSION)/src/MAKE/Makefile.mpi; \
		sed -i "/^LINKFLAGS =/ s/$$/ $(PROJECT_DEBUG) $(PROJECT_TEST)/" lammps-$(LAMMPS_VERSION)/src/MAKE/Makefile.mpi; \
	fi
	cd lammps-$(LAMMPS_VERSION)/src/ && $(MAKE) yes-molecule && $(MAKE) yes-user-nnp && $(MAKE) mpi
	cp lammps-$(LAMMPS_VERSION)/src/lmp_mpi $(PROJECT_BIN)/

clean-lammps-nnp:
	$(RM) -r lammps-$(LAMMPS_VERSION)
	$(RM) $(LAMMPS_VERSION).tar.gz
	$(RM) $(PROJECT_BIN)/lmp_mpi

clean-lammps-nnp-keep-file:
	$(RM) -r lammps-$(LAMMPS_VERSION)
	$(RM) $(PROJECT_BIN)/lmp_mpi
