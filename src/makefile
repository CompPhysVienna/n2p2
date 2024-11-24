#!/bin/make -f

###############################################################################
# PROJECT PATHS
###############################################################################
PROJECT_DIR=..
PROJECT_BIN=$(PROJECT_DIR)/bin
PROJECT_INCLUDE=$(PROJECT_DIR)/include
PROJECT_LIB=$(PROJECT_DIR)/lib


###############################################################################
# GENERAL SETTINGS
###############################################################################
# Default compiler (gnu/intel/llvm).
COMP=gnu

# Default build mode.
# Possible modes are "static", "shared" and "test".
# Note: For target "pynnp" there is no "static" mode, "shared" is automatically
# used instead.
MODE=static

# Installation directory for binaries.
INSTALL_BIN=$(HOME)/local/bin

# Installation directory for libraries.
INSTALL_LIB=$(HOME)/local/lib


###############################################################################
# INTERFACE LIBRARY OPTIONS
###############################################################################

# Select which interfaces should be compiled into libnnpif. Provide a
# space-separated list of the following available options:
#    * LAMMPS
#    * CabanaMD
INTERFACES=LAMMPS CabanaMD


###############################################################################
# APPLICATION TARGET NAMES
###############################################################################

# Variables should not contain "entering directory...", so we temporarily
# disable it
MAKEFLAGS_ORIGINAL:=$(MAKEFLAGS)
MAKEFLAGS += --no-print-directory

# Call makefile in "application" directory to list targets.
APP_LIBNNP:=$(shell cd application && $(MAKE) $(MFLAGS) list-libnnp)
APP_LIBNNPTRAIN:=$(shell cd application && $(MAKE) $(MFLAGS) list-libnnptrain)
APP:=$(shell cd application && $(MAKE) $(MFLAGS) list-all)

# Targets for cleaning.
CLEAN_APP=$(patsubst %, clean-%, $(APP))

MAKEFLAGS:=$(MAKEFLAGS_ORIGINAL)

###############################################################################
# RULES
###############################################################################
.PHONY:	all \
	doc \
	doc-sphinx \
	doc-doxygen \
	libnnp \
	libnnptrain \
	libnnpif \
	lammps-hdnnp \
	pynnp \
	clean-doc \
	clean-doc-sphinx \
	clean-doc-doxygen \
	clean-no-doc \
	clean-libnnp \
	clean-libnnptrain \
	clean-libnnpif \
	clean-lammps-hdnnp \
	clean-pynnp \
	install \
	uninstall

all: libnnp \
     libnnptrain \
     libnnpif \
     all-app

clean: clean-libnnp \
       clean-libnnptrain \
       clean-libnnpif \
       clean-app \
       clean-lammps-hdnnp \
       clean-pynnp

clean-all: clean \
           clean-doc

################################
# doc, doc-sphinx, doc-doxygen #
################################
doc: doc-doxygen doc-sphinx
	cp -r ../doc/doxygen/html ../doc/sphinx/html/doxygen

clean-doc: clean-doc-sphinx clean-doc-doxygen

doc-sphinx:
	cd doc/sphinx && $(MAKE) html

clean-doc-sphinx:
	cd doc/sphinx && $(MAKE) clean

doc-doxygen:
	cd doc/doxygen && $(MAKE)

clean-doc-doxygen:
	cd doc/doxygen && $(MAKE) clean

##########
# libnnp #
##########
libnnp:
	cd libnnp && $(MAKE) COMP=$(COMP) MODE=$(MODE)

clean-libnnp:
	cd libnnp && $(MAKE) clean

###############
# libnnptrain #
###############
libnnptrain: libnnp
	cd $@ && $(MAKE) COMP=$(COMP) MODE=$(MODE)

clean-libnnptrain: LIBDIR=$(subst clean-,,$@)
clean-libnnptrain:
	cd $(LIBDIR) && $(MAKE) clean

############
# libnnpif #
############
libnnpif: libnnp
	cd $@ && $(MAKE) COMP=$(COMP) MODE=$(MODE) INTERFACES="$(INTERFACES)"

clean-libnnpif: LIBDIR=$(subst clean-,,$@)
clean-libnnpif:
	cd $(LIBDIR) && $(MAKE) INTERFACES="$(INTERFACES)" clean

####################
# nnp applications #
####################
# All at once (parallel).
all-app: libnnptrain
	cd application && $(MAKE) COMP=$(COMP) MODE=$(MODE)

clean-app:
	cd application && $(MAKE) clean

# Individual applications.
$(APP_LIBNNP): libnnp
	cd application && $(MAKE) $@ COMP=$(COMP) MODE=$(MODE)

$(APP_LIBNNPTRAIN): libnnptrain
	cd application && $(MAKE) $@ COMP=$(COMP) MODE=$(MODE)

$(CLEAN_APP):
	cd application && $(MAKE) $@

################
# lammps-hdnnp #
################
define LAMMPS_NNP_WARNING
##############################################################
#           __      ___   ___ _  _ ___ _  _  ___             #
#           \ \    / /_\ | _ \ \| |_ _| \| |/ __|            #
#            \ \/\/ / _ \|   / .` || || .` | (_ |            #
#             \_/\_/_/ \_\_|_\_|\_|___|_|\_|\___|            #
#                                                            #
# WARNING: This is a development build of LAMMPS + n2p2!     #
#                                                            #
# n2p2's makefile feature to build an automatically patched  #
# version of LAMMPS is designed for development purposes and #
# not guarantueed to work. The recommended way to build and  #
# link LAMMPS with n2p2 is based on the build system of      #
# LAMMPS and described here:                                 #
#                                                            #
# https://docs.lammps.org/Build_extras.html#ml-hdnnp         #
#                                                            #
##############################################################
endef

export LAMMPS_NNP_WARNING
lammps-hdnnp: libnnpif
	@echo "$$LAMMPS_NNP_WARNING"
	cd libnnpif && $(MAKE) lammps-mf
	cd interface && $(MAKE) $@ COMP=$(COMP) MODE=$(MODE)
	@echo "$$LAMMPS_NNP_WARNING"

clean-lammps-hdnnp:
	#cd interface && $(MAKE) $@
	cd interface && $(MAKE) clean-lammps-hdnnp-keep-file

#########
# pynnp #
#########
pynnp:
	cd pynnp && $(MAKE) MODE=$(MODE)

clean-pynnp:
	cd pynnp && $(MAKE) clean

##############################################
# install (set links to install directories) #
##############################################
EXE=$(APP)
DLIB=libnnp.so libnnpif.so libnnptrain.so
SLIB=$(DLIB:.so=.a)
PYLIB=$(wildcard $(PROJECT_LIB)/pynnp*.so)
install:
	$(foreach exe, $(EXE), ln -s $(CURDIR)/$(PROJECT_BIN)/$(exe) $(INSTALL_BIN)/$(exe);)
	$(foreach dlib, $(DLIB), ln -s $(CURDIR)/$(PROJECT_LIB)/$(dlib) $(INSTALL_LIB)/$(dlib);)
	$(foreach slib, $(SLIB), ln -s $(CURDIR)/$(PROJECT_LIB)/$(slib) $(INSTALL_LIB)/$(slib);)
	$(foreach pylib, $(notdir $(PYLIB)), ln -s $(CURDIR)/$(PROJECT_LIB)/$(pylib) $(INSTALL_LIB)/$(pylib);)

uninstall:
	$(foreach exe, $(EXE), rm $(INSTALL_BIN)/$(exe);)
	$(foreach dlib, $(DLIB), rm $(INSTALL_LIB)/$(dlib);)
	$(foreach slib, $(SLIB), rm $(INSTALL_LIB)/$(slib);)
	$(foreach pylib, $(notdir $(PYLIB)), rm $(INSTALL_LIB)/$(pylib);)
