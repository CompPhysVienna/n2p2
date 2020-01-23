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
# APPLICATION TARGET NAMES
###############################################################################
# Call makefile in "application" directory to list targets.
APP_CORE=$(shell cd application && make list-core)
APP_TRAIN=$(shell cd application && make list-train)
APP=$(shell cd application && make list-all)

# Targets for cleaning.
CLEAN_APP=$(patsubst %, clean-%, $(APP))


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
	pynnp \
	clean-doc \
	clean-doc-sphinx \
	clean-doc-doxygen \
	clean-no-doc \
	clean-libnnp \
	clean-libnnptrain \
	clean-libnnpif \
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

#########################
# libnnptrain, libnnpif #
#########################
libnnptrain libnnpif: libnnp
	cd $@ && $(MAKE) COMP=$(COMP) MODE=$(MODE)

clean-libnnptrain clean-libnnpif: LIBDIR=$(subst clean-,,$@)
clean-libnnptrain clean-libnnpif:
	cd $(LIBDIR) && $(MAKE) clean

####################
# nnp applications #
####################
# All at once (parallel).
all-app: libnnptrain
	cd application && $(MAKE) COMP=$(COMP) MODE=$(MODE)

clean-app:
	cd application && $(MAKE) clean

# Individual applications.
$(APP_CORE): libnnp
	cd application && $(MAKE) $@ COMP=$(COMP) MODE=$(MODE)

$(APP_TRAIN): libnnptrain
	cd application && $(MAKE) $@ COMP=$(COMP) MODE=$(MODE)

$(CLEAN_APP):
	cd application && $(MAKE) $@

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
