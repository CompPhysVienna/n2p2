#!/bin/make -f

###############################################################################
# GENERAL SETTINGS
###############################################################################
# Number of cores used.
CORES=-j4
# Compiler (gnu/intel).
COMP=gnu


###############################################################################
# PHONY TARGETS
###############################################################################
.PHONY: all\
	shared \
	static \
	doc \
        libnnp \
        libnnpif \
        libnnp-static \
        libnnpif-static \
        libnnp-shared \
        libnnpif-shared \
        clean-doc \
        clean-libnnp \
        clean-libnnpif \
        clean-libnnp-static \
        clean-libnnpif-static \
        clean-libnnp-shared \
        clean-libnnpif-shared


###############################################################################
# PROJECT
###############################################################################
PROJECT_DIR=./
PROJECT_BIN=${PROJECT_DIR}/bin
PROJECT_INCLUDE=${PROJECT_DIR}/include
PROJECT_LIB=${PROJECT_DIR}/lib
PROJECT_BUILD=${PROJECT_DIR}/build


###############################################################################
# RULES
###############################################################################

all:    libnnp \
	libnnpif

shared: libnnp-shared \
	libnnpif-shared

static: libnnp-static \
	libnnpif-static

clean:	clean-doc \
	clean-libnnp \
	clean-libnnpif

#######
# doc #
#######
doc:
	cd doc && $(MAKE)

clean-doc:
	cd doc && $(MAKE) clean

##########
# libnnp #
##########
libnnp:
	cd libnnp && $(MAKE) COMP=${COMP} ${CORES}

clean-libnnp:
	cd libnnp && $(MAKE) clean

libnnp-shared:
	cd libnnp && $(MAKE) COMP=${COMP} ${CORES} shared

clean-libnnp-shared:
	cd libnnp && $(MAKE) clean-shared

libnnp-static:
	cd libnnp && $(MAKE) COMP=${COMP} ${CORES} static

clean-libnnp-static:
	cd libnnp && $(MAKE) clean-static

############
# libnnpif #
############
libnnpif: libnnp
	cd $@ && $(MAKE) COMP=${COMP} ${CORES}

clean-libnnpif: LIBDIR=$(subst clean-,,$@)
clean-libnnpif:
	cd ${LIBDIR} && $(MAKE) clean

libnnpif-shared: LIBDIR=$(subst -shared,,$@)
libnnpif-shared: libnnp-shared
	cd ${LIBDIR} && $(MAKE) COMP=${COMP} ${CORES} shared

clean-libnnpif-shared: LIBDIR=$(subst clean-,,$(subst -shared,,$@))
clean-libnnpif-shared:
	cd ${LIBDIR} && $(MAKE) clean-shared

libnnpif-static: LIBDIR=$(subst -static,,$@)
libnnpif-static: libnnp-static
	cd ${LIBDIR} && $(MAKE) COMP=${COMP} ${CORES} static

clean-libnnpif-static: LIBDIR=$(subst clean-,,$(subst -static,,$@))
clean-libnnpif-static:
	cd ${LIBDIR} && $(MAKE) clean-static
