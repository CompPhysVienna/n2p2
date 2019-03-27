#!/bin/make -f

###############################################################################
# GENERAL SETTINGS
###############################################################################
# Number of cores used.
CORES=-j4
# Compiler (gnu/intel).
COMP=gnu
# Installation directory for binaries.
INSTALL_BIN=${HOME}/local/bin
# Installation directory for libraries.
INSTALL_LIB=${HOME}/local/lib


###############################################################################
# PHONY TARGETS
###############################################################################
.PHONY: all\
	shared \
	static \
	doc \
	doc-sphinx \
	doc-doxygen \
        libnnp \
        libnnptrain \
        libnnpif \
        libnnp-static \
        libnnptrain-static \
        libnnpif-static \
        libnnp-shared \
        libnnptrain-shared \
        libnnpif-shared \
        nnp-comp2-shared \
        nnp-comp2-static \
        nnp-convert-shared \
        nnp-convert-static \
        nnp-cutoff-shared \
        nnp-cutoff-static \
        nnp-dataset-shared \
        nnp-dataset-static \
        nnp-dist-shared \
        nnp-dist-static \
        nnp-norm-shared \
        nnp-norm-static \
        nnp-predict-shared \
        nnp-predict-static \
        nnp-prune-shared \
        nnp-prune-static \
        nnp-scaling-shared \
        nnp-scaling-static \
        nnp-select-shared \
        nnp-select-static \
        nnp-symfunc-shared \
        nnp-symfunc-static \
        nnp-train-shared \
        nnp-train-static \
        pynnp \
        clean-doc \
        clean-doc-sphinx \
        clean-doc-doxygen \
	clean-no-doc \
        clean-libnnp \
        clean-libnnptrain \
        clean-libnnpif \
        clean-libnnp-static \
        clean-libnnptrain-static \
        clean-libnnpif-static \
        clean-libnnp-shared \
        clean-libnnptrain-shared \
        clean-libnnpif-shared \
        clean-nnp-comp2 \
        clean-nnp-convert \
        clean-nnp-cutoff \
        clean-nnp-dataset \
        clean-nnp-dist \
        clean-nnp-norm \
        clean-nnp-predict \
        clean-nnp-prune \
        clean-nnp-scaling \
        clean-nnp-select \
        clean-nnp-symfunc \
        clean-nnp-train \
        clean-pynnp \
	install \
	uninstall


###############################################################################
# PROJECT
###############################################################################
PROJECT_DIR=../
PROJECT_BIN=${PROJECT_DIR}/bin
PROJECT_INCLUDE=${PROJECT_DIR}/include
PROJECT_LIB=${PROJECT_DIR}/lib
PROJECT_BUILD=${PROJECT_DIR}/build


###############################################################################
# RULES
###############################################################################

all:    libnnp \
	libnnptrain \
	libnnpif \
	nnp-comp2-shared \
	nnp-convert-shared \
	nnp-cutoff-shared \
	nnp-dataset-shared \
	nnp-dist-shared \
	nnp-norm-shared \
	nnp-predict-shared \
	nnp-prune-shared \
	nnp-scaling-shared \
	nnp-select-shared \
	nnp-symfunc-shared \
	nnp-train-shared

shared: libnnp-shared \
	libnnptrain-shared \
	libnnpif-shared \
	nnp-comp2-shared \
	nnp-convert-shared \
	nnp-cutoff-shared \
	nnp-dataset-shared \
	nnp-dist-shared \
	nnp-norm-shared \
	nnp-predict-shared \
	nnp-prune-shared \
	nnp-scaling-shared \
	nnp-select-shared \
	nnp-symfunc-shared \
	nnp-train-shared

static: libnnp-static \
	libnnptrain-static \
	libnnpif-static \
	nnp-comp2-static \
	nnp-convert-static \
	nnp-cutoff-static \
	nnp-dataset-static \
	nnp-dist-static \
	nnp-norm-static \
	nnp-predict-static \
	nnp-prune-static \
	nnp-scaling-static \
	nnp-select-static \
	nnp-symfunc-static \
	nnp-train-static

clean:	clean-doc \
	clean-doc-sphinx \
	clean-doc-doxygen \
	clean-no-doc

clean-no-doc: clean-libnnp \
	      clean-libnnptrain \
	      clean-libnnpif \
	      clean-nnp-comp2 \
	      clean-nnp-convert \
	      clean-nnp-cutoff \
	      clean-nnp-dataset \
	      clean-nnp-dist \
	      clean-nnp-norm \
	      clean-nnp-predict \
	      clean-nnp-prune \
	      clean-nnp-scaling \
	      clean-nnp-select \
	      clean-nnp-symfunc \
	      clean-nnp-train \
	      clean-pynnp


#######
# doc #
#######
doc: doc-doxygen doc-sphinx
	cp -r ../doc/doxygen/html ../doc/sphinx/html/doxygen
	
clean-doc: clean-doc-sphinx clean-doc-doxygen

##############
# doc-sphinx #
##############
doc-sphinx:
	cd doc/sphinx && $(MAKE) html

clean-doc-sphinx:
	cd doc/sphinx && $(MAKE) clean

###############
# doc-doxygen #
###############
doc-doxygen:
	cd doc/doxygen && $(MAKE)

clean-doc-doxygen:
	cd doc/doxygen && $(MAKE) clean

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

#########################
# libnnptrain, libnnpif #
#########################
libnnptrain libnnpif: libnnp
	cd $@ && $(MAKE) COMP=${COMP} ${CORES}

clean-libnnptrain clean-libnnpif: LIBDIR=$(subst clean-,,$@)
clean-libnnptrain clean-libnnpif:
	cd ${LIBDIR} && $(MAKE) clean

libnnptrain-shared libnnpif-shared: LIBDIR=$(subst -shared,,$@)
libnnptrain-shared libnnpif-shared: libnnp-shared
	cd ${LIBDIR} && $(MAKE) COMP=${COMP} ${CORES} shared

clean-libnnptrain-shared clean-libnnpif-shared: LIBDIR=$(subst clean-,,$(subst -shared,,$@))
clean-libnnptrain-shared clean-libnnpif-shared:
	cd ${LIBDIR} && $(MAKE) clean-shared

libnnptrain-static libnnpif-static: LIBDIR=$(subst -static,,$@)
libnnptrain-static libnnpif-static: libnnp-static
	cd ${LIBDIR} && $(MAKE) COMP=${COMP} ${CORES} static

clean-libnnptrain-static clean-libnnpif-static: LIBDIR=$(subst clean-,,$(subst -static,,$@))
clean-libnnptrain-static clean-libnnpif-static:
	cd ${LIBDIR} && $(MAKE) clean-static

######################################################################################
# nnp-convert, nnp-cutoff, nnp-dist, nnp-predict, nnp-prune, nnp-select, nnp-symfunc #
######################################################################################
nnp-convert-shared nnp-cutoff-shared nnp-dist-shared nnp-predict-shared nnp-prune-shared nnp-select-shared nnp-symfunc-shared: APPDIR=$(subst -shared,,$@)
nnp-convert-shared nnp-cutoff-shared nnp-dist-shared nnp-predict-shared nnp-prune-shared nnp-select-shared nnp-symfunc-shared: libnnp-shared
	cd application/${APPDIR} && $(MAKE) COMP=${COMP} ${CORES} shared

nnp-convert-static nnp-cutoff-static nnp-dist-static nnp-predict-static nnp-prune-static nnp-select-static nnp-symfunc-static: APPDIR=$(subst -static,,$@)
nnp-convert-static nnp-cutoff-static nnp-dist-static nnp-predict-static nnp-prune-static nnp-select-static nnp-symfunc-static: libnnp-static
	cd application/${APPDIR} && $(MAKE) COMP=${COMP} ${CORES} static

clean-nnp-convert clean-nnp-cutoff clean-nnp-dist clean-nnp-predict clean-nnp-prune clean-nnp-select clean-nnp-symfunc: APPDIR=$(subst clean-,,$@)
clean-nnp-convert clean-nnp-cutoff clean-nnp-dist clean-nnp-predict clean-nnp-prune clean-nnp-select clean-nnp-symfunc:
	cd application/${APPDIR} && $(MAKE) clean

############################################################
# nnp-comp2, nnp-dataset, nnp-norm, nnp-scaling, nnp-train #
############################################################
nnp-comp2-shared nnp-dataset-shared nnp-norm-shared nnp-scaling-shared nnp-train-shared: APPDIR=$(subst -shared,,$@)
nnp-comp2-shared nnp-dataset-shared nnp-norm-shared nnp-scaling-shared nnp-train-shared: libnnptrain-shared
	cd application/${APPDIR} && $(MAKE) COMP=${COMP} ${CORES} shared

nnp-comp2-static nnp-dataset-static nnp-norm-static nnp-scaling-static nnp-train-static: APPDIR=$(subst -static,,$@)
nnp-comp2-static nnp-dataset-static nnp-norm-static nnp-scaling-static nnp-train-static: libnnptrain-static
	cd application/${APPDIR} && $(MAKE) COMP=${COMP} ${CORES} static

clean-nnp-comp2 clean-nnp-dataset clean-nnp-norm clean-nnp-scaling clean-nnp-train: APPDIR=$(subst clean-,,$@)
clean-nnp-comp2 clean-nnp-dataset clean-nnp-norm clean-nnp-scaling clean-nnp-train:
	cd application/${APPDIR} && $(MAKE) clean

#########
# pynnp #
#########
pynnp:
	cd pynnp && $(MAKE)

clean-pynnp:
	cd pynnp && $(MAKE) clean

##############################################
# install (set links to install directories) #
##############################################
EXE=nnp-comp2 nnp-convert nnp-cutoff nnp-dataset nnp-dist nnp-norm nnp-predict nnp-prune nnp-scaling nnp-select nnp-symfunc nnp-train
LIB=libnnp.so libnnpif.so libnnptrain.so pynnp.so
install:
	$(foreach exe, $(EXE), ln -s $(CURDIR)/${PROJECT_BIN}/${exe} ${INSTALL_BIN}/${exe};)
	$(foreach lib, $(LIB), ln -s $(CURDIR)/${PROJECT_LIB}/${lib} ${INSTALL_LIB}/${lib};)

uninstall:
	$(foreach exe, $(EXE), rm ${INSTALL_BIN}/${exe};)
	$(foreach lib, $(LIB), rm ${INSTALL_LIB}/${lib};)
