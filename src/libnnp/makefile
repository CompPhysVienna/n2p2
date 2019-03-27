#!/bin/make -f

###############################################################################
# PROJECT MAKEFILE
###############################################################################
# Default compiler, may be overridden by master makefile.
COMP=gnu

# Include global (project-wide) settings.
include ../makefile.${COMP}


###############################################################################
# LIBRARY NAME
###############################################################################
NAME=nnp
LIB=lib${NAME}


###############################################################################
# PROJECT PATHS
###############################################################################
PROJECT_DIR=../..
PROJECT_BIN=${PROJECT_DIR}/bin
PROJECT_INCLUDE=${PROJECT_DIR}/include
PROJECT_LIB=${PROJECT_DIR}/lib
PROJECT_BUILD=${PROJECT_DIR}/build


###############################################################################
# COMPILE-TIME OPTIONS
###############################################################################
OPTIONS+=${PROJECT_OPTIONS}

# Do not use symmetry function groups.
#OPTIONS+= -DNOSFGROUPS

# Do not use cutoff function cache.
#OPTIONS+= -DNOCFCACHE

# Build with dummy Stopwatch class.
#OPTIONS+= -DNOTIME

# Disable check for low number of neighbors.
#OPTIONS+= -DNONEIGHCHECK


###############################################################################
# COMPILER AND FLAGS
###############################################################################
CC=${PROJECT_CC}
MPICC=${PROJECT_MPICC}
CFLAGS=${PROJECT_CFLAGS}
DEBUG=${PROJECT_DEBUG}

# Extra include paths for compiling.
INCLUDES=-I./


###############################################################################
# ARCHIVE
###############################################################################
AR=${PROJECT_AR}
ARFLAGS=${PROJECT_ARFLAGS}


###############################################################################
# BUILD DETAILS
###############################################################################
SRC=$(wildcard *.cpp)
HEADERS=$(wildcard *.h)

# Static linking.
SLIB=${PROJECT_LIB}/${LIB}.a
SDIR=${PROJECT_BUILD}/static
SOBJ=$(SRC:%.cpp=${SDIR}/%.o)

# Dynamic linking.
DLIB=${PROJECT_LIB}/${LIB}.so
DDIR=${PROJECT_BUILD}/shared
DOBJ=$(SRC:%.cpp=${DDIR}/%.o)


###############################################################################
# RULES
###############################################################################
.PHONY: all static shared headers version clean clean-static clean-shared rebuild

all: static shared

static: version ${SLIB} headers

shared: version ${DLIB} headers

headers:
	cp ${HEADERS} ${PROJECT_INCLUDE}

version:
	@$(eval GIT_REV_SHORT = $(shell git rev-parse --short HEAD))
	@sed -i.bak -E "s/(NNP_GIT_REV_SHORT) .*/\1 \"${GIT_REV_SHORT}\"/" version.h
	@$(eval GIT_REV = $(shell git rev-parse HEAD))
	@sed -i.bak -E "s/(NNP_GIT_REV) .*/\1 \"${GIT_REV}\"/" version.h
	@$(eval GIT_BRANCH = $(shell git rev-parse --abbrev-ref HEAD))
	@sed -i.bak -E "s/(NNP_GIT_BRANCH) .*/\1 \"${GIT_BRANCH}\"/" version.h
	@rm version.h.bak

${SLIB}: ${SOBJ}
	${AR} ${ARFLAGS} $@ $+

${DLIB}: CFLAGS+= -fPIC
${DLIB}: ${DOBJ}
	${CC} ${CFLAGS} ${DEBUG} ${OPTIONS} -o $@ $+ -shared

${SDIR}/%.o: %.cpp
	${CC} ${CFLAGS} ${DEBUG} ${INCLUDES} ${OPTIONS} -o $@ -c $<

${DDIR}/%.o: %.cpp
	${CC} ${CFLAGS} ${DEBUG} ${INCLUDES} ${OPTIONS} -o $@ -c $<

clean: clean-static clean-shared clean-headers

clean-static:
	$(RM) ${SOBJ}
	$(RM) ${SLIB}

clean-shared:
	$(RM) ${DOBJ}
	$(RM) ${DLIB}

clean-headers:
	cd ${PROJECT_INCLUDE} && $(RM) ${HEADERS} 

rebuild: clean all
