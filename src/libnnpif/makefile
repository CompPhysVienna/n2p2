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
NAME=nnpif
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

# Compile without MPI support.
#OPTIONS+= -DNOMPI


###############################################################################
# COMPILER AND FLAGS
###############################################################################
CC=${PROJECT_CC}
MPICC=${PROJECT_MPICC}
CFLAGS=${PROJECT_CFLAGS} ${PROJECT_CFLAGS_MPI}
DEBUG=${PROJECT_DEBUG}

# Extra include paths for compiling.
INCLUDES=-I./ -I${PROJECT_INCLUDE}/


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

# Static linking
SLIB=${PROJECT_LIB}/${LIB}.a
SDIR=${PROJECT_BUILD}/static
SOBJ=$(SRC:%.cpp=${SDIR}/%.o)

# Dynamic linking
DLIB=${PROJECT_LIB}/${LIB}.so
DDIR=${PROJECT_BUILD}/shared
DOBJ=$(SRC:%.cpp=${DDIR}/%.o)


###############################################################################
# RULES
###############################################################################
.PHONY: all static shared headers clean clean-static clean-shared rebuild

all: static shared

static: ${SLIB} headers

shared: ${DLIB} headers

headers:
	cp ${HEADERS} ${PROJECT_INCLUDE}

${SLIB}: ${SOBJ}
	${AR} ${ARFLAGS} $@ $+

${DLIB}: CFLAGS+= -fPIC
${DLIB}: ${DOBJ}
	${MPICC} ${CFLAGS} ${DEBUG} ${OPTIONS} -o $@ $+ -shared

${SDIR}/%.o: %.cpp
	${MPICC} ${CFLAGS} ${DEBUG} ${INCLUDES} ${OPTIONS} -o $@ -c $<

${DDIR}/%.o: %.cpp
	${MPICC} ${CFLAGS} ${DEBUG} ${INCLUDES} ${OPTIONS} -o $@ -c $<

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
