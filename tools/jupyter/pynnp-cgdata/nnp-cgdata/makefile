#!/bin/make -f

###############################################################################
# PROJECT MAKEFILE
###############################################################################
# Default compiler, may be overridden by master makefile.
COMP=gnu

# Include global (project-wide) settings.
include ../../../../src/makefile.${COMP}


###############################################################################
# PROGRAM NAME
###############################################################################
EXE=nnp-cgdata


###############################################################################
# PROJECT PATHS
###############################################################################
PROJECT_DIR=../../../..
PROJECT_BIN=${PROJECT_DIR}/bin
PROJECT_INCLUDE=${PROJECT_DIR}/include
PROJECT_LIB=${PROJECT_DIR}/lib
PROJECT_BUILD=${PROJECT_DIR}/build


###############################################################################
# COMPILE-TIME OPTIONS
###############################################################################
OPTIONS+=${PROJECT_OPTIONS}

#
#OPTIONS+=


###############################################################################
# COMPILER AND FLAGS
###############################################################################
CC=${PROJECT_CC}
MPICC=${PROJECT_MPICC}
CFLAGS=${PROJECT_CFLAGS}
DEBUG=${PROJECT_DEBUG}

# Extra include paths for compiling.
INCLUDES=-I./ -I${PROJECT_INCLUDE}/

# Extra flags for shared and static linking.
LDFLAGS_SHARED=-L${PROJECT_LIB} -lnnp
LDFLAGS_STATIC=${PROJECT_LIB}/libnnp.a


###############################################################################
# BUILD DETAILS
###############################################################################
OBJ=$(subst .cpp,.o,$(shell ls *.cpp))


###############################################################################
# RULES
###############################################################################
.PHONY: shared static clean

shared: ${OBJ}
	${CC} ${CFLAGS} ${DEBUG} ${OPTIONS} -o ${EXE} $+ ${LDFLAGS_SHARED}
	cp ${PWD}/${EXE} ${PROJECT_BIN}/

static: ${OBJ}
	${CC} ${CFLAGS} ${DEBUG} ${OPTIONS} -o ${EXE} $+ ${LDFLAGS_STATIC}
	cp ${PWD}/${EXE} ${PROJECT_BIN}/

%.o: %.cpp
	${CC} ${CFLAGS} ${DEBUG} ${INCLUDES} ${OPTIONS} -o $@ -c $<

clean:
	$(RM) *.o ${EXE}
	$(RM) ${PROJECT_BIN}/${EXE}
