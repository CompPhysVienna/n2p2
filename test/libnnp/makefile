#!/bin/make -f

###############################################################################
# PROJECT PATHS
###############################################################################
PROJECT_DIR=../../
PROJECT_BIN=${PROJECT_DIR}/bin
PROJECT_INCLUDE=${PROJECT_DIR}/include
PROJECT_LIB=${PROJECT_DIR}/lib
PROJECT_BUILD=${PROJECT_DIR}/build


###############################################################################
# PROJECT MAKEFILE
###############################################################################
# Default compiler, may be overridden by master makefile.
COMP=gnu

# Include global (project-wide) settings.
include ${PROJECT_DIR}/src/makefile.${COMP}


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
DEBUG=${PROJECT_DEBUG} ${PROJECT_TEST}

# Extra include paths for compiling.
# For correct coverage need use the headers from the build directory!
INCLUDES=-I./ -I${PROJECT_BUILD}/static/

# Extra flags for shared and static linking.
LDFLAGS=${PROJECT_LIB}/libnnp.a -lboost_system -lboost_unit_test_framework -lboost_filesystem


###############################################################################
# BUILD DETAILS
###############################################################################
SRC=$(wildcard test_*.cpp)
EXE=$(SRC:%.cpp=%)


###############################################################################
# RULES
###############################################################################
.PHONY: all test coverage clean clean-coverage

all: ${EXE}

coverage:
	lcov -c -i --no-external        -d ${PROJECT_BUILD}/static/ -o tests.base
	lcov -c    --no-external        -d ${PROJECT_BUILD}/static/ -o tests.run
	lcov -a tests.base -a tests.run -d ${PROJECT_BUILD}/static/ -o tests.info
	genhtml -o htmlcov tests.info

%: %.cpp
	${CC} ${CFLAGS} ${DEBUG} ${INCLUDES} ${OPTIONS} -o $@ $< ${LDFLAGS}
	./$@ --color_output=yes --log_level=all

clean: clean-coverage
	$(RM) *.o *.gcno ${EXE}

clean-coverage:
	$(RM) ${PROJECT_BUILD}/static/*.gcda *.gcda *.base *.run *.info
	$(RM) -r htmlcov
