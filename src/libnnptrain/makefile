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
# Possible modes are "static", "shared" and "test".
MODE=static

# Include global (project-wide) settings.
include $(PROJECT_DIR)/src/makefile.$(COMP)


###############################################################################
# LIBRARY NAME
###############################################################################
NAME=nnptrain
LIB=lib$(NAME)


###############################################################################
# PROJECT COMPILER AND FLAGS
###############################################################################
CC=$(PROJECT_CC)
MPICC=$(PROJECT_MPICC)
CFLAGS=$(PROJECT_CFLAGS) $(PROJECT_CFLAGS_MPI)
OPTIONS+=$(PROJECT_OPTIONS)
DEBUG=$(PROJECT_DEBUG)
ifeq ($(MODE), test)
DEBUG+= $(PROJECT_TEST)
endif
AR=$(PROJECT_AR)
ARFLAGS=$(PROJECT_ARFLAGS)

# Extra include paths for compiling.
INCLUDES=-I./ -I$(PROJECT_INCLUDE)/ -I$(PROJECT_GSL) -I$(PROJECT_EIGEN)


###############################################################################
# DEFINE SOURCES AND HEADERS
###############################################################################
SRC=$(wildcard *.cpp)
HEADERS=$(wildcard *.h)
OBJ=$(SRC:.cpp=.o)


###############################################################################
# RULES
###############################################################################
.PHONY: all headers clean rebuild

ifeq ($(MODE), shared)
all: headers $(LIB).so
else
all: headers $(LIB).a
endif

headers:
	cp $(HEADERS) $(PROJECT_INCLUDE)

$(LIB).so: CFLAGS+= -fPIC
$(LIB).so: $(OBJ)
	$(MPICC) $(CFLAGS) $(DEBUG) $(OPTIONS) -o $@ $+ -shared
	cp $@ $(PROJECT_LIB)

$(LIB).a: $(OBJ)
	$(AR) $(ARFLAGS) $@ $+
	cp $@ $(PROJECT_LIB)

%.o: %.cpp
	$(MPICC) $(CFLAGS) $(DEBUG) $(INCLUDES) $(OPTIONS) -o $@ -c $<

clean: clean-headers
	$(RM) $(OBJ)
	$(RM) *.gcno *.gcda $(LIB).so $(LIB).a
	$(RM) $(PROJECT_LIB)/$(LIB).so $(PROJECT_LIB)/$(LIB).a

clean-headers:
	cd $(PROJECT_INCLUDE) && $(RM) $(HEADERS)

rebuild: clean all
