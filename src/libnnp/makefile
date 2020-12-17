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
NAME=nnp
LIB=lib$(NAME)


###############################################################################
# PROJECT COMPILER AND FLAGS
###############################################################################
CC=$(PROJECT_CC)
MPICC=$(PROJECT_MPICC)
CFLAGS=$(PROJECT_CFLAGS)
OPTIONS+=$(PROJECT_OPTIONS)
DEBUG=$(PROJECT_DEBUG)
ifeq ($(MODE), test)
DEBUG+= $(PROJECT_TEST)
endif
AR=$(PROJECT_AR)
ARFLAGS=$(PROJECT_ARFLAGS)

# Extra include paths for compiling.
INCLUDES=-I./


###############################################################################
# DEFINE SOURCES AND HEADERS
###############################################################################
SRC=$(wildcard *.cpp)
HEADERS=$(wildcard *.h)
OBJ=$(SRC:.cpp=.o)


###############################################################################
# RULES
###############################################################################
.PHONY: all headers version clean rebuild

ifeq ($(MODE), shared)
all: headers $(LIB).so
else
all: headers $(LIB).a
endif

headers: version
	cp $(HEADERS) $(PROJECT_INCLUDE)

version:
	@$(eval GIT_VERSION = $(shell git describe --tags --always))
	@sed -i.bak -E "s/(NNP_GIT_VERSION) .*/\1 \"$(GIT_VERSION)\"/" version.h
	@$(eval GIT_REV = $(shell git rev-parse HEAD))
	@sed -i.bak -E "s/(NNP_GIT_REV) .*/\1 \"$(GIT_REV)\"/" version.h
	@$(eval GIT_BRANCH = $(shell git rev-parse --abbrev-ref HEAD))
	@sed -i.bak -E "s/(NNP_GIT_BRANCH) .*/\1 \"$(GIT_BRANCH)\"/" version.h
	@rm version.h.bak

$(LIB).so: CFLAGS+= -fPIC
$(LIB).so: $(OBJ)
	$(CC) $(CFLAGS) $(DEBUG) $(OPTIONS) -o $@ $+ -shared
	cp $@ $(PROJECT_LIB)

$(LIB).a: $(OBJ)
	$(AR) $(ARFLAGS) $@ $+
	cp $@ $(PROJECT_LIB)

%.o: %.cpp
	$(CC) $(CFLAGS) $(DEBUG) $(INCLUDES) $(OPTIONS) -o $@ -c $<

clean: clean-version clean-headers
	$(RM) $(OBJ)
	$(RM) *.gcno *.gcda $(LIB).so $(LIB).a
	$(RM) $(PROJECT_LIB)/$(LIB).so $(PROJECT_LIB)/$(LIB).a

clean-version:
	@sed -i.bak -E "s/(NNP_GIT_VERSION) .*/\1 \"\"/" version.h
	@sed -i.bak -E "s/(NNP_GIT_REV) .*/\1 \"\"/" version.h
	@sed -i.bak -E "s/(NNP_GIT_BRANCH) .*/\1 \"\"/" version.h
	@rm version.h.bak

clean-headers:
	cd $(PROJECT_INCLUDE) && $(RM) $(HEADERS)

rebuild: clean all
