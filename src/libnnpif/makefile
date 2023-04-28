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
# INTERFACE LIBRARY OPTIONS
###############################################################################

# Default interfaces compiled into libnnpif, may be overridden by master
# makefile or command line. Provide a space-separated list of the following
# available options:
#    * LAMMPS
#    * CabanaMD
INTERFACES=LAMMPS CabanaMD


###############################################################################
# LIBRARY NAME
###############################################################################
NAME=nnpif
LIB=lib$(NAME)


###############################################################################
# PROJECT COMPILER AND FLAGS
###############################################################################
CC=$(PROJECT_CC)
ifneq (,$(findstring N2P2_NO_MPI,${PROJECT_OPTIONS}))
MPICC=${PROJECT_CC}
CFLAGS=${PROJECT_CFLAGS}
else
MPICC=${PROJECT_MPICC}
CFLAGS=${PROJECT_CFLAGS} ${PROJECT_CFLAGS_MPI}
endif
OPTIONS+=$(PROJECT_OPTIONS)
DEBUG=$(PROJECT_DEBUG)
ifeq ($(MODE), test)
DEBUG+= $(PROJECT_TEST)
endif
AR=$(PROJECT_AR)
ARFLAGS=$(PROJECT_ARFLAGS)

# Extra include paths for compiling.
INCLUDES=-I./ -I$(PROJECT_INCLUDE)/ -I$(PROJECT_EIGEN)


###############################################################################
# DEFINE SOURCES AND HEADERS
###############################################################################

$(info The following interfaces were selected: $(INTERFACES))
IF=LAMMPS
ifneq ($(filter $(IF),$(INTERFACES)),)
SRC:=$(SRC) $(wildcard $(IF)/*.cpp)
HEADERS:=$(HEADERS) $(wildcard $(IF)/*.h)
endif

IF=CabanaMD
ifneq ($(filter $(IF),$(INTERFACES)),)
SRC:=$(SRC) $(wildcard $(IF)/*.cpp)
HEADERS:=$(HEADERS) $(wildcard $(IF)/*.h)
endif

OBJ=$(SRC:.cpp=.o)


###############################################################################
# RULES
###############################################################################
.PHONY: all \
	headers \
	lammps-prep \
	lammps-mf \
	lammps-mf-internal \
	lammps-cmake \
	clean rebuild

ifeq ($(MODE), shared)
all: headers lammps-prep $(LIB).so
else
all: headers lammps-prep $(LIB).a
endif

headers:
	cp $(HEADERS) $(PROJECT_INCLUDE)

LAMMPS_OPTIONS := $(filter -DN2P2_%,$(OPTIONS))
lammps-prep: lammps-mf lammps-cmake

lammps-mf:
	cp LAMMPS/Makefile.lammps $(PROJECT_LIB)/Makefile.lammps-extra
	@sed -i.bak -E "s,(hdnnp_SYSINC =).*,\1 $(LAMMPS_OPTIONS)," $(PROJECT_LIB)/Makefile.lammps-extra
	@rm $(PROJECT_LIB)/Makefile.lammps-extra.bak

lammps-mf-internal: lammps-mf
	sed -i.bak -E "s,hdnnp,nnp," $(PROJECT_LIB)/Makefile.lammps-extra
	@rm $(PROJECT_LIB)/Makefile.lammps-extra.bak

lammps-cmake:
	$(RM) -f $(PROJECT_LIB)/lammps-extra.cmake
	touch $(PROJECT_LIB)/lammps-extra.cmake
	for flag in $(LAMMPS_OPTIONS); do \
		echo "target_compile_definitions(lammps PRIVATE $${flag})" >> $(PROJECT_LIB)/lammps-extra.cmake; \
	done

$(LIB).so: CFLAGS+= -fPIC
$(LIB).so: $(OBJ)
	$(MPICC) $(CFLAGS) $(DEBUG) $(OPTIONS) -o $@ $+ -shared
	cp $@ $(PROJECT_LIB)

$(LIB).a: $(OBJ)
	$(AR) $(ARFLAGS) $@ $+
	cp $@ $(PROJECT_LIB)

%.o: %.cpp
	$(MPICC) $(CFLAGS) $(DEBUG) $(INCLUDES) $(OPTIONS) -o $@ -c $<

clean: clean-headers clean-lammps-prep
	$(RM) $(OBJ)
	$(RM) $(OBJ:.o=.gcno) $(OBJ:.o=.gcda) $(LIB).so $(LIB).a
	$(RM) $(PROJECT_LIB)/$(LIB).so $(PROJECT_LIB)/$(LIB).a

clean-headers:
	cd $(PROJECT_INCLUDE) && $(RM) $(notdir $(HEADERS))

clean-lammps-prep:
	$(RM) $(PROJECT_LIB)/Makefile.lammps-extra
	$(RM) $(PROJECT_LIB)/lammps-extra.cmake

rebuild: clean all
