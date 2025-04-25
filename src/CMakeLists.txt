###################################################################################################
# CMake Build System for n2p2
###################################################################################################

###################################################################################################
# Project & CMake Information
cmake_minimum_required(VERSION 3.16)    # Same required version as LAMMPS (no extra dependencies)
project(n2p2 VERSION 2.2.0)
###################################################################################################

###################################################################################################
# Git version info for project
execute_process(COMMAND git describe --tags --always
        OUTPUT_VARIABLE GIT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git rev-parse HEAD
        OUTPUT_VARIABLE GIT_REV OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git rev-parse --abbrev-ref HEAD
        OUTPUT_VARIABLE GIT_BRANCH OUTPUT_STRIP_TRAILING_WHITESPACE)
###################################################################################################

###################################################################################################
# Set C++ Standard
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
endif()
if(CMAKE_CXX_STANDARD LESS 11)
    message(FATAL_ERROR "C++ standard must be set to at least 11")
endif()

# Manually append the standard flag to CMAKE_CXX_FLAGS for external Makefile usage
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
endif()
###################################################################################################

###################################################################################################
# Project Path Variables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(PROJECT_INCLUDE ${CMAKE_BINARY_DIR}/../include)
set(CMAKE_SHARED_LIBRARY_SUFFIX .so)
###################################################################################################

###################################################################################################
# Document CLI Option Defaults
option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)
option(NDEBUG "Disable all C++ asserts (also Eigen debugging)." OFF)

# N2P2 Compile Options
option(N2P2_NO_SF_GROUPS "Disable Symmetry function groups (NOT RECOMMENDED)" OFF)
option(N2P2_FULL_SFD_MEMORY "Store zero symmetry function derivatives" OFF)
option(N2P2_NO_SF_CACHE "Do not use symmetry function cache." OFF)
option(N2P2_NO_ASYM_POLY "Disable asymmetric polynomial symmetry functions." OFF)
option(N2P2_NO_TIME "Build with dummy Stopwatch class." OFF)
option(N2P2_NO_NEIGH_CHECK "Disable check for low number of neighbors." OFF)
option(N2P2_NO_MPI "Compile without MPI support." OFF)

# Which interfaces to build
option(N2P2_IF_LAMMPS "Build LAMMPS interface into libnnpif" ON)
option(N2P2_IF_CABANA "Build CabanaMD interface into libnnpif" ON)

# Eigen Optimization Options
option(EIGEN_DONT_PARALLELIZE "Disable Eigen multi threading." ON)
option(EIGEN_USE_BLAS "Use BLAS together with Eigen." ON)
option(EIGEN_USE_MKL_ALL "Use Intel MKL together with Eigen." OFF)
###################################################################################################

###################################################################################################
# Set N2P2 Preprocessor Options Variable (Used in subdirectory CMakeLists)
if(N2P2_NO_SF_GROUPS)
    set(N2P2_PROJECT_OPTIONS "${N2P2_PROJECT_OPTIONS} -DN2P2_NO_SF_GROUPS")
endif(N2P2_NO_SF_GROUPS)

if(N2P2_FULL_SFD_MEMORY)
    set(N2P2_PROJECT_OPTIONS "${N2P2_PROJECT_OPTIONS} -DN2P2_FULL_SFD_MEMORY")
endif(N2P2_FULL_SFD_MEMORY)

if(N2P2_NO_SF_CACHE)
    set(N2P2_PROJECT_OPTIONS "${N2P2_PROJECT_OPTIONS} -DN2P2_NO_SF_CACHE")
endif(N2P2_NO_SF_CACHE)

if(N2P2_NO_ASYM_POLY)
    set(N2P2_PROJECT_OPTIONS "${N2P2_PROJECT_OPTIONS} -DN2P2_NO_ASYM_POLY")
endif(N2P2_NO_ASYM_POLY)

if(N2P2_NO_TIME)
    set(N2P2_PROJECT_OPTIONS "${N2P2_PROJECT_OPTIONS} -DN2P2_NO_TIME")
endif(N2P2_NO_TIME)

if(N2P2_NO_NEIGH_CHECK)
    set(N2P2_PROJECT_OPTIONS "${N2P2_PROJECT_OPTIONS} -DN2P2_NO_NEIGH_CHECK")
endif(N2P2_NO_NEIGH_CHECK)

if(N2P2_NO_MPI)
    set(N2P2_PROJECT_OPTIONS "${N2P2_PROJECT_OPTIONS} -DN2P2_NO_MPI")
else()
    set(CMAKE_C_COMPILER "mpicc")
    set(CMAKE_CXX_COMPILER "mpicxx")
endif(N2P2_NO_MPI)

if(EIGEN_DONT_PARALLELIZE)
    set(EIGEN_PROJECT_OPTIONS "${EIGEN_PROJECT_OPTIONS} -DEIGEN_DONT_PARALLELIZE")
endif(EIGEN_DONT_PARALLELIZE)

if(EIGEN_USE_BLAS)
    set(EIGEN_PROJECT_OPTIONS "${EIGEN_PROJECT_OPTIONS} -DEIGEN_USE_BLAS")
endif(EIGEN_USE_BLAS)

if(EIGEN_USE_MKL_ALL)
    set(EIGEN_PROJECT_OPTIONS "${EIGEN_PROJECT_OPTIONS} -DEIGEN_USE_MKL_ALL")
endif(EIGEN_USE_MKL_ALL)

set(PROJECT_OPTIONS "${PROJECT_OPTIONS} ${N2P2_PROJECT_OPTIONS} ${EIGEN_PROJECT_OPTIONS}")

if(NDEBUG)
    set(PROJECT_OPTIONS "${PROJECT_OPTIONS} -DNDEBUG")
endif(NDEBUG)

if(N2P2_IF_LAMMPS)
    list(APPEND INTERFACES "${INTERFACES} LAMMPS")
endif(N2P2_IF_LAMMPS)

if(N2P2_IF_CABANA)
    list(APPEND INTERFACES "${INTERFACES} CabanaMD")
endif(N2P2_IF_CABANA)
###################################################################################################

###################################################################################################
find_package(MPI REQUIRED)
find_package(GSL REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
link_directories(${GSL_LIBRARY_DIRS} ${MPI_LIBRARY_DIRS})
list(GET GSL_LIBRARIES 0 FIRST_GSL_LIBRARY)
get_filename_component(GSL_LIBRARY_DIR "${FIRST_GSL_LIBRARY}" DIRECTORY)

# Go into each subdirectory
add_subdirectory(libnnp)
add_subdirectory(libnnptrain)
add_subdirectory(application)
add_subdirectory(libnnpif)
add_subdirectory(interface)
###################################################################################################