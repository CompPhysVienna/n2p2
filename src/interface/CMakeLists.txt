###################################################################################################
# CMake Build System for n2p2 LAMMPS interface
###################################################################################################

###################################################################################################
# Skip if not building LAMMPS or LAMMPS interface
if(NOT N2P2_IF_LAMMPS )
    return()
endif(NOT N2P2_IF_LAMMPS )
###################################################################################################

###################################################################################################
# LAMMPS Specific Options
# Build Internal Version or External (sets name of the lammps package)
option(LAMMPS_INTERNAL
        "Build as internal package. Uses name nnp vs. hdnnp to avoid conflict with release version."
        OFF)
if(LAMMPS_INTERNAL)
    set(LAMMPS_PACKAGE_NAME nnp)
else()
    set(LAMMPS_PACKAGE_NAME hdnnp)
endif(LAMMPS_INTERNAL)

# Download & Build own LAMMPS executable
option(BUILD_LAMMPS
        "Build minimal LAMMPS executable. (Either Specify DOWNLOAD_LAMMPS=ON or LAMMPS_PATH)" OFF)

if (NOT LAMMPS_PATH)
    option(DOWNLOAD_LAMMPS "Download LAMMPS. This supersedes LAMMPS_PATH variable." OFF)
else()
    option(DOWNLOAD_LAMMPS "Download LAMMPS. This supersedes LAMMPS_PATH variable." OFF)
endif (NOT LAMMPS_PATH)

# Specify Default LAMMPS Version
if (NOT LAMMPS_VERSION)
    set(LAMMPS_VERSION "lammps-29Aug2024")
endif (NOT LAMMPS_VERSION)
###################################################################################################

###################################################################################################
# Configure makefiles and extra cmake file to add N2P2 options to LAMMPS build
message("Creating LAMMPS configuration files.")
configure_file(LAMMPS/Makefile.lammps-extra.config.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Makefile.lammps-extra)
configure_file(LAMMPS/lammps-extra.cmake.config.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lammps-extra.cmake)
###################################################################################################

###################################################################################################
# Download LAMMPS if needed
if(DOWNLOAD_LAMMPS)
    if (LAMMPS_PATH)
        message(WARNING "Specified both LAMMPS_PATH and DOWNLOAD_LAMMPS=ON. Ignoring LAMMPS_PATH.")
    endif (LAMMPS_PATH)

    message("Downloading LAMMPS to ${CMAKE_BINARY_DIR}/_deps/lammps-nnp-src/")
    include(FetchContent)
    FetchContent_Declare(lammps-hdnnp
            URL https://download.lammps.org/tars/${LAMMPS_VERSION}.tar.gz)
    FetchContent_MakeAvailable(lammps-hdnnp)

    set(LAMMPS_PATH "${CMAKE_BINARY_DIR}/_deps/lammps-hdnnp-src/")
endif(DOWNLOAD_LAMMPS)
###################################################################################################

if(BUILD_LAMMPS)
    ###################################################################################################
    # Symlink to n2p2 Library from LAMMPS
    add_custom_target(link_n2p2 ALL
            #COMMAND ${CMAKE_COMMAND} -E make_directory ${LAMMPS_PATH}lib/nnp
            COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} ${LAMMPS_PATH}/lib/hdnnp/liblink
            COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_INCLUDE} ${LAMMPS_PATH}/lib/hdnnp/includelink
            COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/Makefile.lammps-extra ${LAMMPS_PATH}/lib/hdnnp/Makefile.lammps
            COMMENT "Creating Symlink from LAMMPS to n2p2.")
    ###################################################################################################

    ###################################################################################################
    # Copy Source Code to LAMMPS
    add_custom_target(update_lammps ALL
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/interface/LAMMPS/src/ML-HDNNP/ ${LAMMPS_PATH}src/ML-HDNNP
            COMMENT "Updating LAMMPS ML-HDNNP package.")
    ###################################################################################################

    ###################################################################################################
    # Patch LAMMPS makefile.mpi
    add_custom_target(patch_lammps_makefile ALL
            COMMAND sed -i.bak "s,^CC .*$$,CC      =       ${MPI_CXX_COMPILER}," ${LAMMPS_PATH}/src/MAKE/Makefile.mpi
            COMMAND sed -i.bak "s,^CCFLAGS .*$$,CCFLAGS =       ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MPI} -I${GSL_INCLUDE_DIRS} -I${EIGEN3_INCLUDE_DIR}," ${LAMMPS_PATH}/src/MAKE/Makefile.mpi
            COMMAND sed -i.bak "s,^LINK .*$$,LINK =          ${MPI_CXX_COMPILER} ," ${LAMMPS_PATH}/src/MAKE/Makefile.mpi
            COMMAND sed -i.bak "s,^LINKFLAGS .*$$,LINKFLAGS =     ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_MPI} -L${GSL_LIBRARY_DIR}," ${LAMMPS_PATH}/src/MAKE/Makefile.mpi
            COMMENT "Patching LAMMPS makefile.mpi")
    ###################################################################################################

    ###################################################################################################
    # Build a minimal LAMMPS Executable (uses LAMMPS make build system)
    add_custom_target(build_lammps ALL
            COMMAND ${CMAKE_COMMAND} -E chdir ${LAMMPS_PATH}/src/ make yes-ml-hdnnp
            COMMAND ${CMAKE_COMMAND} -E chdir ${LAMMPS_PATH}/src/ make yes-kspace
            COMMAND ${CMAKE_COMMAND} -E chdir ${LAMMPS_PATH}/src/ make yes-molecule
            COMMAND ${CMAKE_COMMAND} -E chdir ${LAMMPS_PATH}/src/ make mpi
            DEPENDS "${LAMMPS_PATH}/src/ML-HDNNP"
            COMMENT "Building minimal LAMMPS Distribution. Version: ${LAMMPS_VERSION}")
    install(FILES "${LAMMPS_PATH}/src/lmp_mpi" DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    add_dependencies(build_lammps link_n2p2 update_lammps patch_lammps_makefile nnpif)
    ###################################################################################################
endif(BUILD_LAMMPS)