###################################################################################################
# CMake Build System for n2p2/libnnpif
###################################################################################################

###################################################################################################
# Set Source Files
file(GLOB LAMMPS_SRC_FILES "LAMMPS/*.cpp")
file(GLOB LAMMPS_HDR_FILES "LAMMPS/*.h")
file(GLOB CABANA_SRC_FILES "CabanaMD/*.cpp")
file(GLOB CABANA_HDR_FILES "CabanaMD/*.h")

if(N2P2_IF_LAMMPS)
    list(APPEND SRC_FILES ${LAMMPS_SRC_FILES})
    list(APPEND HDR_FILES ${LAMMPS_HDR_FILES})
endif(N2P2_IF_LAMMPS)

if(N2P2_IF_CABANA)
    list(APPEND SRC_FILES ${CABANA_SRC_FILES})
    list(APPEND HDR_FILES ${CABANA_HDR_FILES})
endif(N2P2_IF_CABANA)
###################################################################################################

###################################################################################################
# Specify Build Targets
add_library(nnpif ${SRC_FILES})
add_compile_options(${PROJECT_OPTIONS})

# Headers need to be available at compile time to enable LAMMPS build
add_custom_target(install-nnpif-headers ALL
        COMMAND ${CMAKE_COMMAND} -E copy ${LAMMPS_HDR_FILES} ${PROJECT_INCLUDE}
        COMMAND ${CMAKE_COMMAND} -E copy ${CABANA_HDR_FILES} ${PROJECT_INCLUDE}
        COMMENT "Installing libnnpif headers.")
#install(FILES ${HDR_FILES} DESTINATION ${PROJECT_INCLUDE})   # Headers copied at install time
###################################################################################################

###################################################################################################
# Find Required Package dependencies
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/libnnp)
target_link_libraries(nnpif MPI::MPI_CXX nnp)
###################################################################################################