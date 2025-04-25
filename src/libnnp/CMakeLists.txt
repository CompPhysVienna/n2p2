###################################################################################################
# CMake Build System for n2p2/libnnp
###################################################################################################

###################################################################################################
# Set Source Files
file(GLOB SRC_FILES "*.cpp")
file(GLOB HDR_FILES "*.h")
###################################################################################################

###################################################################################################
# Specify Build Targets
add_library(nnp ${SRC_FILES})
add_compile_options(${PROJECT_OPTIONS})
configure_file(version.config.h ${CMAKE_CURRENT_SOURCE_DIR}/version.h)
install(FILES ${HDR_FILES} DESTINATION ${PROJECT_INCLUDE})   # Headers copied at install time
###################################################################################################

###################################################################################################
# Find Required Package dependencies
target_link_libraries(nnp PUBLIC MPI::MPI_CXX Eigen3::Eigen)
include_directories(${MPI_INCLUDE_PATH})
###################################################################################################