###################################################################################################
# CMake Build System for n2p2/libnnptrain
###################################################################################################

###################################################################################################
# Set Source Files
file(GLOB SRC_FILES "*.cpp")
file(GLOB HDR_FILES "*.h")
###################################################################################################

###################################################################################################
# Specify Build Targets
add_library(nnptrain ${SRC_FILES})
add_compile_options(${PROJECT_OPTIONS})
install(FILES ${HDR_FILES} DESTINATION ${PROJECT_INCLUDE})   # Headers copied at install time
###################################################################################################

###################################################################################################
# Find Required Package dependencies
include_directories(${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/libnnp ${GSL_INCLUDE_DIRS})
target_link_libraries(nnptrain MPI::MPI_CXX ${GSL_LIBRARIES} Eigen3::Eigen nnp)
###################################################################################################