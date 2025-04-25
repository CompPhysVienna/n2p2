# CMake Cache File Setting Defaults for Mac LLVM toolchain

#set(CMAKE_CXX_COMPILER "mpic++" CACHE STRING "" FORCE)
#set(MPI_CXX "mpic++" CACHE STRING "" FORCE)
#set(MPI_CXX_COMPILER "mpic++" CACHE STRING "" FORCE)

set(CMAKE_CXX_FLAGS "-O3 -march=native -std=c++11" CACHE STRING "" FORCE)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS deployment target")
set(CMAKE_CXX_FLAGS_DEBUG "-g -pedantic-errors -Wall -Wextra" CACHE STRING "" FORCE)
set(CMAKE_AR "ar" CACHE STRING "" FORCE)