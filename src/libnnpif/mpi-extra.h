// Copyright 2018 Andreas Singraber (University of Vienna)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <mpi.h>
// Unfortunately this would only work with C++11 standard, so it's not used.
//#include <cstdint>
//#include <climits>
//
//#if SIZE_MAX == UCHAR_MAX
//   #define MPI_SIZE_T MPI_UNSIGNED_CHAR
//#elif SIZE_MAX == USHRT_MAX
//   #define MPI_SIZE_T MPI_UNSIGNED_SHORT
//#elif SIZE_MAX == UINT_MAX
//   #define MPI_SIZE_T MPI_UNSIGNED
//#elif SIZE_MAX == ULONG_MAX
//   #define MPI_SIZE_T MPI_UNSIGNED_LONG
//#elif SIZE_MAX == ULLONG_MAX
//   #define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
//#else
//   #error "No matching MPI data type for size_t."
//#endif

// Just a guess...
#pragma message ( "CAUTION: Please check if MPI_SIZE_T is correctly defined here!\n" )
#define MPI_SIZE_T MPI_UNSIGNED_LONG
