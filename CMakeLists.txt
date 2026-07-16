cmake_minimum_required(VERSION 3.20)
project(QuantCore VERSION 1.0.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(QUANTCORE_BUILD_TESTS      "Build unit tests"         ON)
option(QUANTCORE_BUILD_BENCHMARKS "Build benchmarks"         ON)
option(QUANTCORE_BUILD_EXAMPLES   "Build example programs"   ON)
option(QUANTCORE_ENABLE_AVX512    "Enable AVX-512 intrinsics" ON)
option(QUANTCORE_ENABLE_AVX2      "Enable AVX2 intrinsics"   ON)
option(QUANTCORE_ENABLE_SSE42     "Enable SSE4.2 intrinsics" ON)
option(QUANTCORE_ENABLE_FAST_MATH "Enable -ffast-math"       OFF)
option(QUANTCORE_ENABLE_PARQUET   "Enable Parquet I/O via Apache Arrow" ON)
option(QUANTCORE_ENABLE_HDF5      "Enable HDF5 I/O support"  ON)

include(cmake/CompilerSettings.cmake)

add_subdirectory(quantcore)

if(QUANTCORE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(QUANTCORE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(QUANTCORE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
