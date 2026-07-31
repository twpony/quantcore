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
add_subdirectory(factors)

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

add_subdirectory(tools)

# ============================================================
# Install & packaging
# ============================================================

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install the exported targets file
install(EXPORT QuantCoreTargets
    FILE QuantCoreTargets.cmake
    NAMESPACE quantcore::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QuantCore
)

# Generate the version file for find_package(QuantCore VERSION ...)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/QuantCoreConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/QuantCoreConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QuantCore
)

# Generate the main config file from template
configure_package_config_file(
    cmake/QuantCoreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/QuantCoreConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QuantCore
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/QuantCoreConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/QuantCore
)
