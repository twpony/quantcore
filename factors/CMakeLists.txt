# Factors — built as a shared library for external consumption
# Each factor is a compilation unit; link against libquantcorefactors.so

add_library(quantcorefactors SHARED
    alpha_0001.cpp
    alpha_0002.cpp
    alpha_0003.cpp
    alpha_0004.cpp
    alpha_0005.cpp
)

target_link_libraries(quantcorefactors PRIVATE quantcore)

# Export factors/ as a public include directory so consumers can
#   #include "factors/alpha_0001.h"
target_include_directories(quantcorefactors PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# ============================================================
# Install rules
# ============================================================

include(GNUInstallDirs)

install(TARGETS quantcorefactors
    EXPORT QuantCoreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install factor headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/factors
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.cpp" EXCLUDE
    PATTERN "CMakeLists.txt" EXCLUDE
)
