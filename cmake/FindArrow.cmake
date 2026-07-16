# FindArrow.cmake — locate Apache Arrow C++ and Parquet libraries
#
# Exported variables:
#   ARROW_FOUND         — TRUE if Arrow found
#   PARQUET_FOUND       — TRUE if Parquet C++ found
#   ARROW_INCLUDE_DIRS  — Arrow include directories
#   ARROW_LIBRARIES     — Arrow libraries for linking
#   PARQUET_LIBRARIES   — Parquet C++ libraries for linking

# Try pkg-config first
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PC_ARROW QUIET arrow)
    pkg_check_modules(PC_PARQUET QUIET parquet)
endif()

# Find Arrow include directory
find_path(ARROW_INCLUDE_DIR
    NAMES arrow/api.h
    HINTS
        ${PC_ARROW_INCLUDE_DIRS}
        ${CONDA_PREFIX}/include
        /usr/include
        /usr/local/include
    PATH_SUFFIXES arrow
)

# Find Arrow library
find_library(ARROW_LIBRARY
    NAMES arrow
    HINTS
        ${PC_ARROW_LIBRARY_DIRS}
        ${CONDA_PREFIX}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
)

# Find Parquet library
find_library(PARQUET_LIBRARY
    NAMES parquet
    HINTS
        ${PC_PARQUET_LIBRARY_DIRS}
        ${CONDA_PREFIX}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
)

# Find Parquet include directory (usually alongside Arrow)
find_path(PARQUET_INCLUDE_DIR
    NAMES parquet/api/reader.h
    HINTS
        ${PC_PARQUET_INCLUDE_DIRS}
        ${CONDA_PREFIX}/include
        /usr/include
        /usr/local/include
)

# Handle standard find_package arguments
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(Arrow
    REQUIRED_VARS ARROW_LIBRARY ARROW_INCLUDE_DIR
)

if(Arrow_FOUND AND NOT TARGET Arrow::Arrow)
    add_library(Arrow::Arrow UNKNOWN IMPORTED)
    set_target_properties(Arrow::Arrow PROPERTIES
        IMPORTED_LOCATION "${ARROW_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ARROW_INCLUDE_DIR}"
    )
endif()

if(PARQUET_LIBRARY AND PARQUET_INCLUDE_DIR)
    set(PARQUET_FOUND TRUE)
    if(NOT TARGET Parquet::Parquet)
        add_library(Parquet::Parquet UNKNOWN IMPORTED)
        set_target_properties(Parquet::Parquet PROPERTIES
            IMPORTED_LOCATION "${PARQUET_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PARQUET_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "Arrow::Arrow"
        )
    endif()
endif()

# Set output variables
set(ARROW_INCLUDE_DIRS ${ARROW_INCLUDE_DIR})
set(ARROW_LIBRARIES    ${ARROW_LIBRARY})
set(PARQUET_LIBRARIES  ${PARQUET_LIBRARY})

mark_as_advanced(ARROW_INCLUDE_DIR ARROW_LIBRARY PARQUET_LIBRARY PARQUET_INCLUDE_DIR)
