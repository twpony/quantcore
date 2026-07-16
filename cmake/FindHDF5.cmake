# FindHDF5.cmake — locate HDF5 C++ library
#
# Exported variables:
#   HDF5_FOUND          — TRUE if HDF5 found
#   HDF5_INCLUDE_DIRS   — HDF5 include directories
#   HDF5_LIBRARIES      — HDF5 C and C++ libraries for linking

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(PC_HDF5 QUIET hdf5)
endif()

# Find HDF5 include directory
find_path(HDF5_INCLUDE_DIR
    NAMES hdf5.h H5Cpp.h
    HINTS
        ${PC_HDF5_INCLUDE_DIRS}
        ${CONDA_PREFIX}/include
        /usr/include
        /usr/local/include
)

# Find HDF5 C library
find_library(HDF5_C_LIBRARY
    NAMES hdf5
    HINTS
        ${PC_HDF5_LIBRARY_DIRS}
        ${CONDA_PREFIX}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
)

# Find HDF5 C++ library (preferred over pure C API for type safety)
find_library(HDF5_CPP_LIBRARY
    NAMES hdf5_cpp
    HINTS
        ${PC_HDF5_LIBRARY_DIRS}
        ${CONDA_PREFIX}/lib
        /usr/lib/x86_64-linux-gnu
        /usr/lib
        /usr/local/lib
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(HDF5
    REQUIRED_VARS HDF5_C_LIBRARY HDF5_INCLUDE_DIR
)

if(HDF5_FOUND)
    set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})

    if(HDF5_CPP_LIBRARY)
        set(HDF5_LIBRARIES ${HDF5_CPP_LIBRARY} ${HDF5_C_LIBRARY})
    else()
        set(HDF5_LIBRARIES ${HDF5_C_LIBRARY})
    endif()

    if(NOT TARGET HDF5::HDF5)
        add_library(HDF5::HDF5 UNKNOWN IMPORTED)
        set_target_properties(HDF5::HDF5 PROPERTIES
            IMPORTED_LOCATION "${HDF5_C_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}"
        )
    endif()

    if(HDF5_CPP_LIBRARY AND NOT TARGET HDF5::HDF5_CPP)
        add_library(HDF5::HDF5_CPP UNKNOWN IMPORTED)
        set_target_properties(HDF5::HDF5_CPP PROPERTIES
            IMPORTED_LOCATION "${HDF5_CPP_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${HDF5_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "HDF5::HDF5"
        )
    endif()
endif()

mark_as_advanced(HDF5_INCLUDE_DIR HDF5_C_LIBRARY HDF5_CPP_LIBRARY)
