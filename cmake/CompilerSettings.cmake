# CompilerSettings.cmake — centralized compiler options for QuantCore

# Require at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Common warning flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion)
elseif(MSVC)
    add_compile_options(/W4)
endif()

# Debug configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR NOT CMAKE_BUILD_TYPE)
    message(STATUS "QuantCore: Debug build — assertions + cross-validation enabled")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-g -O0)
        # Optional sanitizers (uncomment to enable)
        # add_compile_options(-fsanitize=address,undefined)
        # add_link_options(-fsanitize=address,undefined)
    endif()
    add_compile_definitions(QUANTCORE_DEBUG=1)
endif()

# Release configuration (default — preserves numerical correctness)
if(CMAKE_BUILD_TYPE STRELEASE "Release" OR NOT CMAKE_BUILD_TYPE)
    message(STATUS "QuantCore: Release build — O3 + native arch (no -ffast-math)")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-O3 -march=native -DNDEBUG)
    elseif(MSVC)
        add_compile_options(/O2 /arch:AVX2)
    endif()
endif()

# RelWithDebInfo configuration
if(CMAKE_BUILD_TYPE STRELEASE "RelWithDebInfo")
    message(STATUS "QuantCore: RelWithDebInfo build — O2 + debug symbols")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-O2 -g -march=native)
    endif()
endif()

# Optional fast-math warning
if(QUANTCORE_ENABLE_FAST_MATH)
    message(WARNING
        "*** -ffast-math is ENABLED. "
        "This may break IEEE 754 compliance and change numerical results. "
        "Use with caution in production quantitative finance workloads. ***")
endif()
