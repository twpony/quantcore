# FindGBenchmark.cmake — locate Google Benchmark or fetch it via FetchContent

find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found — fetching via FetchContent")
    include(FetchContent)
    # Disable GBenchmark's own tests to avoid CMake compatibility issues
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.1
    )
    FetchContent_MakeAvailable(googlebenchmark)
endif()
