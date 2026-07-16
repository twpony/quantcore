# FindGBenchmark.cmake — locate Google Benchmark or fetch it via FetchContent

find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.8.3
    )
    FetchContent_MakeAvailable(googlebenchmark)
endif()
