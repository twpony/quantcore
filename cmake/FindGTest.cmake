# FindGTest.cmake — locate Google Test or fetch it via FetchContent

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest not found — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()
