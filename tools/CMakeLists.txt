# Tools — standalone utilities for the QuantCore project
# These are NOT part of the main library; they link against quantcore directly.

add_executable(generate_factor generate_factor.cpp)
target_link_libraries(generate_factor PRIVATE quantcore)
target_include_directories(generate_factor PRIVATE ${PROJECT_SOURCE_DIR})
