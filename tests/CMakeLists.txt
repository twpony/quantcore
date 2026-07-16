# Tests CMakeLists.txt

include(../cmake/FindGTest.cmake)

# Unit tests
add_executable(test_column       unit/test_column.cpp)
add_executable(test_market_data  unit/test_market_data.cpp)
add_executable(test_unary_ops    unit/test_unary_ops.cpp)
add_executable(test_binary_ops   unit/test_binary_ops.cpp)
add_executable(test_expression   unit/test_expression.cpp)
add_executable(test_buffer_pool  unit/test_buffer_pool.cpp)
add_executable(test_simd_cross_validate unit/test_simd_cross_validate.cpp)
add_executable(test_registry     unit/test_registry.cpp)
add_executable(test_csv_reader   unit/test_csv_reader.cpp)

# Integration tests
add_executable(test_full_pipeline integration/test_full_pipeline.cpp)
add_executable(test_csv_roundtrip integration/test_csv_roundtrip.cpp)
add_executable(test_parquet_reader integration/test_parquet_reader.cpp)
add_executable(test_hdf5_reader integration/test_hdf5_reader.cpp)

# Regression tests
add_executable(golden_factors    regression/golden_factors.cpp)

# Link all test targets
set(TEST_TARGETS
    test_column test_market_data test_unary_ops test_binary_ops
    test_expression test_buffer_pool test_simd_cross_validate
    test_registry test_csv_reader
    test_full_pipeline test_csv_roundtrip
    test_parquet_reader test_hdf5_reader
    golden_factors
)

foreach(target ${TEST_TARGETS})
    target_link_libraries(${target} PRIVATE quantcore GTest::gtest GTest::gtest_main)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${target} COMMAND ${target})
endforeach()
