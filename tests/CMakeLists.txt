# Tests CMakeLists.txt

include(../cmake/FindGTest.cmake)

# Unit tests
add_executable(test_column       unit/test_column.cpp)
add_executable(test_market_data  unit/test_market_data.cpp)
add_executable(test_expression   unit/test_expression.cpp)
add_executable(test_buffer_pool  unit/test_buffer_pool.cpp)
add_executable(test_engine       unit/test_engine.cpp)
add_executable(test_parser       unit/test_parser.cpp)
add_executable(test_factor_calculator unit/test_factor_calculator.cpp)
add_executable(test_panel_data     unit/test_panel_data.cpp)
add_executable(test_intermediate_column unit/test_intermediate_column.cpp)
add_executable(test_cross_section_workspace unit/test_cross_section_workspace.cpp)
add_executable(test_csv_loader    unit/test_csv_loader.cpp)
add_executable(test_alpha0001 factors/test_alpha_0001.cpp)
add_executable(test_alpha0002 factors/test_alpha_0002.cpp)
add_executable(test_alpha1001 factors/test_alpha_1001.cpp)
add_executable(test_alpha0003_0005 factors/test_alpha_0003_0005.cpp)
add_executable(test_simd_cross_validate unit/test_simd_cross_validate.cpp)
add_executable(test_registry     unit/test_registry.cpp)
add_executable(test_binary_ops   unit/test_binary_ops.cpp)
add_executable(test_sma20_bench   unit/test_sma20_bench.cpp)

# Operator tests — organized by operator family
# Unary operators
add_executable(test_op_abs       op/unary/test_abs.cpp)
add_executable(test_op_exp       op/unary/test_exp.cpp)
add_executable(test_op_log       op/unary/test_log.cpp)
add_executable(test_op_log2      op/unary/test_log2.cpp)
add_executable(test_op_sqrt      op/unary/test_sqrt.cpp)
add_executable(test_op_square    op/unary/test_square.cpp)
add_executable(test_op_inv       op/unary/test_inv.cpp)
add_executable(test_op_log10     op/unary/test_log10.cpp)
add_executable(test_op_sign      op/unary/test_sign.cpp)
add_executable(test_op_not       op/unary/test_not.cpp)
add_executable(test_op_rank      op/unary/test_rank.cpp)

# Binary operators
add_executable(test_op_add    op/binary/test_add.cpp)
add_executable(test_op_sub    op/binary/test_sub.cpp)
add_executable(test_op_mul    op/binary/test_mul.cpp)
add_executable(test_op_div    op/binary/test_div.cpp)
add_executable(test_op_eq    op/binary/test_eq.cpp)
add_executable(test_op_neq   op/binary/test_neq.cpp)
add_executable(test_op_gt    op/binary/test_gt.cpp)
add_executable(test_op_lt    op/binary/test_lt.cpp)
add_executable(test_op_max    op/binary/test_max.cpp)
add_executable(test_op_min    op/binary/test_min.cpp)

# Rolling operators
add_executable(test_op_rolling_diff  op/rolling/test_rolling_diff.cpp)
add_executable(test_op_rolling_shift op/rolling/test_rolling_shift.cpp)
add_executable(test_op_rolling_sma   op/rolling/test_rolling_sma.cpp)
add_executable(test_op_rolling_ema   op/rolling/test_rolling_ema.cpp)
add_executable(test_op_rolling_sum  op/rolling/test_rolling_sum.cpp)
add_executable(test_op_rolling_max  op/rolling/test_rolling_max.cpp)
add_executable(test_op_rolling_min  op/rolling/test_rolling_min.cpp)
add_executable(test_op_rolling_std  op/rolling/test_rolling_std.cpp)
add_executable(test_op_rolling_rank op/rolling/test_rolling_rank.cpp)
add_executable(test_op_rolling_mean    op/rolling/test_rolling_mean.cpp)
add_executable(test_op_rolling_var     op/rolling/test_rolling_var.cpp)
add_executable(test_op_rolling_median  op/rolling/test_rolling_median.cpp)
add_executable(test_op_rolling_mul     op/rolling/test_rolling_mul.cpp)
add_executable(test_op_rolling_argmax  op/rolling/test_rolling_argmax.cpp)
add_executable(test_op_rolling_argmin  op/rolling/test_rolling_argmin.cpp)
add_executable(test_op_rolling_kurt    op/rolling/test_rolling_kurt.cpp)
add_executable(test_op_rolling_skew    op/rolling/test_rolling_skew.cpp)
add_executable(test_op_rolling_quantile op/rolling/test_rolling_quantile.cpp)
add_executable(test_op_rolling_corr  op/rolling/test_rolling_corr.cpp)
add_executable(test_op_rolling_cov   op/rolling/test_rolling_cov.cpp)
add_executable(test_rolling_cross_validate_pandas op/rolling/test_cross_validate_pandas.cpp)

# CS (cross-section) operators
add_executable(test_op_red_ops   op/red/test_red_ops.cpp)
add_executable(test_op_cs_ops    op/cs/test_cs_ops.cpp)

# Integration tests
add_executable(test_full_pipeline integration/test_full_pipeline.cpp)
add_executable(test_parquet_reader integration/test_parquet_reader.cpp)
add_executable(test_hdf5_reader integration/test_hdf5_reader.cpp)

# Regression tests
add_executable(golden_factors    regression/golden_factors.cpp)

# Link all test targets
set(TEST_TARGETS
    test_column test_market_data
    test_expression test_buffer_pool test_engine test_parser test_factor_calculator test_panel_data test_intermediate_column test_cross_section_workspace test_csv_loader test_alpha0001 test_alpha0002 test_alpha1001 test_alpha0003_0005 test_simd_cross_validate
    test_registry test_binary_ops test_sma20_bench
    test_op_abs test_op_exp test_op_log test_op_log2 test_op_sqrt test_op_square test_op_inv test_op_log10 test_op_sign test_op_not test_op_rank
    test_op_add test_op_sub test_op_mul test_op_div test_op_eq test_op_neq test_op_gt test_op_lt test_op_max test_op_min
    test_op_rolling_diff test_op_rolling_shift test_op_rolling_sma test_op_rolling_ema test_op_rolling_sum test_op_rolling_max test_op_rolling_min test_op_rolling_std test_op_rolling_rank
    test_op_rolling_mean test_op_rolling_var test_op_rolling_median test_op_rolling_mul test_op_rolling_argmax test_op_rolling_argmin test_op_rolling_kurt test_op_rolling_skew test_op_rolling_quantile test_op_rolling_corr test_op_rolling_cov
    test_rolling_cross_validate_pandas
    test_op_red_ops
    test_op_cs_ops
    test_full_pipeline
    test_parquet_reader test_hdf5_reader
    golden_factors
)

foreach(target ${TEST_TARGETS})
    target_link_libraries(${target} PRIVATE quantcore GTest::gtest GTest::gtest_main)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${target} COMMAND ${target})
endforeach()

# test_csv_loader needs the project source dir to locate example CSV files
target_compile_definitions(test_csv_loader PRIVATE
    TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# test_alpha0001, test_alpha0002 and test_alpha1001 additionally link against the factors shared library
target_link_libraries(test_alpha0001 PRIVATE quantcorefactors)
target_link_libraries(test_alpha0002 PRIVATE quantcorefactors)
target_link_libraries(test_alpha0003_0005 PRIVATE quantcorefactors)
# test_alpha1001 links against quantcore (static) — computeSma20 is compiled
# into the static lib for direct, non-PLT linkage (zero-overhead calls).
target_link_libraries(test_alpha1001 PRIVATE quantcore)
target_link_libraries(test_sma20_bench PRIVATE quantcore)
