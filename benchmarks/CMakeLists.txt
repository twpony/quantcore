# Benchmarks CMakeLists.txt

include(../cmake/FindGBenchmark.cmake)

add_executable(bench_unary_ops         bench_unary_ops.cpp)
add_executable(bench_binary_ops        bench_binary_ops.cpp)
add_executable(bench_expression_fusion bench_expression_fusion.cpp)
add_executable(bench_buffer_pool       bench_buffer_pool.cpp)

set(BENCH_TARGETS
    bench_unary_ops bench_binary_ops
    bench_expression_fusion bench_buffer_pool
)

foreach(target ${BENCH_TARGETS})
    target_link_libraries(${target} PRIVATE quantcore benchmark::benchmark)
endforeach()
