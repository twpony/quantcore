// bench_buffer_pool.cpp — BufferPool allocation/deallocation microbenchmarks
//
// Benchmarks: single allocation, batch allocate+free cycle,
// multiple size classes, and comparison with raw new/delete.
// Build:  cmake .. -DQUANTCORE_BUILD_BENCHMARKS=ON
// Run:    ./benchmarks/bench_buffer_pool --benchmark_min_time=0.3s

#include <benchmark/benchmark.h>

#include <cstdlib>
#include <vector>

#include "quantcore/engine/BufferPool.h"

using namespace quantcore;

// ============================================================
// Raw malloc/free baseline
// ============================================================

static void BM_MallocFree(benchmark::State& state) {
    std::size_t bytes = static_cast<std::size_t>(state.range(0));

    for (auto _ : state) {
        double* p = static_cast<double*>(std::aligned_alloc(64, bytes));
        benchmark::DoNotOptimize(p);
        std::free(p);
    }
    state.SetBytesProcessed(state.iterations() * bytes);
}

// ============================================================
// Pool allocate + release (single)
// ============================================================

static void BM_PoolAllocRelease(benchmark::State& state) {
    std::size_t count = static_cast<std::size_t>(state.range(0));
    BufferPool pool;

    for (auto _ : state) {
        auto handle = pool.allocate<double>(count);
        benchmark::DoNotOptimize(handle.data());
        // handle goes out of scope, releases back to pool
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * count * sizeof(double));
}

// ============================================================
// Pool: batch allocate N handles of size K, then release all
// ============================================================

static void BM_PoolBatchAlloc(benchmark::State& state) {
    std::size_t batchSize = static_cast<std::size_t>(state.range(0));
    constexpr std::size_t kElemCount = 256;  // 2 KiB per handle
    BufferPool pool;

    for (auto _ : state) {
        std::vector<BufferHandle<double>> handles;
        handles.reserve(batchSize);
        for (std::size_t i = 0; i < batchSize; ++i) {
            handles.push_back(pool.allocate<double>(kElemCount));
        }
        benchmark::DoNotOptimize(handles.data());
        // All handles go out of scope, releasing to pool
    }
    state.SetItemsProcessed(state.iterations() * batchSize);
}

// ============================================================
// Pool: interleaved allocate + release (simulates expression eval)
// ============================================================

static void BM_PoolInterleaved(benchmark::State& state) {
    std::size_t iterCount = static_cast<std::size_t>(state.range(0));
    BufferPool pool;

    for (auto _ : state) {
        for (std::size_t i = 0; i < iterCount; ++i) {
            auto h1 = pool.allocate<double>(1000);  // ~8 KiB
            benchmark::DoNotOptimize(h1.data());
            auto h2 = pool.allocate<double>(500);   // ~4 KiB
            benchmark::DoNotOptimize(h2.data());
            auto h3 = pool.allocate<double>(2000);  // ~16 KiB
            benchmark::DoNotOptimize(h3.data());
            // h1, h2, h3 released in reverse order (LIFO)
        }
    }
    state.SetItemsProcessed(state.iterations() * iterCount * 3);
}

// ============================================================
// Pool: stress test — mixed size allocations
// ============================================================

static void BM_PoolMixedSizes(benchmark::State& state) {
    std::size_t rounds = static_cast<std::size_t>(state.range(0));
    BufferPool pool;
    // Mix of small, medium, large, and overflow allocations
    constexpr std::size_t sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    for (auto _ : state) {
        for (std::size_t i = 0; i < rounds; ++i) {
            auto h = pool.allocate<double>(sizes[i % 9]);
            benchmark::DoNotOptimize(h.data());
        }
    }
    state.SetItemsProcessed(state.iterations() * rounds);
}

// ============================================================
// Pool: re-use same buffer repeatedly (pool hit rate)
// ============================================================

static void BM_PoolReuse(benchmark::State& state) {
    std::size_t elemCount = static_cast<std::size_t>(state.range(0));
    BufferPool pool;

    for (auto _ : state) {
        for (int i = 0; i < 100; ++i) {
            auto handle = pool.allocate<double>(elemCount);
            benchmark::DoNotOptimize(handle.data());
        }
    }
    state.SetItemsProcessed(state.iterations() * 100);
}

BENCHMARK(BM_MallocFree)->Range(256, 65536)->Name("Baseline/MallocFree_256B-64KB");
BENCHMARK(BM_PoolAllocRelease)->Range(1<<7, 1<<18)->Name("Pool/AllocRelease_128-256K_elems");
BENCHMARK(BM_PoolBatchAlloc)->Range(1<<4, 1<<10)->Name("Pool/Batch16-1024");
BENCHMARK(BM_PoolInterleaved)->Range(1<<4, 1<<10)->Name("Pool/Interleaved_LIFO");
BENCHMARK(BM_PoolMixedSizes)->Range(1<<4, 1<<10)->Name("Pool/MixedSizes");
BENCHMARK(BM_PoolReuse)->Range(1<<7, 1<<16)->Name("Pool/Reuse100x");

BENCHMARK_MAIN();
