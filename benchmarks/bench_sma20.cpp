// bench_sma20.cpp — benchmark SIMD vs scalar SMA-20 implementations
//
// Compares:
//   1. Scalar baseline — plain running-sum loop (no SIMD)
//   2. SIMD accelerated — computeSma20() with AVX2 intrinsics
//   3. Expression engine — FactorCalculator + "rolling_mean(close, 20)"
//
// Build:  cmake .. -DQUANTCORE_BUILD_BENCHMARKS=ON
// Run:    ./benchmarks/bench_sma20 --benchmark_min_time=0.5s

#include <benchmark/benchmark.h>

#include <cmath>
#include <memory>
#include <vector>

#include "factors/alpha_1001.h"
#include "quantcore/core/FactorCalculator.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;
using namespace quantcore::factors;

// ============================================================
// Pure scalar baseline (no SIMD, no ifdefs)
// ============================================================

Column<double> sma20_scalar_baseline(const Column<double>& close) {
    constexpr std::size_t kWindow = 20;
    const std::size_t n = close.size();
    Column<double> result(n);

    if (n < kWindow) {
        for (std::size_t i = 0; i < n; ++i)
            result[i] = std::nan("");
        return result;
    }

    for (std::size_t i = 0; i < kWindow - 1; ++i)
        result[i] = std::nan("");

    double runningSum = 0.0;
    for (std::size_t i = 0; i < kWindow; ++i)
        runningSum += close[i];

    double inv = 1.0 / static_cast<double>(kWindow);
    result[kWindow - 1] = runningSum * inv;

    for (std::size_t i = kWindow; i < n; ++i) {
        runningSum += close[i] - close[i - kWindow];
        result[i] = runningSum * inv;
    }
    return result;
}

// ============================================================
// Test data generation
// ============================================================

static Column<double> generateClose(std::size_t n) {
    Column<double> close(n);
    for (std::size_t i = 0; i < n; ++i)
        close[i] = 100.0 + static_cast<double>(i) * 0.01;
    return close;
}

static MarketData generateMarketData(std::size_t n) {
    std::vector<int64_t> timestamps(n);
    for (std::size_t i = 0; i < n; ++i)
        timestamps[i] = static_cast<int64_t>(i);
    TimestampIndex tsIdx(timestamps.data(), n);
    MarketData md("BENCH", std::move(tsIdx));
    md.setColumn(Field::CLOSE, generateClose(n));
    for (std::size_t f = 0; f < static_cast<std::size_t>(Field::kFieldCount); ++f) {
        auto field = static_cast<Field>(f);
        if (field == Field::CLOSE) continue;
        md.setColumn(field, Column<double>(n, 0.0));
    }
    return md;
}

// ============================================================
// Benchmarks
// ============================================================

// 1K rows
static void BM_ScalarBaseline_1K(benchmark::State& state) {
    auto close = generateClose(1'000);
    for (auto _ : state) {
        auto result = sma20_scalar_baseline(close);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * 1'000);
}
BENCHMARK(BM_ScalarBaseline_1K);

static void BM_SimdComputeSma20_1K(benchmark::State& state) {
    auto close = generateClose(1'000);
    for (auto _ : state) {
        auto result = computeSma20(close);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * 1'000);
}
BENCHMARK(BM_SimdComputeSma20_1K);

static void BM_ExpressionEngine_1K(benchmark::State& state) {
    auto md = generateMarketData(1'000);
    FactorCalculator calc;
    calc.registerFormula("sma20", "rolling_mean(close, 20)");
    for (auto _ : state) {
        auto result = calc.evaluate("sma20", md);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * 1'000);
}
BENCHMARK(BM_ExpressionEngine_1K);

// 10K rows
static void BM_ScalarBaseline_10K(benchmark::State& state) {
    auto close = generateClose(10'000);
    for (auto _ : state) {
        auto result = sma20_scalar_baseline(close);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * 10'000);
}
BENCHMARK(BM_ScalarBaseline_10K);

static void BM_SimdComputeSma20_10K(benchmark::State& state) {
    auto close = generateClose(10'000);
    for (auto _ : state) {
        auto result = computeSma20(close);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * 10'000);
}
BENCHMARK(BM_SimdComputeSma20_10K);

static void BM_ExpressionEngine_10K(benchmark::State& state) {
    auto md = generateMarketData(10'000);
    FactorCalculator calc;
    calc.registerFormula("sma20", "rolling_mean(close, 20)");
    for (auto _ : state) {
        auto result = calc.evaluate("sma20", md);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * 10'000);
}
BENCHMARK(BM_ExpressionEngine_10K);

// 100K rows
static void BM_ScalarBaseline_100K(benchmark::State& state) {
    auto close = generateClose(100'000);
    for (auto _ : state) {
        auto result = sma20_scalar_baseline(close);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * 100'000);
}
BENCHMARK(BM_ScalarBaseline_100K);

static void BM_SimdComputeSma20_100K(benchmark::State& state) {
    auto close = generateClose(100'000);
    for (auto _ : state) {
        auto result = computeSma20(close);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * 100'000);
}
BENCHMARK(BM_SimdComputeSma20_100K);

static void BM_ExpressionEngine_100K(benchmark::State& state) {
    auto md = generateMarketData(100'000);
    FactorCalculator calc;
    calc.registerFormula("sma20", "rolling_mean(close, 20)");
    for (auto _ : state) {
        auto result = calc.evaluate("sma20", md);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * 100'000);
}
BENCHMARK(BM_ExpressionEngine_100K);

// 1M rows
static void BM_ScalarBaseline_1M(benchmark::State& state) {
    auto close = generateClose(1'000'000);
    for (auto _ : state) {
        auto result = sma20_scalar_baseline(close);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * 1'000'000);
}
BENCHMARK(BM_ScalarBaseline_1M);

static void BM_SimdComputeSma20_1M(benchmark::State& state) {
    auto close = generateClose(1'000'000);
    for (auto _ : state) {
        auto result = computeSma20(close);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * 1'000'000);
}
BENCHMARK(BM_SimdComputeSma20_1M);

static void BM_ExpressionEngine_1M(benchmark::State& state) {
    auto md = generateMarketData(1'000'000);
    FactorCalculator calc;
    calc.registerFormula("sma20", "rolling_mean(close, 20)");
    for (auto _ : state) {
        auto result = calc.evaluate("sma20", md);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * 1'000'000);
}
BENCHMARK(BM_ExpressionEngine_1M);

BENCHMARK_MAIN();
