// bench_unary_ops.cpp — microbenchmarks for unary operators
//
// Benchmarks each unary operator at multiple data sizes.
// Build:  cmake .. -DQUANTCORE_BUILD_BENCHMARKS=ON
// Run:    ./benchmarks/bench_unary_ops --benchmark_min_time=0.3s

#include <benchmark/benchmark.h>

#include <cmath>
#include <memory>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/engine/ExecutionEngine.h"
#include "quantcore/expression/ColumnRef.h"
#include "quantcore/expression/UnaryExpr.h"
#include "quantcore/registry/OperatorRegistry.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;

// ============================================================
// Data generation
// ============================================================

static MarketData makeMarketData(std::size_t n) {
    std::vector<int64_t> timestamps(n);
    for (std::size_t i = 0; i < n; ++i)
        timestamps[i] = static_cast<int64_t>(i);
    TimestampIndex tsIdx(timestamps.data(), n);
    MarketData md("BENCH", std::move(tsIdx));

    Column<double> close(n);
    for (std::size_t i = 0; i < n; ++i)
        close[i] = 12.0 + static_cast<double>(i);
    md.setColumn(Field::CLOSE, std::move(close));
    return md;
}

// ============================================================
// Raw dispatch benchmarks
// ============================================================

template <UnaryOpCode Op>
static void BM_UnaryRaw(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<double> input(n, 3.14159);
    std::vector<double> output(n);
    Operand op(input.data());
    auto& reg = OperatorRegistry::instance();
    auto mask = static_cast<const uint64_t*>(nullptr);

    for (auto _ : state) {
        reg.invokeUnary(Op, op, output.data(), n, mask);
        benchmark::DoNotOptimize(output.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_UnaryRaw<UnaryOpCode::ABS>)->Range(1<<10, 1<<20)->Name("UnaryRaw/ABS");
BENCHMARK(BM_UnaryRaw<UnaryOpCode::LOG>)->Range(1<<10, 1<<20)->Name("UnaryRaw/LOG");
BENCHMARK(BM_UnaryRaw<UnaryOpCode::SQRT>)->Range(1<<10, 1<<20)->Name("UnaryRaw/SQRT");
BENCHMARK(BM_UnaryRaw<UnaryOpCode::EXP>)->Range(1<<10, 1<<20)->Name("UnaryRaw/EXP");
BENCHMARK(BM_UnaryRaw<UnaryOpCode::NEG>)->Range(1<<10, 1<<20)->Name("UnaryRaw/NEG");
BENCHMARK(BM_UnaryRaw<UnaryOpCode::SIGN>)->Range(1<<10, 1<<20)->Name("UnaryRaw/SIGN");

// ============================================================
// Scalar function benchmarks (for fused loop comparison)
// ============================================================

template <UnaryOpCode Op>
static void BM_UnaryScalar(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<double> input(n, 3.14159);
    std::vector<double> output(n);
    auto& reg = OperatorRegistry::instance();
    auto fn = reg.getUnaryScalar(Op);

    for (auto _ : state) {
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = fn(input[i]);
        }
        benchmark::DoNotOptimize(output.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_UnaryScalar<UnaryOpCode::ABS>)->Range(1<<10, 1<<20)->Name("UnaryScalar/ABS");
BENCHMARK(BM_UnaryScalar<UnaryOpCode::LOG>)->Range(1<<10, 1<<20)->Name("UnaryScalar/LOG");
BENCHMARK(BM_UnaryScalar<UnaryOpCode::SQRT>)->Range(1<<10, 1<<20)->Name("UnaryScalar/SQRT");

// ============================================================
// Expression engine benchmarks (AST → evaluate)
// ============================================================

template <UnaryOpCode Op, const char* Name>
static void BM_UnaryEngine(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));
    auto md = makeMarketData(n);
    UnaryExpr expr(Op, std::make_unique<ColumnRef>(Field::CLOSE));
    ExecutionEngine engine;

    for (auto _ : state) {
        auto result = engine.evaluate(expr, md);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static const char kAbsName[] = "abs";
static const char kLogName[] = "log";
static const char kSqrtName[] = "sqrt";
static const char kExpName[] = "exp";

BENCHMARK(BM_UnaryEngine<UnaryOpCode::ABS, kAbsName>)->Range(1<<10, 1<<20)->Name("Engine/ABS");
BENCHMARK(BM_UnaryEngine<UnaryOpCode::LOG, kLogName>)->Range(1<<10, 1<<20)->Name("Engine/LOG");
BENCHMARK(BM_UnaryEngine<UnaryOpCode::SQRT, kSqrtName>)->Range(1<<10, 1<<20)->Name("Engine/SQRT");
BENCHMARK(BM_UnaryEngine<UnaryOpCode::EXP, kExpName>)->Range(1<<10, 1<<20)->Name("Engine/EXP");

BENCHMARK_MAIN();
