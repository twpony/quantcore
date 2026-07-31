// bench_binary_ops.cpp — microbenchmarks for binary operators
//
// Benchmarks each binary operator at multiple data sizes
// with column+column, column+scalar, and scalar+column patterns.
// Build:  cmake .. -DQUANTCORE_BUILD_BENCHMARKS=ON
// Run:    ./benchmarks/bench_binary_ops --benchmark_min_time=0.3s

#include <benchmark/benchmark.h>

#include <memory>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/engine/ExecutionEngine.h"
#include "quantcore/expression/BinaryExpr.h"
#include "quantcore/expression/ColumnRef.h"
#include "quantcore/expression/Scalar.h"
#include "quantcore/registry/OperatorRegistry.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;

// ============================================================
// Data helpers
// ============================================================

static MarketData makeMarketData(std::size_t n) {
    std::vector<int64_t> timestamps(n);
    for (std::size_t i = 0; i < n; ++i)
        timestamps[i] = static_cast<int64_t>(i);
    TimestampIndex tsIdx(timestamps.data(), n);
    MarketData md("BENCH", std::move(tsIdx));

    Column<double> close(n), open(n);
    for (std::size_t i = 0; i < n; ++i) {
        close[i] = 12.0 + static_cast<double>(i);
        open[i]  = 10.0 + static_cast<double>(i);
    }
    md.setColumn(Field::CLOSE, std::move(close));
    md.setColumn(Field::OPEN, std::move(open));
    return md;
}

// ============================================================
// Raw dispatch (column + column)
// ============================================================

template <BinaryOpCode Op>
static void BM_BinaryRawColCol(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<double> lhs(n, 3.0), rhs(n, 2.0), out(n);
    Operand lop(lhs.data()), rop(rhs.data());
    auto& reg = OperatorRegistry::instance();

    for (auto _ : state) {
        reg.invokeBinary(Op, lop, rop, out.data(), n, nullptr);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================
// Engine-based (AST → result)
// ============================================================

template <BinaryOpCode Op>
static void BM_BinaryEngineColCol(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));
    auto md = makeMarketData(n);
    auto expr = std::make_unique<BinaryExpr>(Op,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<ColumnRef>(Field::OPEN));
    ExecutionEngine engine;

    for (auto _ : state) {
        auto result = engine.evaluate(*expr, md);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

template <BinaryOpCode Op>
static void BM_BinaryEngineColScalar(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));
    auto md = makeMarketData(n);
    auto expr = std::make_unique<BinaryExpr>(Op,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<Scalar>(2.0));
    ExecutionEngine engine;

    for (auto _ : state) {
        auto result = engine.evaluate(*expr, md);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// Register benchmarks for all 4 arithmetic ops
BENCHMARK(BM_BinaryRawColCol<BinaryOpCode::ADD>)->Range(1<<10, 1<<20)->Name("Raw/ADD");
BENCHMARK(BM_BinaryRawColCol<BinaryOpCode::MUL>)->Range(1<<10, 1<<20)->Name("Raw/MUL");

BENCHMARK(BM_BinaryEngineColCol<BinaryOpCode::ADD>)->Range(1<<10, 1<<20)->Name("Engine/ADD_ColCol");
BENCHMARK(BM_BinaryEngineColCol<BinaryOpCode::SUB>)->Range(1<<10, 1<<20)->Name("Engine/SUB_ColCol");
BENCHMARK(BM_BinaryEngineColCol<BinaryOpCode::MUL>)->Range(1<<10, 1<<20)->Name("Engine/MUL_ColCol");
BENCHMARK(BM_BinaryEngineColCol<BinaryOpCode::DIV>)->Range(1<<10, 1<<20)->Name("Engine/DIV_ColCol");

BENCHMARK(BM_BinaryEngineColScalar<BinaryOpCode::ADD>)->Range(1<<10, 1<<20)->Name("Engine/ADD_ColSca");
BENCHMARK(BM_BinaryEngineColScalar<BinaryOpCode::MUL>)->Range(1<<10, 1<<20)->Name("Engine/MUL_ColSca");

BENCHMARK_MAIN();
