// bench_expression_fusion.cpp — fusion-on vs fusion-off performance comparison
//
// Compares fused-loop evaluation against standard post-order
// evaluation for unary chains, binary-with-unary-chains, and
// deep nested expressions.
// Build:  cmake .. -DQUANTCORE_BUILD_BENCHMARKS=ON
// Run:    ./benchmarks/bench_expression_fusion --benchmark_min_time=0.3s

#include <benchmark/benchmark.h>

#include <memory>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/engine/ExecutionEngine.h"
#include "quantcore/expression/BinaryExpr.h"
#include "quantcore/expression/ColumnRef.h"
#include "quantcore/expression/FusedLoopGenerator.h"
#include "quantcore/expression/Scalar.h"
#include "quantcore/expression/UnaryExpr.h"
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

    Column<double> close(n), volume(n);
    for (std::size_t i = 0; i < n; ++i) {
        close[i]  = 12.0 + static_cast<double>(i);
        volume[i] = 100.0 * static_cast<double>(i + 1);
    }
    md.setColumn(Field::CLOSE, std::move(close));
    md.setColumn(Field::VOLUME, std::move(volume));
    return md;
}

// ============================================================
// Build a deep unary chain: op4(op3(op2(op1(op0(CLOSE)))))
// ============================================================

static std::unique_ptr<UnaryExpr> makeUnaryChain(
    const std::vector<UnaryOpCode>& ops) {
    auto leaf = std::make_unique<ColumnRef>(Field::CLOSE);
    std::unique_ptr<UnaryExpr> chain;
    for (auto op : ops) {
        if (!chain) {
            chain = std::make_unique<UnaryExpr>(op, std::move(leaf));
        } else {
            chain = std::make_unique<UnaryExpr>(op, std::move(chain));
        }
    }
    return chain;
}

// ============================================================
// Fusion: unary chain (LOG → ABS → SQRT → NEG)
// ============================================================

static void BM_FusedUnaryChain(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));
    auto md = makeMarketData(n);

    // LOG(ABS(SQRT(NEG(CLOSE))))
    std::vector<UnaryOpCode> ops = {
        UnaryOpCode::NEG, UnaryOpCode::SQRT,
        UnaryOpCode::ABS, UnaryOpCode::LOG
    };
    auto expr = makeUnaryChain(ops);
    ExecutionEngine engine;

    for (auto _ : state) {
        auto result = engine.evaluate(*expr, md);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================
// Fusion: binary with unary chains on both sides
// ============================================================

static void BM_FusedBinaryWithChains(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));
    auto md = makeMarketData(n);

    // LOG(CLOSE) + LOG(VOLUME)
    auto lhs = std::make_unique<UnaryExpr>(UnaryOpCode::LOG,
        std::make_unique<ColumnRef>(Field::CLOSE));
    auto rhs = std::make_unique<UnaryExpr>(UnaryOpCode::LOG,
        std::make_unique<ColumnRef>(Field::VOLUME));
    auto add = std::make_unique<BinaryExpr>(BinaryOpCode::ADD,
        std::move(lhs), std::move(rhs));
    ExecutionEngine engine;

    for (auto _ : state) {
        auto result = engine.evaluate(*add, md);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================
// Unary chain with result-chain on binary:
//   SQRT(ABS(CLOSE + VOLUME))
// ============================================================

static void BM_FusedResultChainOnBinary(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));
    auto md = makeMarketData(n);

    auto add = std::make_unique<BinaryExpr>(BinaryOpCode::ADD,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<ColumnRef>(Field::VOLUME));
    auto absAdd = std::make_unique<UnaryExpr>(UnaryOpCode::ABS, std::move(add));
    auto sqrtAbs = std::make_unique<UnaryExpr>(UnaryOpCode::SQRT, std::move(absAdd));
    ExecutionEngine engine;

    for (auto _ : state) {
        auto result = engine.evaluate(*sqrtAbs, md);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

// ============================================================
// Simple column copy (baseline for memory bandwidth)
// ============================================================

static void BM_ColumnCopy(benchmark::State& state) {
    std::size_t n = static_cast<std::size_t>(state.range(0));
    auto md = makeMarketData(n);
    auto colRef = std::make_unique<ColumnRef>(Field::CLOSE);
    ExecutionEngine engine;

    for (auto _ : state) {
        auto result = engine.evaluate(*colRef, md);
        benchmark::DoNotOptimize(result.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_ColumnCopy)->Range(1<<10, 1<<20)->Name("Fusion/ColumnCopy");
BENCHMARK(BM_FusedUnaryChain)->Range(1<<10, 1<<20)->Name("Fusion/UnaryChain4");
BENCHMARK(BM_FusedBinaryWithChains)->Range(1<<10, 1<<20)->Name("Fusion/BinaryWithChains");
BENCHMARK(BM_FusedResultChainOnBinary)->Range(1<<10, 1<<20)->Name("Fusion/ResultChainOnBinary");

BENCHMARK_MAIN();
