// ExecutionEngine.h — orchestrates expression evaluation with buffer management
// Phase: 二期必实现
//
// Provides:
//   1. evaluate(expr, md) → Column<double>  unified entry point
//   2. BufferPool lifecycle management
//   3. EngineMetrics collection
//
// The engine uses post-order traversal implicitly via ExprNode::evaluate().
// Temporary buffers are allocated from the internal BufferPool when the
// pool-aware evaluate() overload is used.
//
// Phase 4 will add FusedLoopGenerator for operator fusion.
//
// Thread safety: ExecutionEngine is NOT thread-safe internally (BufferPool
// is single-threaded).  Create one Engine per thread for concurrent use.
#pragma once

#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/engine/BufferPool.h"
#include "quantcore/engine/EngineMetrics.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"

namespace quantcore {

class ExecutionEngine {
public:
    ExecutionEngine() = default;

    // ============================================================
    // Main entry point
    // ============================================================

    /// Evaluate an expression tree against MarketData.
    ///
    /// The expression tree is traversed read-only — the engine does not
    /// modify it.  For concurrent evaluation of the same tree on different
    /// data, create multiple Engine instances (one per thread).
    ///
    /// The output buffer is allocated from the internal BufferPool for
    /// 64-byte alignment, then copied to a Column<double>.  Internal
    /// temporary buffers (for BinaryExpr, RollingExpr, etc.) also use
    /// the pool when available.
    ///
    /// @return A Column<double> containing the result.  The data memory
    ///         is 64-byte-aligned.
    Column<double> evaluate(const ExprNode& expr, const MarketData& md);

    // ============================================================
    // Metrics
    // ============================================================

    const EngineMetrics& metrics() const noexcept { return metrics_; }
    void resetMetrics() { metrics_.reset(); }

    // ============================================================
    // Buffer pool access
    // ============================================================

    BufferPool& pool() noexcept { return pool_; }
    const BufferPool& pool() const noexcept { return pool_; }

private:
    BufferPool    pool_;
    EngineMetrics metrics_;
};

}  // namespace quantcore
