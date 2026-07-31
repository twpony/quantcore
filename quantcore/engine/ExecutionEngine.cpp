// ExecutionEngine.cpp — expression evaluation orchestrator
// Phase: 二期必实现
//
// See ExecutionEngine.h for the public interface and design rationale.

#include "quantcore/engine/ExecutionEngine.h"

#include <chrono>
#include <cstring>

#include "quantcore/expression/FusedLoopGenerator.h"
#include "quantcore/storage/Column.h"

namespace quantcore {

Column<double> ExecutionEngine::evaluate(const ExprNode& expr,
                                          const MarketData& md) {
    std::size_t n = md.rowCount();
    if (n == 0) {
        return Column<double>();
    }

    auto t0 = std::chrono::steady_clock::now();

    // 1. Allocate the result buffer from the pool for 64-byte alignment.
    auto resultHandle = pool_.allocate<double>(n);

    // 2. Try fused-loop evaluation first.  If the expression tree or any
    //    subtree contains a fusion boundary (RollingExpr, RedExpr, CsExpr),
    //    fall back to standard post-order evaluation.
    const uint64_t* nullMask = nullptr;

    FusedLoopGenerator fusionGen;
    auto kernel = fusionGen.tryCompile(&expr);
    if (kernel) {
        // Fused path: single loop over all elements
        kernel->evaluate(md, resultHandle.data(), n, nullMask);
    } else {
        // Standard path: post-order evaluate with pool-backed temp buffers
        nullMask = expr.evaluate(md, resultHandle.data(), n, &pool_);
    }

    // 3. Build a Column<double> from the result data.
    //    Copy the aligned pool buffer into the Column's own aligned storage.
    Column<double> result(n);
    std::memcpy(result.data(), resultHandle.data(), n * sizeof(double));

    // 4. Propagate the null mask if present.
    if (nullMask) {
        for (std::size_t i = 0; i < n; ++i) {
            if ((nullMask[i / 64] >> (i % 64)) & uint64_t{1}) {
                result.setNull(i);
            }
        }
    }

    // resultHandle released automatically — memory returns to pool.

    auto t1 = std::chrono::steady_clock::now();
    auto usec = static_cast<std::size_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    metrics_.recordEvaluation(usec, n, expr.nodeCount());

    return result;
}

}  // namespace quantcore
