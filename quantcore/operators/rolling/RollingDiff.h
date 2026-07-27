// RollingDiff.h — Difference (差分)
// Phase: 一期实现
//
// Mathematical form:  rolling_diff(X, n) = x[i] - x[i-n]
// Reference:  pandas.Series.diff(periods=n)  — verified match
//
// The operator computes the difference between the current value
// and the value `n` positions before it.  For i < n, there is
// insufficient history and evaluateScalar returns NaN.
//
// Null propagation: if input[i] or input[i-n] is null/NaN, IEEE 754
// arithmetic ensures NaN propagation (NaN - x = x - NaN = NaN).
//
// Edge cases (matching pandas):
//   diff(periods=0)  →  all zeros  (x[i] - x[i])
//   diff(periods>=N) →  all NaN
//   Inf - Inf  -> NaN
//   Inf - any  -> ±Inf
//   NaN - any  -> NaN

#pragma once

#include <cmath>
#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"

namespace quantcore {

struct RollingDiffOp : public RollingOperator<RollingDiffOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_DIFF;
    static constexpr const char* name = "rolling_diff";

    using RollingOperator<RollingDiffOp>::RollingOperator;

    // ============================================================
    // Scalar reference implementation
    // ============================================================
    //
    // Computes x[i] - x[i-n].  Returns NaN when i < n (the lag).
    //
    // @param input   Input data pointer (length >= i+1, non-owning)
    // @param i       Current position (0-indexed)
    // @param window  Lag distance n

    static double evaluateScalar(const double* input,
                                 std::size_t i,
                                 std::size_t window) noexcept
    {
        if (i < window) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return input[i] - input[i - window];
    }

    // ============================================================
    // SIMD kernel (auto-vectorized scalar loop)
    // ============================================================

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                      double*         output,
                      std::size_t     window) noexcept
    {
        std::size_t n = input.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            if (i < window) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                output[i] = input[i] - input[i - window];
            }
        }
    }
};

}  // namespace quantcore
