// RollingEma.h — Exponential Moving Average (指数移动平均)
// Phase: 二期修正（添加窗口边界检查）
//
// Mathematical form (matches pandas ewm(span=n, adjust=False).mean()):
//   α = 2/(n+1)
//   EMA[0] = X[0]                                           (seed)
//   EMA[i] = α·X[i] + (1-α)·EMA[i-1]   for i > 0           (recurrence)
//
// Reference:  pandas.Series.ewm(span=n, adjust=False).mean()
//
// Boundary convention (consistent with all rolling window operators):
//   - First (window-1) positions are invalid → return NaN.
//   - Internally the EMA recurrence starts from position 0 so that
//     when position (window-1) becomes valid the recurrence has
//     already been warmed up by all preceding data.
//
// Null propagation:
//   - If input[i] is NaN/null, EMA[i] = NaN and the NaN drags forward
//     through the recurrence (consistent with pandas ignore_na=False).
//   - evaluateScalar recomputes from i=0 each time, so NaN at any
//     position contaminates all subsequent values.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"

namespace quantcore {

struct RollingEmaOp : public RollingOperator<RollingEmaOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_EMA;
    static constexpr const char* name = "rolling_ema";

    using RollingOperator<RollingEmaOp>::RollingOperator;

    // ============================================================
    // Scalar reference implementation
    // ============================================================
    //
    // Returns NaN when i+1 < window (not enough history).
    // Otherwise recomputes EMA from i=0 to position i in O(i) time.
    // EMA[0] = input[0]; EMA[k] = α·input[k] + (1-α)·EMA[k-1].

    static double evaluateScalar(const double* input,
                                 std::size_t i,
                                 std::size_t window) noexcept
    {
        if (i + 1 < window) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double alpha = 2.0 / static_cast<double>(window + 1);
        double oneMinusAlpha = 1.0 - alpha;

        double ema = input[0];
        for (std::size_t k = 1; k <= i; ++k) {
            ema = alpha * input[k] + oneMinusAlpha * ema;
        }
        return ema;
    }

    // ============================================================
    // SIMD kernel — single forward pass O(n)
    // ============================================================
    //
    // Outputs NaN for the first (window-1) positions.
    // The EMA recurrence still propagates through those positions
    // so that output[window-1] is the EMA of all preceding data.

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                      double*         output,
                      std::size_t     window) noexcept
    {
        std::size_t n = input.size();
        if (n == 0) return;

        double alpha = 2.0 / static_cast<double>(window + 1);
        double oneMinusAlpha = 1.0 - alpha;
        double nan = std::numeric_limits<double>::quiet_NaN();

        double ema = input[0];

        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0) {
                ema = alpha * input[i] + oneMinusAlpha * ema;
            }
            if (i + 1 < window) {
                output[i] = nan;
            } else {
                output[i] = ema;
            }
        }
    }
};

}  // namespace quantcore
