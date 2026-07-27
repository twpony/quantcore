// RollingShift.h — Shift / Lag (平移/滞后)
// Phase: 一期实现
//
// Mathematical form:  rolling_shift(X, n) = x[i-n]
// Reference:  pandas.Series.shift(periods=n)  — verified match
//
// Returns the value `n` positions before the current position.
// For i < n, there is insufficient history and evaluateScalar
// returns NaN (matching pandas).
//
// Null propagation: null/NaN at input[i-n] propagates naturally
// — shift just moves the reference, so whatever was at i-n
// (including NaN) is returned.
//
// Edge cases (matching pandas):
//   shift(periods=0)  →  identity  (x[i])
//   shift(periods>=N) →  all NaN
//   NaN input propagates as NaN
//   ±Inf input propagates as ±Inf

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"

namespace quantcore {

struct RollingShiftOp : public RollingOperator<RollingShiftOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_SHIFT;
    static constexpr const char* name = "rolling_shift";

    using RollingOperator<RollingShiftOp>::RollingOperator;

    // ============================================================
    // Scalar reference implementation
    // ============================================================
    //
    // Returns x[i-n].  Returns NaN when i < n.
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
        return input[i - window];
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
                output[i] = input[i - window];
            }
        }
    }
};

}  // namespace quantcore
