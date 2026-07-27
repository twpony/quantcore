// RollingSma.h — Simple Moving Average (简单移动平均)
// Phase: 一期实现
//
// Mathematical form:  rolling_sma(X, n)[i] = mean(x[i-n+1], ..., x[i])
// Reference:  pandas.Series.rolling(n).mean()  — verified match
//
// Computes the arithmetic mean of the most recent `n` observations.
// For i < n-1, there are fewer than `n` observations and
// evaluateScalar returns NaN (matching pandas default min_periods=n).
//
// Null propagation: if any input in the window is null, the engine
// marks output[i] as null.  evaluateScalar assumes clean input.
//
// Numerical notes:
//   - Uses naive summation for the scalar reference.  The SIMD path
//     uses running-sum / prefix-sum difference for O(1) per element.
//   - For large windows, consider compensated (Kahan) summation
//     to reduce floating-point error.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"

namespace quantcore {

struct RollingSmaOp : public RollingOperator<RollingSmaOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_SMA;
    static constexpr const char* name = "rolling_sma";

    using RollingOperator<RollingSmaOp>::RollingOperator;

    // ============================================================
    // Scalar reference implementation
    // ============================================================
    //
    // Computes the simple moving average over the window ending at i.
    // Returns NaN when i < window-1 (fewer than `window` elements).

    static double evaluateScalar(const double* input,
                                 std::size_t i,
                                 std::size_t window) noexcept
    {
        if (i + 1 < window) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::size_t start = i + 1 - window;
        double sum = 0.0;
        for (std::size_t j = start; j <= i; ++j) {
            sum += input[j];
        }
        return sum / static_cast<double>(window);
    }

    // ============================================================
    // SIMD kernel (auto-vectorized scalar loop with prefix-sum)
    // ============================================================

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                      double*         output,
                      std::size_t     window) noexcept
    {
        std::size_t n = input.size();
        if (n == 0) return;

        // Build prefix sum for O(1) window queries
        // Use running sum to avoid extra allocation
        double runningSum = 0.0;
        double invWindow = 1.0 / static_cast<double>(window);

        for (std::size_t i = 0; i < n; ++i) {
            runningSum += input[i];
            if (i + 1 < window) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                if (i >= window) {
                    runningSum -= input[i - window];
                }
                output[i] = runningSum * invWindow;
            }
        }
    }
};

}  // namespace quantcore
