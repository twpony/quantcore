// RollingMean.h — Rolling mean (滚动均值)
// Phase: 一期实现
//
// Mathematical form:  rolling_mean(X, n)[i] = mean(x[i-n+1], ..., x[i])
// Reference:  pandas.Series.rolling(n).mean()  — verified match
//
// Computes the arithmetic mean of the most recent `n` observations.
// For i < n-1, there are fewer than `n` observations and
// evaluateScalar returns NaN (matching pandas default min_periods=n).
//
// Null propagation: if any input in the window is null, the engine
// marks output[i] as null.  evaluateScalar assumes clean input.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"

namespace quantcore {

struct RollingMeanOp : public RollingOperator<RollingMeanOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_MEAN;
    static constexpr const char* name = "rolling_mean";

    using RollingOperator<RollingMeanOp>::RollingOperator;

    // ============================================================
    // Scalar reference implementation
    // ============================================================

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
    // SIMD kernel — O(n) with running sum
    // ============================================================

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                      double*         output,
                      std::size_t     window) noexcept
    {
        std::size_t n = input.size();
        if (n == 0) return;

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
