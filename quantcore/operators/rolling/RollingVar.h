// RollingVar.h — Rolling variance (滚动方差, ddof=0)
// Phase: 一期实现
//
// Mathematical form:  rolling_var(X, n)[i] = population variance of window
// Reference:  pandas.Series.rolling(n).var(ddof=0)  — verified match
//
// Computes the population variance (ddof=0 divisor=n) of the most recent
// `n` observations.  For i < n-1, evaluateScalar returns NaN.
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

struct RollingVarOp : public RollingOperator<RollingVarOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_VAR;
    static constexpr const char* name = "rolling_var";

    using RollingOperator<RollingVarOp>::RollingOperator;

    // ============================================================
    // Scalar reference implementation — two-pass algorithm
    // ============================================================

    static double evaluateScalar(const double* input,
                                 std::size_t i,
                                 std::size_t window) noexcept
    {
        if (i + 1 < window) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::size_t start = i + 1 - window;
        std::size_t count = window;

        double sum = 0.0;
        for (std::size_t j = start; j <= i; ++j) {
            sum += input[j];
        }
        double mean = sum / static_cast<double>(count);

        double sumSqDiff = 0.0;
        for (std::size_t j = start; j <= i; ++j) {
            double diff = input[j] - mean;
            sumSqDiff += diff * diff;
        }
        return sumSqDiff / static_cast<double>(count);
    }

    // ============================================================
    // SIMD kernel — O(n) with running sum and sum-of-squares
    // ============================================================

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                      double*         output,
                      std::size_t     window) noexcept
    {
        std::size_t n = input.size();
        if (n == 0) return;

        double runningSum = 0.0;
        double runningSumSq = 0.0;
        double invWindow = 1.0 / static_cast<double>(window);

        for (std::size_t i = 0; i < n; ++i) {
            double val = input[i];
            runningSum += val;
            runningSumSq += val * val;

            if (i + 1 < window) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                if (i >= window) {
                    double oldVal = input[i - window];
                    runningSum -= oldVal;
                    runningSumSq -= oldVal * oldVal;
                }
                double mean = runningSum * invWindow;
                double variance = runningSumSq * invWindow - mean * mean;
                // Clamp negative variance from floating-point error
                if (variance < 0.0) variance = 0.0;
                output[i] = variance;
            }
        }
    }
};

}  // namespace quantcore
