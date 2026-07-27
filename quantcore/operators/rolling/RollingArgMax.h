// RollingArgMax.h — Rolling argmax (滚动窗口内最大值位置)
// Phase: 一期实现
//
// Mathematical form:  rolling_argmax(X, n)[i] = position of max in window
// Reference:  pandas.Series.rolling(n).apply(np.argmax)  — verified match
//
// Returns the 0-based positional index (0 to n-1) of the first occurrence
// of the maximum value within the window ending at position i.
// For i < n-1, evaluateScalar returns NaN.
//
// NaN handling: if any input value in the window is NaN, the result is NaN
// (consistent with IEEE 754 NaN propagation).
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

struct RollingArgMaxOp : public RollingOperator<RollingArgMaxOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_ARGMAX;
    static constexpr const char* name = "rolling_argmax";

    using RollingOperator<RollingArgMaxOp>::RollingOperator;

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

        double maxVal = input[start];
        if (std::isnan(maxVal)) return maxVal;
        std::size_t argmax = 0;

        for (std::size_t j = start + 1; j <= i; ++j) {
            if (std::isnan(input[j])) return input[j];
            if (input[j] > maxVal) {
                maxVal = input[j];
                argmax = j - start;
            }
        }
        return static_cast<double>(argmax);
    }

    // ============================================================
    // SIMD kernel
    // ============================================================

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                      double*         output,
                      std::size_t     window) noexcept
    {
        std::size_t n = input.size();
        for (std::size_t i = 0; i < n; ++i) {
            if (i + 1 < window) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                std::size_t start = i + 1 - window;
                double maxVal = input[start];
                std::size_t argmax = 0;
                for (std::size_t j = start + 1; j <= i; ++j) {
                    if (input[j] > maxVal) {
                        maxVal = input[j];
                        argmax = j - start;
                    }
                }
                output[i] = static_cast<double>(argmax);
            }
        }
    }
};

}  // namespace quantcore
