// RollingArgMin.h — Rolling argmin (滚动窗口内最小值位置)
// Phase: 一期实现
//
// Mathematical form:  rolling_argmin(X, n)[i] = position of min in window
// Reference:  pandas.Series.rolling(n).apply(np.argmin)  — verified match
//
// Returns the 0-based positional index (0 to n-1) of the first occurrence
// of the minimum value within the window ending at position i.
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

struct RollingArgMinOp : public RollingOperator<RollingArgMinOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_ARGMIN;
    static constexpr const char* name = "rolling_argmin";

    using RollingOperator<RollingArgMinOp>::RollingOperator;

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

        double minVal = input[start];
        if (std::isnan(minVal)) return minVal;
        std::size_t argmin = 0;

        for (std::size_t j = start + 1; j <= i; ++j) {
            if (std::isnan(input[j])) return input[j];
            if (input[j] < minVal) {
                minVal = input[j];
                argmin = j - start;
            }
        }
        return static_cast<double>(argmin);
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
                double minVal = input[start];
                std::size_t argmin = 0;
                for (std::size_t j = start + 1; j <= i; ++j) {
                    if (input[j] < minVal) {
                        minVal = input[j];
                        argmin = j - start;
                    }
                }
                output[i] = static_cast<double>(argmin);
            }
        }
    }
};

}  // namespace quantcore
