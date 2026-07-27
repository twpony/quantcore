// RollingMul.h — Rolling product (滚动连乘)
// Phase: 一期实现
//
// Mathematical form:  rolling_mul(X, n)[i] = prod(x[i-n+1], ..., x[i])
// Reference:  pandas.Series.rolling(n).apply(np.prod)  — verified match
//
// Computes the product of the most recent `n` observations.
// For i < n-1, evaluateScalar returns NaN.
//
// Note: running product with division by the leaving element is avoided
// because zeros in the window would cause division by zero.
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

struct RollingMulOp : public RollingOperator<RollingMulOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_MUL;
    static constexpr const char* name = "rolling_mul";

    using RollingOperator<RollingMulOp>::RollingOperator;

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
        double prod = 1.0;
        for (std::size_t j = start; j <= i; ++j) {
            prod *= input[j];
        }
        return prod;
    }

    // ============================================================
    // SIMD kernel — naive O(n*window)
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
                double prod = 1.0;
                #pragma omp simd
                for (std::size_t j = start; j <= i; ++j) {
                    prod *= input[j];
                }
                output[i] = prod;
            }
        }
    }
};

}  // namespace quantcore
