// RollingSkew.h — Rolling skewness (滚动偏度, 有偏三阶矩)
// Phase: 一期实现
//
// Mathematical form:  rolling_skew(X, n)[i] = skewness of window
// Reference:  pandas.Series.rolling(n).skew()  — bias=True (population)
//
// Computes the biased skewness of the most recent `n` observations:
//   skew = m3 / (m2)^(3/2)
// where m2 = Σ(x-μ)² / n  and  m3 = Σ(x-μ)³ / n.
//
// For i < n-1, evaluateScalar returns NaN.
// For constant windows (variance = 0), returns NaN.
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

struct RollingSkewOp : public RollingOperator<RollingSkewOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_SKEW;
    static constexpr const char* name = "rolling_skew";

    using RollingOperator<RollingSkewOp>::RollingOperator;

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

        // First pass: mean
        double sum = 0.0;
        for (std::size_t j = start; j <= i; ++j) {
            sum += input[j];
        }
        double mean = sum / static_cast<double>(count);

        // Second pass: central moments
        double m2 = 0.0;  // variance
        double m3 = 0.0;  // 3rd central moment
        for (std::size_t j = start; j <= i; ++j) {
            double diff = input[j] - mean;
            double diff2 = diff * diff;
            m2 += diff2;
            m3 += diff2 * diff;  // diff^2 * diff = diff^3
        }
        m2 /= static_cast<double>(count);
        m3 /= static_cast<double>(count);

        if (m2 == 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return m3 / (m2 * std::sqrt(m2));
    }

    // ============================================================
    // SIMD kernel — O(n*window) two-pass per window
    // ============================================================

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                      double*         output,
                      std::size_t     window) noexcept
    {
        std::size_t n = input.size();
        double invWindow = 1.0 / static_cast<double>(window);

        for (std::size_t i = 0; i < n; ++i) {
            if (i + 1 < window) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                std::size_t start = i + 1 - window;

                double sum = 0.0;
                for (std::size_t j = start; j <= i; ++j) {
                    sum += input[j];
                }
                double mean = sum * invWindow;

                double m2 = 0.0;
                double m3 = 0.0;
                for (std::size_t j = start; j <= i; ++j) {
                    double diff = input[j] - mean;
                    double diff2 = diff * diff;
                    m2 += diff2;
                    m3 += diff2 * diff;
                }
                m2 *= invWindow;
                m3 *= invWindow;

                if (m2 == 0.0) {
                    output[i] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    output[i] = m3 / (m2 * std::sqrt(m2));
                }
            }
        }
    }
};

}  // namespace quantcore
