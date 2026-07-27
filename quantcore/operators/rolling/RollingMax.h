// RollingMax.h — Rolling maximum (滚动最大值)
// Phase: 一期实现

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"

namespace quantcore {

struct RollingMaxOp : public RollingOperator<RollingMaxOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_MAX;
    static constexpr const char* name = "rolling_max";

    using RollingOperator<RollingMaxOp>::RollingOperator;

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
        for (std::size_t j = start + 1; j <= i; ++j) {
            if (std::isnan(input[j])) return input[j];
            if (input[j] > maxVal) {
                maxVal = input[j];
            }
        }
        return maxVal;
    }

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
                #pragma omp simd reduction(max:maxVal)
                for (std::size_t j = start + 1; j <= i; ++j) {
                    if (input[j] > maxVal) maxVal = input[j];
                }
                output[i] = maxVal;
            }
        }
    }
};

}  // namespace quantcore
