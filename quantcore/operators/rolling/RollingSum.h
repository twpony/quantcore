// RollingSum.h — Rolling sum (滚动求和)
// Phase: 一期实现

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"

namespace quantcore {

struct RollingSumOp : public RollingOperator<RollingSumOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_SUM;
    static constexpr const char* name = "rolling_sum";

    using RollingOperator<RollingSumOp>::RollingOperator;

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
        return sum;
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                      double*         output,
                      std::size_t     window) noexcept
    {
        std::size_t n = input.size();
        if (n == 0) return;

        double runningSum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            runningSum += input[i];
            if (i + 1 < window) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                if (i >= window) {
                    runningSum -= input[i - window];
                }
                output[i] = runningSum;
            }
        }
    }
};

}  // namespace quantcore
