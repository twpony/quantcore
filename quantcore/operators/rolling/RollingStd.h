// RollingStd.h — Rolling standard deviation (滚动标准差)
// Phase: 一期实现

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"

namespace quantcore {

struct RollingStdOp : public RollingOperator<RollingStdOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_STD;
    static constexpr const char* name = "rolling_std";

    using RollingOperator<RollingStdOp>::RollingOperator;

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
        return std::sqrt(sumSqDiff / static_cast<double>(count));
    }

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
                if (variance < 0.0) variance = 0.0;
                output[i] = std::sqrt(variance);
            }
        }
    }
};

}  // namespace quantcore
