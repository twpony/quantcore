// RedZScore.h — Cross-sectional z-score (截面标准化)
// Phase: 一期实现
//
// zscore[i] = (values[i] - mean) / std
// where mean and std are computed across all non-null assets.
// If std == 0 or count < 2, returns 0.0.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RedOperator.h"

namespace quantcore {

struct RedZScoreOp : public RedOperator<RedZScoreOp> {
    static constexpr RedOpCode kOpCode = RedOpCode::RED_ZSCORE;
    static constexpr const char* name = "red_zscore";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t i
    ) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        // Check if position i itself is null
        if (std::isnan(data[i]))
            return std::numeric_limits<double>::quiet_NaN();

        // Compute cross-sectional mean and std (skip nulls)
        double sum = 0.0, sumSq = 0.0;
        std::size_t count = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            double v = data[j];
            sum += v;
            sumSq += v * v;
            ++count;
        }
        if (count < 2) return 0.0;  // insufficient data

        double mean = sum / static_cast<double>(count);
        double variance = sumSq / static_cast<double>(count) - mean * mean;
        if (variance <= 0.0) return 0.0;  // constant → z=0
        return (data[i] - mean) / std::sqrt(variance);
    }
};

}  // namespace quantcore
