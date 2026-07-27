// RedQuantile.h — Cross-sectional quantile (截面分位数)
// Phase: 一期实现
//
// Returns the quantile rank (0.0 to 1.0) of each element within the
// cross-section, using "average" tie-breaking.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RedOperator.h"

namespace quantcore {

struct RedQuantileOp : public RedOperator<RedQuantileOp> {
    static constexpr RedOpCode kOpCode = RedOpCode::RED_QUANTILE;
    static constexpr const char* name = "red_quantile";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t i
    ) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        // Check if position i itself is null
        if (std::isnan(data[i]))
            return std::numeric_limits<double>::quiet_NaN();

        double current = data[i];
        std::size_t lessCount = 0;
        std::size_t equalCount = 0;
        std::size_t validCount = 0;

        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            ++validCount;
            if (data[j] < current)
                ++lessCount;
            else if (data[j] == current)
                ++equalCount;
        }

        if (validCount == 0) return std::numeric_limits<double>::quiet_NaN();

        // Average rank → percentile
        double avgRank = static_cast<double>(lessCount)
                       + (static_cast<double>(equalCount) + 1.0) / 2.0;
        return avgRank / static_cast<double>(validCount);
    }
};

}  // namespace quantcore
