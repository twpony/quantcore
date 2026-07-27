// RedMean.h — Cross-sectional mean (截面均值)
// Phase: 一期实现

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RedOperator.h"

namespace quantcore {

struct RedMeanOp : public RedOperator<RedMeanOp> {
    static constexpr RedOpCode kOpCode = RedOpCode::RED_MEAN;
    static constexpr const char* name = "red_mean";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t /*i*/
    ) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();
        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            sum += data[j];
            ++count;
        }
        return (count > 0) ? sum / static_cast<double>(count)
                           : std::numeric_limits<double>::quiet_NaN();
    }
};

}  // namespace quantcore
