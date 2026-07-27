// RedMin.h — Cross-sectional minimum (截面最小值)
// Phase: 一期实现

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RedOperator.h"

namespace quantcore {

struct RedMinOp : public RedOperator<RedMinOp> {
    static constexpr RedOpCode kOpCode = RedOpCode::RED_MIN;
    static constexpr const char* name = "red_min";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t /*i*/
    ) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();
        double minVal = std::numeric_limits<double>::quiet_NaN();
        bool anyValid = false;
        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            if (!anyValid || data[j] < minVal) {
                minVal = data[j];
                anyValid = true;
            }
        }
        return minVal;
    }
};

}  // namespace quantcore
