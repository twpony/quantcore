// RedMax.h — Cross-sectional maximum (截面最大值)
// Phase: 一期实现

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RedOperator.h"

namespace quantcore {

struct RedMaxOp : public RedOperator<RedMaxOp> {
    static constexpr RedOpCode kOpCode = RedOpCode::RED_MAX;
    static constexpr const char* name = "red_max";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t /*i*/
    ) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();
        double maxVal = std::numeric_limits<double>::quiet_NaN();
        bool anyValid = false;
        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            if (!anyValid || data[j] > maxVal) {
                maxVal = data[j];
                anyValid = true;
            }
        }
        return maxVal;
    }
};

}  // namespace quantcore
