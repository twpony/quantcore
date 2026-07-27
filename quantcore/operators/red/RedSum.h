// RedSum.h — Cross-sectional sum (截面求和)
// Phase: 一期实现

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RedOperator.h"

namespace quantcore {

struct RedSumOp : public RedOperator<RedSumOp> {
    static constexpr RedOpCode kOpCode = RedOpCode::RED_SUM;
    static constexpr const char* name = "red_sum";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t /*i*/
    ) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();
        double sum = 0.0;
        bool anyValid = false;
        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            sum += data[j];
            anyValid = true;
        }
        return anyValid ? sum : std::numeric_limits<double>::quiet_NaN();
    }
};

}  // namespace quantcore
