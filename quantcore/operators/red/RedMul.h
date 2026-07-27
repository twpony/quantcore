// RedMul.h — Cross-sectional product (截面连乘积)
// Phase: 一期实现

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RedOperator.h"

namespace quantcore {

struct RedMulOp : public RedOperator<RedMulOp> {
    static constexpr RedOpCode kOpCode = RedOpCode::RED_MUL;
    static constexpr const char* name = "red_mul";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t /*i*/
    ) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();
        double prod = 1.0;
        bool anyValid = false;
        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            prod *= data[j];
            anyValid = true;
        }
        return anyValid ? prod : std::numeric_limits<double>::quiet_NaN();
    }
};

}  // namespace quantcore
