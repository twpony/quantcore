// RedVar.h — Cross-sectional variance (截面方差, ddof=0)
// Phase: 一期实现

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RedOperator.h"

namespace quantcore {

struct RedVarOp : public RedOperator<RedVarOp> {
    static constexpr RedOpCode kOpCode = RedOpCode::RED_VAR;
    static constexpr const char* name = "red_var";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t /*i*/
    ) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();
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
        if (count < 2) return std::numeric_limits<double>::quiet_NaN();
        double mean = sum / static_cast<double>(count);
        double variance = sumSq / static_cast<double>(count) - mean * mean;
        return (variance < 0.0) ? 0.0 : variance;
    }
};

}  // namespace quantcore
