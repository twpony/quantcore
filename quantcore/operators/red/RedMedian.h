// RedMedian.h — Cross-sectional median (截面中位数)
// Phase: 一期实现

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RedOperator.h"

namespace quantcore {

struct RedMedianOp : public RedOperator<RedMedianOp> {
    static constexpr RedOpCode kOpCode = RedOpCode::RED_MEDIAN;
    static constexpr const char* name = "red_median";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t /*i*/
    ) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        // Collect non-NaN values
        std::vector<double> sorted;
        sorted.reserve(n);
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isnan(data[j]))
                sorted.push_back(data[j]);
        }

        std::size_t count = sorted.size();
        if (count == 0)
            return std::numeric_limits<double>::quiet_NaN();

        // nth_element for O(n) median (no need to fully sort)
        if (count % 2 == 1) {
            auto mid = sorted.begin() + count / 2;
            std::nth_element(sorted.begin(), mid, sorted.end());
            return *mid;
        } else {
            auto mid1 = sorted.begin() + count / 2 - 1;
            auto mid2 = sorted.begin() + count / 2;
            std::nth_element(sorted.begin(), mid1, sorted.end());
            double a = *mid1;
            std::nth_element(sorted.begin(), mid2, sorted.end());
            double b = *mid2;
            return (a + b) / 2.0;
        }
    }
};

}  // namespace quantcore
