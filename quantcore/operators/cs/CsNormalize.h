// CsNormalize.h — Cross-sectional min-max normalization (截面归一化)
// Phase: 一期实现
//
// x'_i = (x_i - min) / (max - min)  →  [0, 1]
// Min and max are computed across all non-NaN values.
// If max == min or no valid data, returns 0.0.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/CsOperator.h"

namespace quantcore {

struct CsNormalizeOp : public CsOperator<CsNormalizeOp> {
    static constexpr CsOpCode kOpCode = CsOpCode::CS_NORMALIZE;
    static constexpr const char* name = "cs_normalize";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t i) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        if (std::isnan(data[i]))
            return std::numeric_limits<double>::quiet_NaN();

        double minVal = std::numeric_limits<double>::infinity();
        double maxVal = -std::numeric_limits<double>::infinity();
        bool anyValid = false;

        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            minVal = std::min(minVal, data[j]);
            maxVal = std::max(maxVal, data[j]);
            anyValid = true;
        }

        if (!anyValid || maxVal == minVal)
            return 0.0;

        return (data[i] - minVal) / (maxVal - minVal);
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> values,
                             double*         output) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        // First pass: find min / max of non-NaN values
        double minVal = std::numeric_limits<double>::infinity();
        double maxVal = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]))
                continue;
            if (data[i] < minVal) minVal = data[i];
            if (data[i] > maxVal) maxVal = data[i];
        }

        bool noRange = (minVal == std::numeric_limits<double>::infinity())  // no valid data
                       || (maxVal == minVal);

        // Second pass: normalize each element
        if (noRange) {
            for (std::size_t i = 0; i < n; ++i) {
                if (std::isnan(data[i]))
                    output[i] = std::numeric_limits<double>::quiet_NaN();
                else
                    output[i] = 0.0;
            }
        } else {
            double range = maxVal - minVal;
            for (std::size_t i = 0; i < n; ++i) {
                if (std::isnan(data[i]))
                    output[i] = std::numeric_limits<double>::quiet_NaN();
                else
                    output[i] = (data[i] - minVal) / range;
            }
        }
    }
};

}  // namespace quantcore
