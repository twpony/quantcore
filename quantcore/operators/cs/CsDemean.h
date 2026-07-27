// CsDemean.h — Cross-sectional demean (截面去均值)
// Phase: 一期实现
//
// x'_i = x_i - mean
// Mean is computed across all non-NaN values.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/CsOperator.h"

namespace quantcore {

struct CsDemeanOp : public CsOperator<CsDemeanOp> {
    static constexpr CsOpCode kOpCode = CsOpCode::CS_DEMEAN;
    static constexpr const char* name = "cs_demean";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t i) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        if (std::isnan(data[i]))
            return std::numeric_limits<double>::quiet_NaN();

        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            sum += data[j];
            ++count;
        }

        if (count == 0)
            return std::numeric_limits<double>::quiet_NaN();

        return data[i] - sum / static_cast<double>(count);
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> values,
                             double*         output) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        // First pass: compute mean of non-NaN values
        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]))
                continue;
            sum += data[i];
            ++count;
        }

        if (count == 0) {
            for (std::size_t i = 0; i < n; ++i)
                output[i] = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        double mean = sum / static_cast<double>(count);

        // Second pass: subtract mean from each element
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]))
                output[i] = std::numeric_limits<double>::quiet_NaN();
            else
                output[i] = data[i] - mean;
        }
    }
};

}  // namespace quantcore
