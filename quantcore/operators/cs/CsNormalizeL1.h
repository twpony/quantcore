// CsNormalizeL1.h — Cross-sectional L1 normalization (截面L1归一化)
// Phase: 一期实现
//
// x'_i = x_i / sum(|x_j|)  —  unit L1 norm
// If the L1 norm (sum of absolute values) is 0, returns 0.0.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/CsOperator.h"

namespace quantcore {

struct CsNormalizeL1Op : public CsOperator<CsNormalizeL1Op> {
    static constexpr CsOpCode kOpCode = CsOpCode::CS_NORMALIZE_L1;
    static constexpr const char* name = "cs_normalize_l1";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t i) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        if (std::isnan(data[i]))
            return std::numeric_limits<double>::quiet_NaN();

        double sumAbs = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            sumAbs += std::abs(data[j]);
        }

        if (sumAbs == 0.0) return 0.0;

        return data[i] / sumAbs;
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> values,
                             double*         output) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        // First pass: compute L1 norm (sum of absolute values)
        double sumAbs = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]))
                continue;
            sumAbs += std::abs(data[i]);
        }

        // Second pass: divide each element by L1 norm
        if (sumAbs == 0.0) {
            for (std::size_t i = 0; i < n; ++i) {
                if (std::isnan(data[i]))
                    output[i] = std::numeric_limits<double>::quiet_NaN();
                else
                    output[i] = 0.0;
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                if (std::isnan(data[i]))
                    output[i] = std::numeric_limits<double>::quiet_NaN();
                else
                    output[i] = data[i] / sumAbs;
            }
        }
    }
};

}  // namespace quantcore
