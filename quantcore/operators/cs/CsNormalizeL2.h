// CsNormalizeL2.h — Cross-sectional L2 normalization (截面L2归一化)
// Phase: 一期实现
//
// x'_i = x_i / sqrt(sum(x_j^2))  —  unit L2 norm
// If the L2 norm is 0, returns 0.0.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/CsOperator.h"

namespace quantcore {

struct CsNormalizeL2Op : public CsOperator<CsNormalizeL2Op> {
    static constexpr CsOpCode kOpCode = CsOpCode::CS_NORMALIZE_L2;
    static constexpr const char* name = "cs_normalize_l2";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t i) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        if (std::isnan(data[i]))
            return std::numeric_limits<double>::quiet_NaN();

        double sumSq = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            sumSq += data[j] * data[j];
        }

        double norm = std::sqrt(sumSq);
        if (norm == 0.0) return 0.0;

        return data[i] / norm;
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> values,
                             double*         output) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        // First pass: compute sum of squares
        double sumSq = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]))
                continue;
            sumSq += data[i] * data[i];
        }

        double norm = std::sqrt(sumSq);

        // Second pass: divide each element by L2 norm
        if (norm == 0.0) {
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
                    output[i] = data[i] / norm;
            }
        }
    }
};

}  // namespace quantcore
