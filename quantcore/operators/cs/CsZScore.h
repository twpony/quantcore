// CsZScore.h — Cross-sectional z-score (截面标准化)
// Phase: 一期实现
//
// z_i = (x_i - mean) / std
//
// Mean and biased (population) standard deviation are computed across all
// non-NaN values using a two-pass algorithm for numerical stability:
//   pass 1: mean = Σx_j / N
//   pass 2: σ² = Σ(x_j - mean)² / N   (population variance)
//
// If σ² ≤ 0 or count < 2, returns 0.0.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/CsOperator.h"

namespace quantcore {

struct CsZScoreOp : public CsOperator<CsZScoreOp> {
    static constexpr CsOpCode kOpCode = CsOpCode::CS_ZSCORE;
    static constexpr const char* name = "cs_zscore";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t i) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        if (std::isnan(data[i]))
            return std::numeric_limits<double>::quiet_NaN();

        // ---- pass 1: compute mean ----
        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            sum += data[j];
            ++count;
        }

        if (count < 2) return 0.0;
        double mean = sum / static_cast<double>(count);

        // ---- pass 2: compute biased (population) variance ----
        double sumSqDev = 0.0;  // Σ (x_j - mean)^2
        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            double dev = data[j] - mean;
            sumSqDev += dev * dev;
        }
        double variance = sumSqDev / static_cast<double>(count);
        if (variance <= 0.0) return 0.0;

        return (data[i] - mean) / std::sqrt(variance);
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> values,
                             double*         output) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        // Pass 1: compute mean of non-NaN values
        double sum = 0.0;
        std::size_t count = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]))
                continue;
            sum += data[i];
            ++count;
        }

        if (count < 2) {
            for (std::size_t i = 0; i < n; ++i) {
                if (std::isnan(data[i]))
                    output[i] = std::numeric_limits<double>::quiet_NaN();
                else
                    output[i] = 0.0;
            }
            return;
        }

        double mean = sum / static_cast<double>(count);

        // Pass 2: compute biased (population) variance
        double sumSqDev = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]))
                continue;
            double dev = data[i] - mean;
            sumSqDev += dev * dev;
        }
        double variance = sumSqDev / static_cast<double>(count);

        if (variance <= 0.0) {
            for (std::size_t i = 0; i < n; ++i) {
                if (std::isnan(data[i]))
                    output[i] = std::numeric_limits<double>::quiet_NaN();
                else
                    output[i] = 0.0;
            }
            return;
        }

        double invStd = 1.0 / std::sqrt(variance);

        // Pass 3: compute z-score for each element
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]))
                output[i] = std::numeric_limits<double>::quiet_NaN();
            else
                output[i] = (data[i] - mean) * invStd;
        }
    }
};

}  // namespace quantcore
