// CsQuantile.h — Cross-sectional quantile (截面分位数)
// Phase: 一期实现
//
// Returns quantile rank in [0, 1] for each element within the cross-section.
// Uses average tie-breaking: ties get the same quantile.
// NaN values are skipped and map to NaN output.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/CsOperator.h"

namespace quantcore {

struct CsQuantileOp : public CsOperator<CsQuantileOp> {
    static constexpr CsOpCode kOpCode = CsOpCode::CS_QUANTILE;
    static constexpr const char* name = "cs_quantile";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t i) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        if (std::isnan(data[i]))
            return std::numeric_limits<double>::quiet_NaN();

        double current = data[i];
        std::size_t lessCount = 0;
        std::size_t equalCount = 0;
        std::size_t validCount = 0;

        for (std::size_t j = 0; j < n; ++j) {
            if (std::isnan(data[j]))
                continue;
            ++validCount;
            if (data[j] < current)
                ++lessCount;
            else if (data[j] == current)
                ++equalCount;
        }

        if (validCount == 0)
            return std::numeric_limits<double>::quiet_NaN();

        // Average rank → quantile (0 to 1)
        double avgRank = static_cast<double>(lessCount)
                       + (static_cast<double>(equalCount) + 1.0) / 2.0;
        return avgRank / static_cast<double>(validCount);
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> values,
                             double*         output) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        // Collect (value, original_index) for non-NaN entries
        std::vector<std::pair<double, std::size_t>> sorted;
        sorted.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isnan(data[i]))
                sorted.emplace_back(data[i], i);
            else
                output[i] = std::numeric_limits<double>::quiet_NaN();
        }

        std::size_t count = sorted.size();
        if (count == 0)
            return;

        // Sort by value
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // Assign quantiles: ties share the average rank
        double denom = static_cast<double>(count);
        std::size_t start = 0;
        while (start < count) {
            std::size_t end = start;
            while (end + 1 < count && sorted[end + 1].first == sorted[start].first)
                ++end;

            // Average rank of positions [start, end] (0-indexed):
            //   rank_first = start + 1, rank_last = end + 1
            //   avg_rank = (rank_first + rank_last) / 2 = (start + end + 2) / 2
            double quantile = static_cast<double>(start + end + 2)
                            / (2.0 * denom);

            for (std::size_t k = start; k <= end; ++k)
                output[sorted[k].second] = quantile;

            start = end + 1;
        }
    }
};

}  // namespace quantcore
