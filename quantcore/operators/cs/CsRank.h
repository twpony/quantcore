// CsRank.h — Cross-sectional rank (截面排名)
// Phase: 一期实现
//
// Returns 1-based rank with average tie-breaking within the cross-section.
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

struct CsRankOp : public CsOperator<CsRankOp> {
    static constexpr CsOpCode kOpCode = CsOpCode::CS_RANK;
    static constexpr const char* name = "cs_rank";

    static double evaluateScalar(ColView<double> values,
                                 std::size_t i) noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        if (std::isnan(data[i]))
            return std::numeric_limits<double>::quiet_NaN();

        // Collect (value, index) for non-NaN elements
        struct Pair { double val; std::size_t idx; };
        std::vector<Pair> pairs;
        pairs.reserve(n);
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isnan(data[j]))
                pairs.push_back({data[j], j});
        }

        std::size_t count = pairs.size();
        if (count == 0)
            return std::numeric_limits<double>::quiet_NaN();

        // Sort by value
        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair& a, const Pair& b) { return a.val < b.val; });

        // Compute average ranks (handle ties)
        std::vector<double> ranks(count);
        std::size_t pos = 0;
        while (pos < count) {
            std::size_t end = pos + 1;
            while (end < count && pairs[end].val == pairs[pos].val)
                ++end;
            // Average rank for tied group: (pos+1 + end) / 2
            double avgRank = (static_cast<double>(pos + 1) + static_cast<double>(end)) / 2.0;
            for (std::size_t k = pos; k < end; ++k)
                ranks[k] = avgRank;
            pos = end;
        }

        // Map back to original index
        for (std::size_t k = 0; k < count; ++k) {
            if (pairs[k].idx == i)
                return ranks[k];
        }
        return std::numeric_limits<double>::quiet_NaN();  // unreachable
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

        // Assign average ranks: ties share the same rank
        std::size_t pos = 0;
        while (pos < count) {
            std::size_t end = pos + 1;
            while (end < count && sorted[end].first == sorted[pos].first)
                ++end;

            // Average rank for tied group [pos, end):
            //   first_rank = pos + 1, last_rank = end
            //   avg_rank = (first_rank + last_rank) / 2
            double avgRank = (static_cast<double>(pos + 1) + static_cast<double>(end)) / 2.0;

            for (std::size_t k = pos; k < end; ++k)
                output[sorted[k].second] = avgRank;

            pos = end;
        }
    }
};

}  // namespace quantcore
