// RollingRank.h — Rolling rank (滚动排名, 窗口内百分位)
// Phase: 一期实现

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"

namespace quantcore {

struct RollingRankOp : public RollingOperator<RollingRankOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_RANK;
    static constexpr const char* name = "rolling_rank";

    using RollingOperator<RollingRankOp>::RollingOperator;

    static double evaluateScalar(const double* input,
                                 std::size_t i,
                                 std::size_t window) noexcept
    {
        if (i + 1 < window) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::size_t start = i + 1 - window;
        double current = input[i];

        std::size_t lessCount = 0;
        std::size_t equalCount = 0;

        for (std::size_t j = start; j <= i; ++j) {
            if (input[j] < current) {
                ++lessCount;
            } else if (input[j] == current) {
                ++equalCount;
            }
        }

        double avgRank = static_cast<double>(lessCount)
                       + (static_cast<double>(equalCount) + 1.0) / 2.0;

        return avgRank / static_cast<double>(window);
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                      double*         output,
                      std::size_t     window) noexcept
    {
        std::size_t n = input.size();
        double invWindow = 1.0 / static_cast<double>(window);

        for (std::size_t i = 0; i < n; ++i) {
            if (i + 1 < window) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                std::size_t start = i + 1 - window;
                double current = input[i];
                std::size_t lessCount = 0;
                std::size_t equalCount = 0;

                for (std::size_t j = start; j <= i; ++j) {
                    if (input[j] < current) {
                        ++lessCount;
                    } else if (input[j] == current) {
                        ++equalCount;
                    }
                }

                double avgRank = static_cast<double>(lessCount)
                               + (static_cast<double>(equalCount) + 1.0) / 2.0;
                output[i] = avgRank * invWindow;
            }
        }
    }
};

}  // namespace quantcore
