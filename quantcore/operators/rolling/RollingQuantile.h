// RollingQuantile.h — Rolling quantile (滚动分位数)
// Phase: 一期实现
//
// Mathematical form:  rolling_quantile(X, n, q)[i] = q-th quantile of window
// Reference:  pandas.Series.rolling(n).quantile(q, interpolation="linear")
//
// Computes the q-th quantile (0 <= q <= 1) of the most recent `n`
// observations using linear interpolation between adjacent sorted values.
// For i < n-1, evaluateScalar returns NaN.
//
// Linear interpolation:  pos = q * (n - 1)
//   lower = floor(pos), upper = ceil(pos)
//   if lower == upper: sorted[lower]
//   else: sorted[lower] * (1 - fraction) + sorted[upper] * fraction
//
// This operator takes an extra constructor parameter `q` in addition to
// the window size.  evaluateScalar is a non-static member function to
// access q_.
//
// Null propagation: if any input in the window is null, the engine
// marks output[i] as null.  evaluateScalar assumes clean input.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <set>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"

namespace quantcore {

struct RollingQuantileOp : public RollingOperator<RollingQuantileOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_QUANTILE;
    static constexpr const char* name = "rolling_quantile";

    // ============================================================
    // Construction
    // ============================================================

    /// @param window  Number of consecutive observations in the window.
    /// @param q       Quantile to compute (0.0 to 1.0).
    explicit RollingQuantileOp(std::size_t window, double q) noexcept
        : RollingOperator<RollingQuantileOp>(window), q_(q)
    {}

    /// Access the quantile parameter.
    double q() noexcept { return q_; }

    // ============================================================
    // Scalar reference implementation (non-static: uses q_)
    // ============================================================

    double evaluateScalar(const double* input,
                          std::size_t i,
                          std::size_t window) const noexcept
    {
        if (i + 1 < window) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::size_t start = i + 1 - window;

        // Copy window values and sort
        std::vector<double> buf(window);
        for (std::size_t j = 0; j < window; ++j) {
            buf[j] = input[start + j];
        }
        std::sort(buf.begin(), buf.end());

        // Linear interpolation quantile
        double pos = q_ * static_cast<double>(window - 1);
        std::size_t lowerIdx = static_cast<std::size_t>(pos);
        std::size_t upperIdx = (lowerIdx + 1 < window) ? lowerIdx + 1 : lowerIdx;
        double fraction = pos - static_cast<double>(lowerIdx);

        if (lowerIdx == upperIdx) {
            return buf[lowerIdx];
        }
        return buf[lowerIdx] * (1.0 - fraction) + buf[upperIdx] * fraction;
    }

    // ============================================================
    // SIMD kernel — multiset sliding window O(n log w)
    // ============================================================
    //
    // Maintains a std::multiset<double> of the current window (sorted).
    // Each window slide costs O(log w): erase old + insert new.
    // Quantile lookup uses std::advance to reach the interpolation
    // positions, O(w) in the worst case but avoids O(w log w) sort.
    //
    // For small-to-moderate windows this is significantly faster than
    // the per-element copy+sort in evaluateScalar.

    template <SimdLevel L>
    void evaluateSimd(ColView<double> input,
                      double*         output,
                      std::size_t     window) noexcept
    {
        std::size_t n = input.size();
        if (n == 0) return;

        const double* data = input.data();
        double nan = std::numeric_limits<double>::quiet_NaN();

        // First (window-1) positions are invalid
        for (std::size_t i = 0; i + 1 < window && i < n; ++i) {
            output[i] = nan;
        }
        if (n < window) return;

        // Build initial sorted window
        std::multiset<double> sorted;
        for (std::size_t i = 0; i < window; ++i) {
            sorted.insert(data[i]);
        }

        // Quantile interpolation helper (captures q_)
        auto quantile = [this, window](const std::multiset<double>& s) -> double {
            double pos = q_ * static_cast<double>(window - 1);
            std::size_t lowerIdx = static_cast<std::size_t>(pos);
            std::size_t upperIdx = (lowerIdx + 1 < window) ? lowerIdx + 1 : lowerIdx;
            double fraction = pos - static_cast<double>(lowerIdx);

            if (lowerIdx == upperIdx) {
                auto it = s.begin();
                std::advance(it, lowerIdx);
                return *it;
            }
            auto it = s.begin();
            std::advance(it, lowerIdx);
            double lower = *it;
            ++it;
            double upper = *it;
            return lower * (1.0 - fraction) + upper * fraction;
        };

        output[window - 1] = quantile(sorted);

        // Slide the window
        for (std::size_t i = window; i < n; ++i) {
            sorted.erase(sorted.find(data[i - window]));
            sorted.insert(data[i]);
            output[i] = quantile(sorted);
        }
    }

private:
    double q_;
};

}  // namespace quantcore
