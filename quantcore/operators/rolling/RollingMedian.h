// RollingMedian.h — Rolling median (滚动中位数)
// Phase: 一期实现
//
// Mathematical form:  rolling_median(X, n)[i] = median of window
// Reference:  pandas.Series.rolling(n).median()  — verified match
//
// For odd-sized windows, returns the middle element after sorting.
// For even-sized windows, returns the average of the two middle elements.
// For i < n-1, evaluateScalar returns NaN.
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

struct RollingMedianOp : public RollingOperator<RollingMedianOp> {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_MEDIAN;
    static constexpr const char* name = "rolling_median";

    using RollingOperator<RollingMedianOp>::RollingOperator;

    // ============================================================
    // Scalar reference implementation
    // ============================================================

    static double evaluateScalar(const double* input,
                                 std::size_t i,
                                 std::size_t window) noexcept
    {
        if (i + 1 < window) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::size_t start = i + 1 - window;

        // Copy window values and find median via partial sort
        std::vector<double> buf(window);
        for (std::size_t j = 0; j < window; ++j) {
            buf[j] = input[start + j];
        }

        std::size_t mid = window / 2;
        std::nth_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(mid), buf.end());

        if (window % 2 == 1) {
            // Odd window: middle element
            return buf[mid];
        } else {
            // Even window: average of two middle elements
            double a = buf[mid];
            double b = *std::max_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(mid));
            return (a + b) / 2.0;
        }
    }

    // ============================================================
    // SIMD kernel — two-multiset balanced-streaming median O(n log w)
    // ============================================================
    //
    // Maintains two std::multiset objects: lo (lower half) and hi (upper half).
    // Invariant: lo.size() == hi.size() or lo.size() == hi.size() + 1.
    // Median is *lo.rbegin() (odd window) or avg(*lo.rbegin(), *hi.begin()) (even).
    //
    // Each window slide costs O(log w): one erase + one insert + rebalance.

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
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

        std::multiset<double> lo, hi;

        // Helper: keep size invariant  lo.size() == hi.size() or lo.size() == hi.size()+1
        auto rebalance = [&]() {
            if (lo.size() > hi.size() + 1) {
                hi.insert(*lo.rbegin());
                lo.erase(std::prev(lo.end()));
            } else if (hi.size() > lo.size()) {
                lo.insert(*hi.begin());
                hi.erase(hi.begin());
            }
        };

        // Helper: insert a value into the correct half
        auto insert_val = [&](double v) {
            if (lo.empty() || v <= *lo.rbegin()) {
                lo.insert(v);
            } else {
                hi.insert(v);
            }
            rebalance();
        };

        // Helper: remove a value (must exist in exactly one of lo or hi)
        auto remove_val = [&](double v) {
            auto it = lo.find(v);
            if (it != lo.end()) {
                lo.erase(it);
            } else {
                hi.erase(hi.find(v));
            }
            rebalance();
        };

        // Helper: compute median from the two multisets
        auto median = [&]() -> double {
            if (lo.size() > hi.size()) {
                return *lo.rbegin();                       // odd window
            }
            return (*lo.rbegin() + *hi.begin()) * 0.5;    // even window
        };

        // Fill the first window
        for (std::size_t i = 0; i < window; ++i) {
            insert_val(data[i]);
        }
        output[window - 1] = median();

        // Slide the window
        for (std::size_t i = window; i < n; ++i) {
            remove_val(data[i - window]);
            insert_val(data[i]);
            output[i] = median();
        }
    }
};

}  // namespace quantcore
