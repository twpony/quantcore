// CsWinsorizeMAD.h — Cross-sectional MAD-based winsorization (截面MAD缩尾)
// Phase: 一期实现
//
// Clips values to [median - n*MAD, median + n*MAD].
//
// evaluate(values, output, n):
//   n: number of MADs from median (e.g. 3.0 = 3-sigma-like threshold)
//
//   Step 1: median = median of non-NaN values
//   Step 2: MAD = median(|x_i - median|)  (Median Absolute Deviation)
//   Step 3: lower = median - n * MAD
//           upper = median + n * MAD
//   Step 4: output[i] = clip(data[i], lower, upper)
//
// NaN values are skipped in all computations and map to NaN output.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/CsOperator.h"

namespace quantcore {

struct CsWinsorizeMADOp : public CsOperator<CsWinsorizeMADOp> {
    static constexpr CsOpCode kOpCode = CsOpCode::CS_WINSORIZE_MAD;
    static constexpr const char* name = "cs_winsorize_mad";

    CsWinsorizeMADOp() = default;

    // ============================================================
    // Per-element scalar evaluation with per-call n
    // ============================================================

    double evaluateScalar(ColView<double> values,
                          std::size_t i,
                          double n) const noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t size = values.size();

        if (std::isnan(data[i]))
            return std::numeric_limits<double>::quiet_NaN();

        // Collect non-NaN values
        std::vector<double> sorted;
        sorted.reserve(size);
        for (std::size_t j = 0; j < size; ++j) {
            if (!std::isnan(data[j]))
                sorted.push_back(data[j]);
        }

        std::size_t count = sorted.size();
        if (count == 0)
            return std::numeric_limits<double>::quiet_NaN();

        std::sort(sorted.begin(), sorted.end());

        // Compute median
        double median = sorted[count / 2];
        if (count % 2 == 0)
            median = (sorted[count / 2 - 1] + median) * 0.5;

        // Compute MAD = median(|x_i - median|)
        std::vector<double> absDev;
        absDev.reserve(count);
        for (std::size_t j = 0; j < count; ++j)
            absDev.push_back(std::abs(sorted[j] - median));
        std::sort(absDev.begin(), absDev.end());

        double mad = absDev[count / 2];
        if (count % 2 == 0)
            mad = (absDev[count / 2 - 1] + mad) * 0.5;

        if (mad == 0.0) return data[i];

        double lo = median - n * mad;
        double hi = median + n * mad;

        return std::max(lo, std::min(hi, data[i]));
    }

    // ============================================================
    // Batch evaluation with per-call n
    // ============================================================

    void evaluate(ColView<double> values,
                  double*         output,
                  double          n) const noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t size = values.size();

        // Collect non-NaN values
        std::vector<double> sorted;
        sorted.reserve(size);
        for (std::size_t j = 0; j < size; ++j) {
            if (!std::isnan(data[j]))
                sorted.push_back(data[j]);
        }

        std::size_t count = sorted.size();
        if (count == 0) {
            for (std::size_t i = 0; i < size; ++i)
                output[i] = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        std::sort(sorted.begin(), sorted.end());

        // ---- median ----
        double median = sorted[count / 2];
        if (count % 2 == 0)
            median = (sorted[count / 2 - 1] + median) * 0.5;

        // ---- MAD = median(|x_i - median|) ----
        std::vector<double> absDev;
        absDev.reserve(count);
        for (std::size_t j = 0; j < count; ++j)
            absDev.push_back(std::abs(sorted[j] - median));
        std::sort(absDev.begin(), absDev.end());

        double mad = absDev[count / 2];
        if (count % 2 == 0)
            mad = (absDev[count / 2 - 1] + mad) * 0.5;

        if (mad == 0.0) {
            for (std::size_t i = 0; i < size; ++i)
                output[i] = std::isnan(data[i])
                    ? std::numeric_limits<double>::quiet_NaN()
                    : data[i];
            return;
        }

        double lo = median - n * mad;
        double hi = median + n * mad;

        for (std::size_t i = 0; i < size; ++i) {
            if (std::isnan(data[i]))
                output[i] = std::numeric_limits<double>::quiet_NaN();
            else
                output[i] = std::max(lo, std::min(hi, data[i]));
        }
    }
    // ============================================================
    // SIMD-dispatched evaluation with per-call n
    // ============================================================

    template <SimdLevel L>
    void evaluateSimd(ColView<double> values,
                      double*         output,
                      double          n) const noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t size = values.size();

        // Collect non-NaN values
        std::vector<double> sorted;
        sorted.reserve(size);
        for (std::size_t i = 0; i < size; ++i) {
            if (!std::isnan(data[i]))
                sorted.push_back(data[i]);
        }

        std::size_t count = sorted.size();
        if (count == 0) {
            for (std::size_t i = 0; i < size; ++i)
                output[i] = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        std::sort(sorted.begin(), sorted.end());

        // Median
        double median = sorted[count / 2];
        if (count % 2 == 0)
            median = (sorted[count / 2 - 1] + median) * 0.5;

        // MAD = median(|x_i - median|)
        std::vector<double> absDev;
        absDev.reserve(count);
        for (std::size_t j = 0; j < count; ++j)
            absDev.push_back(std::abs(sorted[j] - median));
        std::sort(absDev.begin(), absDev.end());

        double mad = absDev[count / 2];
        if (count % 2 == 0)
            mad = (absDev[count / 2 - 1] + mad) * 0.5;

        if (mad == 0.0) {
            for (std::size_t i = 0; i < size; ++i)
                output[i] = std::isnan(data[i])
                    ? std::numeric_limits<double>::quiet_NaN()
                    : data[i];
            return;
        }

        double lo = median - n * mad;
        double hi = median + n * mad;

        for (std::size_t i = 0; i < size; ++i) {
            if (std::isnan(data[i]))
                output[i] = std::numeric_limits<double>::quiet_NaN();
            else
                output[i] = std::max(lo, std::min(hi, data[i]));
        }
    }
};

}  // namespace quantcore
