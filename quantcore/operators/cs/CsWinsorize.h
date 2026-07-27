// CsWinsorize.h — Cross-sectional winsorization (截面缩尾)
// Phase: 一期实现
//
// Clips values at the lowerPct and upperPct percentile thresholds
// computed from the cross-section.
//
// evaluate(values, output, lowerPct, upperPct):
//   lowerPct: lower percentile threshold (e.g. 0.01 = 1%)
//   upperPct: upper percentile threshold (e.g. 0.99 = 99%)
//
// Parameters are passed per-call so the same instance can be reused
// with different thresholds.
//
// NaN values are skipped in percentile computation and map to NaN output.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/CsOperator.h"

namespace quantcore {

struct CsWinsorizeOp : public CsOperator<CsWinsorizeOp> {
    static constexpr CsOpCode kOpCode = CsOpCode::CS_WINSORIZE;
    static constexpr const char* name = "cs_winsorize";

    CsWinsorizeOp() = default;

    // ============================================================
    // Per-element scalar evaluation with per-call thresholds
    // ============================================================

    double evaluateScalar(ColView<double> values,
                          std::size_t i,
                          double lowerPct,
                          double upperPct) const noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        if (std::isnan(data[i]))
            return std::numeric_limits<double>::quiet_NaN();

        // Collect non-NaN values and compute percentile thresholds
        std::vector<double> sorted;
        sorted.reserve(n);
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isnan(data[j]))
                sorted.push_back(data[j]);
        }

        std::size_t count = sorted.size();
        if (count == 0)
            return std::numeric_limits<double>::quiet_NaN();

        std::sort(sorted.begin(), sorted.end());

        auto percentile = [&](double pct) -> double {
            double pos = pct * static_cast<double>(count - 1);
            std::size_t lo = static_cast<std::size_t>(pos);
            std::size_t hi = (lo + 1 < count) ? lo + 1 : lo;
            double frac = pos - static_cast<double>(lo);
            return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
        };

        double lo = percentile(lowerPct);
        double hi = percentile(upperPct);

        return std::max(lo, std::min(hi, data[i]));
    }

    // ============================================================
    // Batch evaluation with per-call thresholds
    // ============================================================

    void evaluate(ColView<double> values,
                  double*         output,
                  double          lowerPct,
                  double          upperPct) const noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        // Collect and sort non-NaN values once
        std::vector<double> sorted;
        sorted.reserve(n);
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isnan(data[j]))
                sorted.push_back(data[j]);
        }

        std::size_t count = sorted.size();
        if (count == 0) {
            for (std::size_t i = 0; i < n; ++i)
                output[i] = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        std::sort(sorted.begin(), sorted.end());

        auto percentile = [&](double pct) -> double {
            double pos = pct * static_cast<double>(count - 1);
            std::size_t lo = static_cast<std::size_t>(pos);
            std::size_t hi = (lo + 1 < count) ? lo + 1 : lo;
            double frac = pos - static_cast<double>(lo);
            return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
        };

        double lo = percentile(lowerPct);
        double hi = percentile(upperPct);

        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]))
                output[i] = std::numeric_limits<double>::quiet_NaN();
            else
                output[i] = std::max(lo, std::min(hi, data[i]));
        }
    }
    // ============================================================
    // SIMD-dispatched evaluation with per-call thresholds
    // ============================================================

    template <SimdLevel L>
    void evaluateSimd(ColView<double> values,
                      double*         output,
                      double          lowerPct,
                      double          upperPct) const noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();

        // Collect and sort non-NaN values
        std::vector<double> sorted;
        sorted.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isnan(data[i]))
                sorted.push_back(data[i]);
        }

        std::size_t count = sorted.size();
        if (count == 0) {
            for (std::size_t i = 0; i < n; ++i)
                output[i] = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        std::sort(sorted.begin(), sorted.end());

        // Percentile via linear interpolation
        auto percentile = [&](double pct) -> double {
            double pos = pct * static_cast<double>(count - 1);
            std::size_t lo = static_cast<std::size_t>(pos);
            std::size_t hi = (lo + 1 < count) ? lo + 1 : lo;
            double frac = pos - static_cast<double>(lo);
            return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
        };

        double lo = percentile(lowerPct);
        double hi = percentile(upperPct);

        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]))
                output[i] = std::numeric_limits<double>::quiet_NaN();
            else
                output[i] = std::max(lo, std::min(hi, data[i]));
        }
    }
};

}  // namespace quantcore
