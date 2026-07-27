// CsClip.h — Cross-sectional value clipping (截面截断)
// Phase: 一期实现
//
// Clips values to [lower, upper] bounds.
//
// evaluate(values, output, lower, upper):
//   lower: minimum allowed value (per-call)
//   upper: maximum allowed value (per-call)
//
// Unlike other parameterized CS operators that store parameters at
// construction time, CsClip accepts lower/upper on each evaluate()
// call so the same instance can be reused with different bounds.
//
// NaN values map to NaN output.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/CsOperator.h"

namespace quantcore {

struct CsClipOp : public CsOperator<CsClipOp> {
    static constexpr CsOpCode kOpCode = CsOpCode::CS_CLIP;
    static constexpr const char* name = "cs_clip";

    CsClipOp() = default;

    // ============================================================
    // Per-element scalar evaluation with per-call bounds
    // ============================================================

    double evaluateScalar(ColView<double> values,
                          std::size_t i,
                          double lower,
                          double upper) const noexcept
    {
        const double* __restrict__ data = values.data();

        if (std::isnan(data[i]))
            return std::numeric_limits<double>::quiet_NaN();

        return std::max(lower, std::min(upper, data[i]));
    }

    // ============================================================
    // Batch evaluation with per-call bounds
    // ============================================================

    void evaluate(ColView<double> values,
                  double*         output,
                  double          lower,
                  double          upper) const noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]))
                output[i] = std::numeric_limits<double>::quiet_NaN();
            else
                output[i] = std::max(lower, std::min(upper, data[i]));
        }
    }
    // ============================================================
    // SIMD-dispatched evaluation with per-call bounds
    // ============================================================

    template <SimdLevel L>
    void evaluateSimd(ColView<double> values,
                      double*         output,
                      double          lower,
                      double          upper) const noexcept
    {
        const double* __restrict__ data = values.data();
        std::size_t n = values.size();
        for (std::size_t i = 0; i < n; ++i) {
            if (std::isnan(data[i]))
                output[i] = std::numeric_limits<double>::quiet_NaN();
            else
                output[i] = std::max(lower, std::min(upper, data[i]));
        }
    }
};

}  // namespace quantcore
