// Min.h — Element-wise minimum (逐元素最小值)
// Phase: 一期必实现
//
// Mathematical form:  z_i = min(x_i, y_i)
// Reference:  numpy.minimum
//
// Null propagation is handled by the BinaryOperator CRTP base class:
// when EITHER lhs[i] or rhs[i] is null, output[i] is set to 0.0 and
// the caller marks it null — evaluateScalar is NOT called for null inputs.
//
// Input modes (handled by BinaryOperator base class):
//   1. Column vs Column:  z_i = min(lhs[i], rhs[i])
//   2. Column vs Scalar:  z_i = min(lhs[i], scalar)
//   3. Scalar vs Column:  z_i = min(scalar, rhs[i])
//
// Edge cases (delegated to std::min):
//   NaN vs any    -> NaN (NaN propagates per IEEE 754)
//   +Inf vs any   -> any
//   -Inf vs any   -> -Inf

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/BinaryOperator.h"

namespace quantcore {

struct MinOp : public BinaryOperator<MinOp> {
    // ============================================================
    // Operator identity
    // ============================================================

    static constexpr BinaryOpCode kOpCode = BinaryOpCode::MIN;
    static constexpr const char* name = "min";

    // ============================================================
    // Scalar reference implementation
    // ============================================================
    //
    // Computes std::min(a, b).  Correctly handles NaN (propagates),
    // ±Inf, signed zero, and all finite values.
    //
    // This is the cross-validation baseline that SIMD kernels are
    // tested against via test_simd_cross_validate.

    static double evaluateScalar(double a, double b) noexcept {
        // std::min does NOT propagate NaN — if either operand is NaN,
        // NaN must be returned (matching numpy.minimum semantics).
        if (std::isnan(a) || std::isnan(b))
            return std::numeric_limits<double>::quiet_NaN();
        return std::min(a, b);
    }

    // ============================================================
    // SIMD kernel (Phase 1 — auto-vectorized scalar loop)
    // ============================================================
    //
    // Uses #pragma omp simd to hint auto-vectorization.  Future phases
    // will replace this with hand-written intrinsics per SimdLevel.
    //
    // @param lhs     Left-hand-side data pointer (length n)
    // @param rhs     Right-hand-side data pointer (length n)
    // @param output  Output buffer (length n, pre-allocated, 64-byte aligned)
    // @param n       Number of elements

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> lhs,
                             ColView<double> rhs,
                             double*         output) noexcept {
        const double* __restrict__ lhsData = lhs.data();
        const double* __restrict__ rhsData = rhs.data();
        std::size_t n = lhs.size();
        for (std::size_t i = 0; i < n; ++i) {
            double a = lhsData[i], b = rhsData[i];
            if (std::isnan(a) || std::isnan(b))
                output[i] = std::numeric_limits<double>::quiet_NaN();
            else
                output[i] = std::min(a, b);
        }
    }
};

}  // namespace quantcore
