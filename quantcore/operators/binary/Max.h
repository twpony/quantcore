// Max.h — Element-wise maximum (逐元素最大值)
// Phase: 一期必实现
//
// Mathematical form:  z_i = max(x_i, y_i)
// Reference:  numpy.maximum
//
// Null propagation is handled by the BinaryOperator CRTP base class:
// when EITHER lhs[i] or rhs[i] is null, output[i] is set to 0.0 and
// the caller marks it null — evaluateScalar is NOT called for null inputs.
//
// Input modes (handled by BinaryOperator base class):
//   1. Column vs Column:  z_i = max(lhs[i], rhs[i])
//   2. Column vs Scalar:  z_i = max(lhs[i], scalar)
//   3. Scalar vs Column:  z_i = max(scalar, rhs[i])
//
// Edge cases (delegated to std::max):
//   NaN vs any    -> NaN (NaN propagates per IEEE 754)
//   +Inf vs any   -> +Inf
//   -Inf vs -Inf  -> -Inf

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/BinaryOperator.h"

namespace quantcore {

struct MaxOp : public BinaryOperator<MaxOp> {
    // ============================================================
    // Operator identity
    // ============================================================

    static constexpr BinaryOpCode kOpCode = BinaryOpCode::MAX;
    static constexpr const char* name = "max";

    // ============================================================
    // Scalar reference implementation
    // ============================================================
    //
    // Computes std::max(a, b).  Correctly handles NaN (propagates),
    // ±Inf, signed zero, and all finite values.
    //
    // This is the cross-validation baseline that SIMD kernels are
    // tested against via test_simd_cross_validate.

    static double evaluateScalar(double a, double b) noexcept {
        // std::max does NOT propagate NaN — if either operand is NaN,
        // NaN must be returned (matching numpy.maximum semantics).
        if (std::isnan(a) || std::isnan(b))
            return std::numeric_limits<double>::quiet_NaN();
        return std::max(a, b);
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
                output[i] = std::max(a, b);
        }
    }
};

}  // namespace quantcore
