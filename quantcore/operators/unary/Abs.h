// Abs.h — Absolute value (绝对值)
// Phase: 一期必实现
//
// Mathematical form:  y_i = |x_i|
// Reference:  numpy.abs
//
// Null propagation is handled by the UnaryOperator CRTP base class:
// when input[i] is null, output[i] is set to 0.0 and the caller
// marks it null — evaluateScalar is NOT called for null inputs.
//
// Edge cases (delegated to std::abs):
//   NaN  -> NaN
//   +Inf -> +Inf
//   -Inf -> +Inf

#pragma once

#include <cmath>
#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/operators/UnaryOperator.h"

namespace quantcore {

struct AbsOp : public UnaryOperator<AbsOp> {
    // ============================================================
    // Operator identity
    // ============================================================

    static constexpr UnaryOpCode kOpCode = UnaryOpCode::ABS;
    static constexpr const char* name = "abs";

    // ============================================================
    // Scalar reference implementation
    // ============================================================
    //
    // Computes fabs(x).  Correctly handles NaN (±NaN -> NaN),
    // ±Inf (-Inf -> +Inf, +Inf -> +Inf), signed zero (-0.0 -> 0.0),
    // and all finite values.
    //
    // This is the cross-validation baseline that SIMD kernels are
    // tested against via test_simd_cross_validate.

    static double evaluateScalar(double x) noexcept {
        return std::abs(x);
    }

    // ============================================================
    // SIMD kernel (Phase 1 — auto-vectorized scalar loop)
    // ============================================================
    //
    // Uses #pragma omp simd to hint auto-vectorization.  When OpenMP
    // is enabled (-fopenmp), the compiler attempts to vectorize the
    // loop using the target SIMD level.  Future phases will replace
    // this with hand-written intrinsics per SimdLevel.
    //
    // @param input   Input data pointer (length n, non-owning)
    // @param output  Output buffer (length n, pre-allocated, 64-byte aligned)
    // @param n       Number of elements

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::abs(input[i]);
        }
    }
};

}  // namespace quantcore
