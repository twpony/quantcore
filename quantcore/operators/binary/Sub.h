// Sub.h — Element-wise subtraction (逐元素减法)
// Phase: 一期必实现
//
// Mathematical form:  z_i = x_i - y_i
// Reference:  numpy.subtract
//
// Null propagation is handled by the BinaryOperator CRTP base class.
//
// Input modes (handled by BinaryOperator base class):
//   1. Column vs Column:  z_i = lhs[i] - rhs[i]
//   2. Column vs Scalar:  z_i = lhs[i] - scalar
//   3. Scalar vs Column:  z_i = scalar - rhs[i]
//
// Note: Subtraction is non-commutative — Scalar vs Column gives
//       different results from Column vs Scalar.
//
// Edge cases (IEEE 754):
//   ±Inf - ±Inf  -> NaN (same signs) or ±Inf (opposite signs)
//   NaN - any    -> NaN
//   Overflow      -> ±Inf

#pragma once

#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/operators/BinaryOperator.h"

namespace quantcore {

struct SubOp : public BinaryOperator<SubOp> {
    static constexpr BinaryOpCode kOpCode = BinaryOpCode::SUB;
    static constexpr const char* name = "sub";

    static double evaluateScalar(double a, double b) noexcept {
        return a - b;
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> lhs,
                             ColView<double> rhs,
                             double*         output) noexcept {
        const double* __restrict__ lhsData = lhs.data();
        const double* __restrict__ rhsData = rhs.data();
        std::size_t n = lhs.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = lhsData[i] - rhsData[i];
        }
    }
};

}  // namespace quantcore
