// Div.h — Element-wise division (逐元素除法)
// Phase: 一期必实现
//
// Mathematical form:  z_i = x_i / y_i
// Reference:  numpy.divide
//
// Null propagation is handled by the BinaryOperator CRTP base class.
//
// Input modes (handled by BinaryOperator base class):
//   1. Column vs Column:  z_i = lhs[i] / rhs[i]
//   2. Column vs Scalar:  z_i = lhs[i] / scalar
//   3. Scalar vs Column:  z_i = scalar / rhs[i]
//
// Note: Division is non-commutative — Scalar vs Column gives
//       different results from Column vs Scalar.
//
// Edge cases (IEEE 754):
//   x / 0.0     -> ±Inf (x ≠ 0), NaN (0/0)
//   ±Inf / ±Inf -> NaN
//   NaN / any   -> NaN
//   Overflow     -> ±Inf

#pragma once

#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/operators/BinaryOperator.h"

namespace quantcore {

struct DivOp : public BinaryOperator<DivOp> {
    static constexpr BinaryOpCode kOpCode = BinaryOpCode::DIV;
    static constexpr const char* name = "div";

    static double evaluateScalar(double a, double b) noexcept {
        return a / b;
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
            output[i] = lhsData[i] / rhsData[i];
        }
    }
};

}  // namespace quantcore
