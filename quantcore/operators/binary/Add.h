// Add.h — Element-wise addition (逐元素加法)
// Phase: 一期必实现
//
// Mathematical form:  z_i = x_i + y_i
// Reference:  numpy.add
//
// Null propagation is handled by the BinaryOperator CRTP base class.
//
// Input modes (handled by BinaryOperator base class):
//   1. Column vs Column:  z_i = lhs[i] + rhs[i]
//   2. Column vs Scalar:  z_i = lhs[i] + scalar
//   3. Scalar vs Column:  z_i = scalar + rhs[i]
//
// Edge cases (IEEE 754):
//   ±Inf + ±Inf  -> ±Inf (same signs) or NaN (opposite signs)
//   NaN + any    -> NaN
//   Overflow      -> ±Inf

#pragma once

#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/operators/BinaryOperator.h"

namespace quantcore {

struct AddOp : public BinaryOperator<AddOp> {
    static constexpr BinaryOpCode kOpCode = BinaryOpCode::ADD;
    static constexpr const char* name = "add";

    static double evaluateScalar(double a, double b) noexcept {
        return a + b;
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
            output[i] = lhsData[i] + rhsData[i];
        }
    }
};

}  // namespace quantcore
