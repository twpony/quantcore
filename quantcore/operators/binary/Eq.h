// Eq.h — Element-wise equality (逐元素等于比较)
// Phase: 一期必实现
//
// Mathematical form:  z_i = 1.0 if x_i == y_i else 0.0
// Reference:  numpy.equal (as float)
//
// Null propagation is handled by the BinaryOperator CRTP base class.
//
// Input modes (handled by BinaryOperator base class):
//   1. Column vs Column:  z_i = lhs[i] == rhs[i] ? 1.0 : 0.0
//   2. Column vs Scalar:  z_i = lhs[i] == scalar ? 1.0 : 0.0
//   3. Scalar vs Column:  z_i = scalar == rhs[i] ? 1.0 : 0.0
//
// Edge cases (IEEE 754):
//   NaN == any   -> false (0.0) — NaN comparisons always return false
//   +Inf == +Inf -> true  (1.0)
//   -Inf == -Inf -> true  (1.0)
//   +Inf == -Inf -> false (0.0)

#pragma once

#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/operators/BinaryOperator.h"

namespace quantcore {

struct EqOp : public BinaryOperator<EqOp> {
    static constexpr BinaryOpCode kOpCode = BinaryOpCode::EQ;
    static constexpr const char* name = "eq";

    /// Returns 1.0 if a == b, else 0.0.  NaN comparisons return 0.0 (false).
    static double evaluateScalar(double a, double b) noexcept {
        return (a == b) ? 1.0 : 0.0;
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
            output[i] = (lhsData[i] == rhsData[i]) ? 1.0 : 0.0;
        }
    }
};

}  // namespace quantcore
