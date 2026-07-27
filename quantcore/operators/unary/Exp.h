// Exp.h — Exponential (指数函数)
// Phase: 一期必实现
//
// Mathematical form:  y_i = e^{x_i}
// Reference:  numpy.exp
//
// Null propagation is handled by the UnaryOperator CRTP base class.
//
// Edge cases (delegated to std::exp):
//   finite x -> e^x
//   -Inf    -> 0.0
//   +Inf    -> +Inf
//   NaN     -> NaN

#pragma once

#include <cmath>
#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/operators/UnaryOperator.h"

namespace quantcore {

struct ExpOp : public UnaryOperator<ExpOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::EXP;
    static constexpr const char* name = "exp";

    /// Scalar reference — std::exp matches numpy.exp across all values.
    static double evaluateScalar(double x) noexcept {
        return std::exp(x);
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::exp(input[i]);
        }
    }
};

}  // namespace quantcore
