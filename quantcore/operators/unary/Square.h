// Square.h — Element-wise square (逐元素平方)
// Phase: 一期必实现
//
// Mathematical form:  y_i = x_i^2
// Reference:  numpy.square
//
// Null propagation is handled by the UnaryOperator CRTP base class.
//
// Edge cases:
//   finite x -> x^2
//   ±Inf    -> +Inf
//   NaN     -> NaN
//   overflow -> +Inf (IEEE 754)

#pragma once

#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/operators/UnaryOperator.h"

namespace quantcore {

struct SquareOp : public UnaryOperator<SquareOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::SQUARE;
    static constexpr const char* name = "square";

    /// Scalar reference — x * x matches numpy.square.
    /// Uses plain multiplication instead of std::pow(x, 2) for performance.
    static double evaluateScalar(double x) noexcept {
        return x * x;
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = input[i] * input[i];
        }
    }
};

}  // namespace quantcore
