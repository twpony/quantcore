// Sqrt.h — Square root (平方根)
// Phase: 一期必实现
//
// Mathematical form:  y_i = sqrt(x_i)
// Reference:  numpy.sqrt
//
// Null propagation is handled by the UnaryOperator CRTP base class.
//
// Domain handling (matches numpy.sqrt):
//   x > 0  -> sqrt(x)
//   x = 0  -> 0.0
//   x < 0  -> NaN   (domain error)
//   -0.0   -> -0.0  (preserves sign of zero)
//   +Inf   -> +Inf
//   NaN    -> NaN

#pragma once

#include <cmath>
#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/operators/UnaryOperator.h"

namespace quantcore {

struct SqrtOp : public UnaryOperator<SqrtOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::SQRT;
    static constexpr const char* name = "sqrt";

    /// Scalar reference — std::sqrt matches numpy.sqrt:
    ///   sqrt(negative) = NaN, sqrt(0) = 0, sqrt(+Inf) = +Inf.
    static double evaluateScalar(double x) noexcept {
        return std::sqrt(x);
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::sqrt(input[i]);
        }
    }
};

}  // namespace quantcore
