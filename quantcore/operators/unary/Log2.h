// Log2.h — Base-2 logarithm (以2为底对数)
// Phase: 一期必实现
//
// Mathematical form:  y_i = log_2(x_i)
// Reference:  numpy.log2
//
// Null propagation is handled by the UnaryOperator CRTP base class.
//
// Domain handling (matches numpy.log2):
//   x > 0  -> log_2(x)
//   x = 0  -> -Inf  (pole error)
//   x < 0  -> NaN   (domain error)
//   +Inf   -> +Inf
//   NaN    -> NaN

#pragma once

#include <cmath>
#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/operators/UnaryOperator.h"

namespace quantcore {

struct Log2Op : public UnaryOperator<Log2Op> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::LOG2;
    static constexpr const char* name = "log2";

    /// Scalar reference — std::log2 matches numpy.log2
    static double evaluateScalar(double x) noexcept {
        return std::log2(x);
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::log2(input[i]);
        }
    }
};

}  // namespace quantcore
