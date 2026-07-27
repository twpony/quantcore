// Log10.h — Base-10 logarithm (常用对数)
// Phase: 一期必实现
//
// Mathematical form:  y_i = log_10(x_i)
// Reference:  numpy.log10
//
// Null propagation is handled by the UnaryOperator CRTP base class.
//
// Domain handling (matches numpy.log10):
//   x > 0  -> log_10(x)
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

struct Log10Op : public UnaryOperator<Log10Op> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::LOG10;
    static constexpr const char* name = "log10";

    /// Scalar reference — std::log10 matches numpy.log10
    static double evaluateScalar(double x) noexcept {
        return std::log10(x);
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::log10(input[i]);
        }
    }
};

}  // namespace quantcore
