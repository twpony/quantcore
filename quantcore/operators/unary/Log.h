// Log.h — Natural logarithm (自然对数)
// Phase: 一期必实现
//
// Mathematical form:  y_i = ln(x_i)
// Reference:  numpy.log
//
// Null propagation is handled by the UnaryOperator CRTP base class.
//
// Domain handling (matches numpy.log):
//   x > 0  -> ln(x)
//   x = 0  -> -Inf  (pole error, errno ERANGE)
//   x < 0  -> NaN   (domain error, errno EDOM)
//   +Inf   -> +Inf
//   NaN    -> NaN

#pragma once

#include <cmath>
#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/operators/UnaryOperator.h"

namespace quantcore {

struct LogOp : public UnaryOperator<LogOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::LOG;
    static constexpr const char* name = "log";

    /// Scalar reference — std::log matches numpy.log:
    ///   log(0) = -Inf, log(negative) = NaN, log(+Inf) = +Inf.
    static double evaluateScalar(double x) noexcept {
        return std::log(x);
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::log(input[i]);
        }
    }
};

}  // namespace quantcore
