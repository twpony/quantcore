// Sign.h — Sign function (符号函数)
// Phase: 一期必实现
//
// Mathematical form:  y_i = sign(x_i)
// Reference:  numpy.sign
//
// Null propagation is handled by the UnaryOperator CRTP base class.
//
// Sign convention (matches numpy.sign):
//   x > 0  ->  1.0
//   x = 0  ->  0.0
//   x < 0  -> -1.0
//   NaN    ->  NaN
//   +Inf   ->  1.0
//   -Inf   -> -1.0

#pragma once

#include <cmath>
#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/operators/UnaryOperator.h"

namespace quantcore {

struct SignOp : public UnaryOperator<SignOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::SIGN;
    static constexpr const char* name = "sign";

    /// Scalar reference — matches numpy.sign.
    /// Returns 1.0 for positive, -1.0 for negative, 0.0 for zero.
    /// NaN propagates as NaN.
    static double evaluateScalar(double x) noexcept {
        if (std::isnan(x)) return x;          // NaN -> NaN
        if (x > 0.0) return 1.0;
        if (x < 0.0) return -1.0;
        return 0.0;                            // 0.0 or -0.0 -> 0.0
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            double x = input[i];
            if (std::isnan(x))
                output[i] = x;
            else if (x > 0.0)
                output[i] = 1.0;
            else if (x < 0.0)
                output[i] = -1.0;
            else
                output[i] = 0.0;
        }
    }
};

}  // namespace quantcore
