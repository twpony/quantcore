// Inv.h — Reciprocal (倒数)
// Phase: 一期必实现
//
// Mathematical form:  y_i = 1 / x_i
// Reference:  numpy.reciprocal
//
// Null propagation is handled by the UnaryOperator CRTP base class.
//
// Edge cases:
//   finite x != 0 -> 1/x
//   x = 0         -> ±Inf (division by zero, sign preserved)
//   ±Inf          -> ±0.0
//   NaN           -> NaN

#pragma once

#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/operators/UnaryOperator.h"

namespace quantcore {

struct InvOp : public UnaryOperator<InvOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::INV;
    static constexpr const char* name = "inv";

    /// Scalar reference — 1.0 / x matches numpy.reciprocal.
    /// 1/0 = Inf, 1/Inf = 0, 1/NaN = NaN.
    static double evaluateScalar(double x) noexcept {
        return 1.0 / x;
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = 1.0 / input[i];
        }
    }
};

}  // namespace quantcore
