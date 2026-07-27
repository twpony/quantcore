// Not.h — Logical NOT (逻辑非)
// Phase: 一期必实现
//
// Mathematical form:  y_i = NOT(x_i)
// Reference:  numpy.logical_not
//
// Null propagation is handled by the UnaryOperator CRTP base class.
//
// Logic (matches numpy.logical_not):
//   x != 0  -> 0.0  (truthy -> false)
//   x == 0  -> 1.0  (falsy  -> true)
//   NaN     -> 0.0  (NaN is truthy, NOT(NaN) = 0.0)
//   ±Inf    -> 0.0  (Inf is truthy, NOT(Inf) = 0.0)

#pragma once

#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/operators/UnaryOperator.h"

namespace quantcore {

struct NotOp : public UnaryOperator<NotOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::NOT;
    static constexpr const char* name = "not";

    /// Scalar reference — non-zero -> 0.0, zero -> 1.0.
    /// Matches numpy.logical_not: truthy values (including NaN, Inf)
    /// evaluate to 0.0; only exact zero evaluates to 1.0.
    static double evaluateScalar(double x) noexcept {
        return (x == 0.0) ? 1.0 : 0.0;
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = (input[i] == 0.0) ? 1.0 : 0.0;
        }
    }
};

}  // namespace quantcore
