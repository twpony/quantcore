// Neg.h — Negation (取反)
// Phase: 一期必实现
//
// Mathematical form:  y_i = -x_i
// Reference:  numpy.negative

#pragma once

#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/operators/UnaryOperator.h"

namespace quantcore {

struct NegOp : public UnaryOperator<NegOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::NEG;
    static constexpr const char* name = "neg";

    static double evaluateScalar(double x) noexcept {
        return -x;
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = -input[i];
        }
    }
};

}  // namespace quantcore
