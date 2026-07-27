// CsOperator.h — CRTP base class for cross-section transform operators
// Phase: 一期实现
//
// CS (Cross-Section) operators work on multi-asset values at a single time
// point and produce per-element transformed output.  Unlike RedOperator
// (which includes reduction ops), CsOperator is exclusively for transform ops:
// every position receives a potentially different result.
//
//   N values → N values
//
// Design:
//   - CRTP static polymorphism
//   - Input:  ColView<double>  (N asset values at one time point)
//   - Output: double* output (length = input.size())
//   - Null handling: NaN values are SKIPPED when computing cross-sectional
//     statistics.  Null input positions map to NaN output.
//     If all values are NaN or count < required minimum, returns NaN.
//   - Each derived op MUST provide evaluateScalar() as the reference
//     implementation.
//   - Parameterized operators (CsWinsorize, CsClip) pass parameters
//     as per-call arguments to evaluate() instead of constructor params;
//     evaluateScalar is non-static for those.
//
// evaluateScalar signature (stateless ops):
//   static double evaluateScalar(ColView<double> values,
//                                std::size_t i) noexcept;
//   - Uses values.data() + values.size() to iterate.
//   - Uses `i` to compute per-element result.
//   - NaN in values[] is treated as "skip this asset".
//
// evaluateScalar signature (parameterized ops):
//   double evaluateScalar(ColView<double> values,
//                         std::size_t i,
//                         double param1, double param2) const noexcept;
//   - Extra parameters are passed per-call (e.g. lowerPct, upperPct).
//
// Concrete ops: CsRank, CsQuantile, CsZScore, CsNormalize, CsNormalizeL1,
//                CsNormalizeL2, CsWinsorize, CsWinsorizeMAD, CsClip, CsDemean

#pragma once

#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"

namespace quantcore {

template <typename Derived>
class CsOperator {
public:
    CsOperator() = default;

    // ============================================================
    // Batch evaluate — fill output with per-element results
    // ============================================================

    void evaluate(ColView<double> values,
                  double*         output) const noexcept
    {
        const Derived& self = static_cast<const Derived&>(*this);
        std::size_t n = values.size();
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = self.evaluateScalar(values, i);
        }
    }

    // ============================================================
    // Op-code / name accessors
    // ============================================================

    static constexpr CsOpCode opCode() noexcept { return Derived::kOpCode; }
    static constexpr const char* opName() noexcept { return Derived::name; }

    // ============================================================
    // SIMD-dispatched batch evaluation
    // ============================================================

    template <SimdLevel L>
    void evaluateSimd(ColView<double> values,
                      double*         output) const noexcept
    {
        evaluate(values, output);  // default: fall back to scalar
    }

    // ============================================================
    // Interface that each derived operator MUST provide
    // ============================================================
    //
    // 1. static constexpr CsOpCode kOpCode = CsOpCode::CS_XXX;
    // 2. static constexpr const char* name = "cs_xxx";
    // 3. evaluateScalar(ColView<double> values, std::size_t i) noexcept;
    //    — Use values.data() + values.size() to iterate.
    //      Skip NaN values; i is the output position.
    //    — Static for stateless ops, const non-static for parameterized ops.
    // 4. (optional) template<SimdLevel L> void evaluateSimd(...) noexcept;
};

}  // namespace quantcore
