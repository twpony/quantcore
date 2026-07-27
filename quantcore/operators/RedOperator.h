// RedOperator.h — CRTP base class for cross-section operators
// Phase: 一期实现
//
// CS operators work on multi-asset values at a single time point.
// They unify what were previously separate "cross_section" and "reduction"
// operator families:
//
//   Reduction ops  (SUM, MEAN, STD, VAR, MIN, MAX, PROD):
//     N values → 1 scalar. Call reduce() for the scalar, or evaluate()
//     to broadcast the same value to every output position.
//
//   Transform ops  (ZSCORE, QUANTILE):
//     N values → N values. Call evaluate() to compute per-element results.
//
// Design:
//   - CRTP static polymorphism
//   - Input:  ColView<double>  (N asset values at one time point)
//   - Output: double* output (length = input.size()) for evaluate()
//             double scalar for reduce()
//   - Null handling: NaN values are SKIPPED.  No separate nullMask needed —
//     the caller places NaN at positions to exclude (e.g. halted stocks).
//     If all values are NaN or N == 0, returns NaN.
//   - Each derived op MUST provide evaluateScalar() as the reference
//     implementation.
//
// evaluateScalar signature:
//   static double evaluateScalar(ColView<double> values,
//                                std::size_t i) noexcept;
//   - For reduction ops: ignores `i`, returns the aggregate scalar.
//     Use values.data() and values.size() to iterate.
//   - For transform ops: uses `i` to compute per-element result
//   - NaN in values[] is treated as "skip this asset"
//
// Concrete ops: RedSum, RedMean, RedStd, RedVar, RedMin, RedMax, RedMul,
//                RedMedian, RedZScore, RedQuantile

#pragma once

#include <cstddef>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"

namespace quantcore {

template <typename Derived>
class RedOperator {
public:
    RedOperator() = default;

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
    // Reduce — compute scalar aggregate (convenience)
    // ============================================================

    double reduce(ColView<double> values) const noexcept
    {
        return static_cast<const Derived&>(*this).evaluateScalar(values, 0);
    }

    // ============================================================
    // Op-code / name accessors
    // ============================================================

    static constexpr RedOpCode opCode() noexcept { return Derived::kOpCode; }
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
    // 1. static constexpr RedOpCode kOpCode = RedOpCode::RED_XXX;
    // 2. static constexpr const char* name = "red_xxx";
    // 3. static double evaluateScalar(ColView<double> values,
    //                                 std::size_t i) noexcept;
    //    — Use values.data() + values.size() to iterate.
    //      Skip NaN values; i is ignored by reduction ops.
    // 4. (optional) template<SimdLevel L> void evaluateSimd(...) noexcept;
};

}  // namespace quantcore
