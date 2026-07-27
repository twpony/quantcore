// RollingOperator.h — CRTP base class for rolling-window operators
// Phase: 一期实现
//
// Rolling operators scan a fixed-size window over a time series and
// compute a statistic (mean, max, std, etc.) at each position.
//
// Design:
//   - CRTP static polymorphism (no virtual functions)
//   - Window size is a runtime constructor parameter (not a template
//     argument) so users can vary it without recompilation.
//   - Input:  ColView<double> (zero-copy view, carries data + size + nullMask)
//   - Output: double pointer to engine-provided buffer
//   - Each derived op MUST provide evaluateScalar() as the reference
//     implementation for SIMD cross-validation.
//   - Null propagation: if input.isNull(i), output[i] is set to 0.0
//     (caller marks output null).
//
// Concrete ops: all 18 rolling/ operators
//
// Boundary convention:
//   - Window operators: first (window-1) elements are invalid.
//     evaluateScalar returns NaN for these positions.
//   - Lag operators (RollingDiff, RollingShift): first `window` elements
//     are invalid. evaluateScalar returns NaN for these positions.
//   - Each operator's evaluateScalar is responsible for returning NaN
//     when the position does not have enough history.
//
// SIMD notes:
//   - RollingSma / RollingSum: O(n) with prefix-sum difference
//   - RollingMax / RollingMin: O(n) with monotonic deque
//   - RollingStd: Welford's online algorithm or running sum-of-squares
//   - RollingEma: single-pass iterative computation
//   - RollingDiff / RollingShift: trivial O(1) per element
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"

namespace quantcore {

template <typename Derived>
class RollingOperator {
public:
    // ============================================================
    // Construction
    // ============================================================

    /// @param window  Number of consecutive observations in the window.
    ///                For lag operators (RollingDiff, RollingShift), this is the lag.
    ///                Must be >= 1.
    explicit RollingOperator(std::size_t window) noexcept
        : window_(window)
    {}

    // ============================================================
    // Window access
    // ============================================================

    std::size_t window() const noexcept { return window_; }

    // ============================================================
    // Batch evaluation — primary entry point
    // ============================================================
    //
    /// Evaluate the rolling operator over all elements in the input view.
    ///
    /// @param input   ColView<double> — carries data, size, and null mask
    /// @param output  Output buffer (length = input.size(), pre-allocated)
    ///
    /// For each position i, calls Derived::evaluateScalar(input.data(), i, window_).
    /// If input.isNull(i), output[i] is set to 0.0.

    void evaluate(ColView<double> input,
                  double*         output) const noexcept
    {
        const Derived& self = static_cast<const Derived&>(*this);
        const double* __restrict__ data = input.data();
        std::size_t n = input.size();

        if (!input.hasNullMask()) {
            for (std::size_t i = 0; i < n; ++i) {
                output[i] = self.evaluateScalar(data, i, window_);
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                if (input.isNull(i)) {
                    output[i] = 0.0;
                } else {
                    output[i] = self.evaluateScalar(data, i, window_);
                }
            }
        }
    }

    // ============================================================
    // SIMD-dispatched batch evaluation (static, like evaluateScalar)
    // ============================================================
    //
    // @param input   ColView<double> — carries data + size
    // @param output  Output buffer (length = input.size())
    // @param window  Window size — passed explicitly (consistent with evaluateScalar)
    //
    // Input is clean (no nulls); the caller handles null masking separately.
    // Default falls back to calling evaluateScalar per element.

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output,
                             std::size_t     window) noexcept
    {
        const double* __restrict__ data = input.data();
        std::size_t n = input.size();
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = Derived::evaluateScalar(data, i, window);
        }
    }

    // ============================================================
    // Op-code accessor (convenience)
    // ============================================================

    static constexpr RollingOpCode opCode() noexcept {
        return Derived::kOpCode;
    }

    static constexpr const char* opName() noexcept {
        return Derived::name;
    }

    // ============================================================
    // Interface that each derived operator MUST provide
    // ============================================================
    //
    // 1. Static op-code constant:
    //    static constexpr RollingOpCode kOpCode = RollingOpCode::XXX;
    //
    // 2. Scalar reference implementation (cross-validation baseline):
    //    static double evaluateScalar(const double* input,
    //                                 std::size_t i,
    //                                 std::size_t window) noexcept;
    //
    // 3. SIMD kernel (optional override — static, like evaluateScalar):
    //    template <SimdLevel L>
    //    static void evaluateSimd(ColView<double> input,
    //                             double*         output,
    //                             std::size_t     window) noexcept;
    //
    // 4. Human-readable name:
    //    static constexpr const char* name = "xxx";

protected:
    std::size_t window_;
};

}  // namespace quantcore
