// UnaryOperator.h — CRTP base class for unary element-wise operators
// Phase: 一期必实现
//
// Unary operators transform a single input column element-by-element:
//   output[i] = f(input[i])
//
// This base class defines the mandatory interface that every unary
// operator struct (AbsOp, LogOp, NegOp, ...) must implement.  The CRTP
// pattern provides compile-time polymorphism with zero virtual-function
// overhead.
//
// Design:
//   - Stateless: operators hold no data; all methods are static
//   - Input:  ColView<double> referenced by the engine, OR a scalar double
//   - Output: Column<double> written into engine-provided buffer
//   - Null propagation: handled by the base class; derived ops only
//     implement the pure arithmetic via evaluateScalar()
//   - SIMD: each derived op provides per-SimdLevel kernels; base class
//     dispatches via the engine's selected level
//
// Each derived operator MUST provide:
//
//   1. static constexpr UnaryOpCode kOpCode = UnaryOpCode::XXX;
//      — operator identity (for registry, logging, dispatch)
//
//   2. static constexpr const char* name = "xxx";
//      — human-readable name (for logging, future formula parser)
//
//   3. static double evaluateScalar(double x) noexcept;
//      — scalar reference implementation (cross-validation baseline)
//        Must handle NaN, ±Inf correctly.  Must NOT check for null —
//        null propagation is done by the base class before calling.
//
//   4. template<SimdLevel L>
//      static void evaluateSimd(const double* __restrict__ input,
//                               double*       __restrict__ output,
//                               std::size_t   n) noexcept;
//      — SIMD kernel.  Phase 1 may delegate to evaluateScalar loop
//        with #pragma omp simd; hand-written intrinsics come later.
//
// Unified interface (recommended):
//
//   AbsOp op;
//   op.evaluate(closeData, outputData, n, nullMask);  // column input
//   op.evaluate(2.0,      outputData, n, nullMask);  // scalar broadcast
#pragma once

#include <cstddef>
#include <cstdint>

#include "quantcore/core/Types.h"

namespace quantcore {

template <typename Derived>
class UnaryOperator {
public:
    // ============================================================
    // Construction
    // ============================================================

    UnaryOperator() = default;

    // ============================================================
    // Batch evaluation — primary entry point called by engine
    // ============================================================

    // ============================================================
    // Unified evaluate — dispatches to optimal path automatically
    // ============================================================
    //
    // Accepts either a column pointer (const double*) or a scalar
    // (double) via implicit Operand construction:
    //
    //   op.evaluate(closeData, out, n, mask);  // column input
    //   op.evaluate(2.0,      out, n, mask);  // scalar broadcast
    //
    // The dispatch happens once per call — zero per-element overhead.

    void evaluate(const Operand& input,
                  double*       __restrict__ output,
                  std::size_t   n,
                  const uint64_t* __restrict__ nullMask) const noexcept
    {
        if (input.kind() == Operand::Kind::kColumn) {
            // Column input — apply f element-wise
            evaluateCol(input.column(), output, n, nullMask);
        } else {
            // Scalar input — compute f(scalar) once, broadcast to all elements
            const Derived& self = static_cast<const Derived&>(*this);
            double scalarResult = self.evaluateScalar(input.scalar());
            if (nullMask == nullptr) {
                for (std::size_t i = 0; i < n; ++i) {
                    output[i] = scalarResult;
                }
            } else {
                for (std::size_t i = 0; i < n; ++i) {
                    if ((nullMask[i / 64] >> (i % 64)) & uint64_t{1}) {
                        output[i] = 0.0;
                    } else {
                        output[i] = scalarResult;
                    }
                }
            }
        }
    }

    // ============================================================
    // Column evaluate (internal — direct pointer path)
    // ============================================================
    //
    /// Evaluate the operator over `n` elements with a column input.
    ///
    /// @param input     Input data pointer (length n, non-owning)
    /// @param output    Output buffer (length n, pre-allocated, 64-byte aligned)
    /// @param n         Number of elements
    /// @param nullMask  Input null bitmask (may be nullptr if no nulls)
    ///
    /// Null propagation: for each i where input is null, output[i] is
    /// also marked null and evaluateScalar is NOT called.
    void evaluateCol(const double* __restrict__ input,
                     double*       __restrict__ output,
                     std::size_t   n,
                     const uint64_t* __restrict__ nullMask) const noexcept
    {
        const Derived& self = static_cast<const Derived&>(*this);

        if (nullMask == nullptr) {
            // Fast path — no nulls, pure arithmetic
            for (std::size_t i = 0; i < n; ++i) {
                output[i] = self.evaluateScalar(input[i]);
            }
        } else {
            // Slow path — check null bitmask before each call
            for (std::size_t i = 0; i < n; ++i) {
                if ((nullMask[i / 64] >> (i % 64)) & uint64_t{1}) {
                    output[i] = 0.0;  // caller marks output null
                } else {
                    output[i] = self.evaluateScalar(input[i]);
                }
            }
        }
    }

    // ============================================================
    // SIMD-dispatched evaluation (future)
    // ============================================================
    //
    // template<SimdLevel L>
    // void evaluateSimd(const double* __restrict__ input,
    //                   double*       __restrict__ output,
    //                   std::size_t   n,
    //                   const uint64_t* __restrict__ nullMask) const noexcept
    // {
    //     const Derived& self = static_cast<const Derived&>(*this);
    //     // ... vectorized loop with tail handling ...
    // }

    // ============================================================
    // Op-code accessor (convenience)
    // ============================================================

    static constexpr UnaryOpCode opCode() noexcept {
        return Derived::kOpCode;
    }

    static constexpr const char* opName() noexcept {
        return Derived::name;
    }
};

}  // namespace quantcore
