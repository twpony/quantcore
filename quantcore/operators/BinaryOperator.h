// BinaryOperator.h — CRTP base class for binary element-wise operators
// Phase: 一期必实现
//
// Binary operators combine two input columns element-by-element:
//   output[i] = f(lhs[i], rhs[i])
//
// This also covers the column-vs-scalar case where rhs[i] is a
// constant — the engine handles broadcasting before calling evaluate().
//
// This base class defines the mandatory interface that every binary
// operator struct (AddOp, MulOp, GtOp, ...) must implement.
// The CRTP pattern provides compile-time polymorphism with zero
// virtual-function overhead.
//
// Design:
//   - Stateless: operators hold no data; all methods are static
//   - Input:  two ColView<double> (or one ColView + one broadcast scalar)
//   - Output: Column<double> written into engine-provided buffer
//   - Null propagation: if EITHER lhs[i] or rhs[i] is null, output[i]
//     is marked null and evaluateScalar is NOT called
//   - NaN policy (IEEE 754 compliant, differentiated by operation category):
//     | Category              | Operators            | NaN Behavior          |
//     | Arithmetic            | Add,Sub,Mul,Div,     | NaN propagates        |
//     |                       | Max,Min              | (input NaN → out NaN) |
//     | Ordered comparison    | Eq,Gt,Lt             | NaN → 0.0 (false)    |
//     | Unordered comparison  | Neq                  | NaN → 1.0 (true)     |
//     Rationale: arithmetic operations produce values (NaN input → NaN output
//     is the safe default); comparison predicates answer true/false questions
//     (IEEE 754 unordered comparisons return false, except != which is true
//     since NaN never equals anything).  Do NOT "unify" across categories —
//     each behavior is mathematically correct in its context.
//   - SIMD: each derived op provides per-SimdLevel kernels
//
// Each derived operator MUST provide:
//
//   1. static constexpr BinaryOpCode kOpCode = BinaryOpCode::XXX;
//      — operator identity
//
//   2. static constexpr const char* name = "xxx";
//      — human-readable name
//
//   3. static double evaluateScalar(double a, double b) noexcept;
//      — scalar reference implementation (cross-validation baseline)
//        Must handle NaN, ±Inf, div-by-zero correctly.
//        Must NOT check for null — null propagation is done by the
//        base class before calling.
//
//   4. template<SimdLevel L>
//      static void evaluateSimd(ColView<double> lhs,
//                               ColView<double> rhs,
//                               double*         output) noexcept;
//      — SIMD kernel (future).  n is derived from lhs.size().
//
// Unified interface (recommended):
//
//   MaxOp op;
//   op.evaluate(closeData, 1.0, outputData, n, nullMask);       // col vs scalar
//   op.evaluate(closeData, highData, outputData, n, nullMask);  // col vs col
//   op.evaluate(2.0, lowData, outputData, n, nullMask);         // scalar vs col
//   op.evaluate(3.0, 5.0, outputData, n, nullMask);             // scalar vs scalar
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"

namespace quantcore {

template <typename Derived>
class BinaryOperator {
public:
    // ============================================================
    // Construction
    // ============================================================

    BinaryOperator() = default;

    // ============================================================
    // Unified evaluate — dispatches to optimal path automatically
    // ============================================================
    //
    // Accepts any combination of scalar (double) and column
    // (const double*) via implicit Operand construction:
    //
    //   op.evaluate(colA, colB, out, n, mask);    // col vs col
    //   op.evaluate(colA, 2.0, out, n, mask);     // col vs scalar
    //   op.evaluate(3.0, colB, out, n, mask);     // scalar vs col
    //   op.evaluate(3.0, 5.0, out, n, mask);      // scalar vs scalar
    //
    // The dispatch to the optimal internal path happens once
    // per call — there is zero per-element overhead.

    void evaluate(const Operand& lhs,
                  const Operand& rhs,
                  double*       __restrict__ output,
                  std::size_t   n,
                  const uint64_t* __restrict__ nullMask) const noexcept
    {
        const Derived& self = static_cast<const Derived&>(*this);

        if (lhs.kind() == Operand::Kind::kColumn &&
            rhs.kind() == Operand::Kind::kColumn) {
            // Column vs Column
            evaluateColCol(lhs.column(), rhs.column(), output, n, nullMask);
        } else if (lhs.kind() == Operand::Kind::kColumn &&
                   rhs.kind() == Operand::Kind::kScalar) {
            // Column vs Scalar
            evaluateColScalar(lhs.column(), rhs.scalar(), output, n, nullMask);
        } else if (lhs.kind() == Operand::Kind::kScalar &&
                   rhs.kind() == Operand::Kind::kColumn) {
            // Scalar vs Column
            evaluateScalarCol(lhs.scalar(), rhs.column(), output, n, nullMask);
        } else {
            // Scalar vs Scalar — broadcast to all elements
            double scalarResult = self.evaluateScalar(lhs.scalar(), rhs.scalar());
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
    // Typed evaluate variants (internal / backward-compatible)
    // ============================================================

    /// Column vs Column
    void evaluateColCol(const double* __restrict__ lhs,
                         const double* __restrict__ rhs,
                         double*       __restrict__ output,
                         std::size_t   n,
                         const uint64_t* __restrict__ nullMask) const noexcept
    {
        const Derived& self = static_cast<const Derived&>(*this);

        if (nullMask == nullptr) {
            for (std::size_t i = 0; i < n; ++i) {
                output[i] = self.evaluateScalar(lhs[i], rhs[i]);
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                if ((nullMask[i / 64] >> (i % 64)) & uint64_t{1}) {
                    output[i] = 0.0;
                } else {
                    output[i] = self.evaluateScalar(lhs[i], rhs[i]);
                }
            }
        }
    }

    /// Column vs Scalar
    void evaluateColScalar(const double* __restrict__ lhs,
                           double        scalar,
                           double*       __restrict__ output,
                           std::size_t   n,
                           const uint64_t* __restrict__ nullMask) const noexcept
    {
        const Derived& self = static_cast<const Derived&>(*this);

        if (nullMask == nullptr) {
            for (std::size_t i = 0; i < n; ++i) {
                output[i] = self.evaluateScalar(lhs[i], scalar);
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                if ((nullMask[i / 64] >> (i % 64)) & uint64_t{1}) {
                    output[i] = 0.0;
                } else {
                    output[i] = self.evaluateScalar(lhs[i], scalar);
                }
            }
        }
    }

    /// Scalar vs Column
    void evaluateScalarCol(double        scalar,
                           const double* __restrict__ rhs,
                           double*       __restrict__ output,
                           std::size_t   n,
                           const uint64_t* __restrict__ nullMask) const noexcept
    {
        const Derived& self = static_cast<const Derived&>(*this);

        if (nullMask == nullptr) {
            for (std::size_t i = 0; i < n; ++i) {
                output[i] = self.evaluateScalar(scalar, rhs[i]);
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                if ((nullMask[i / 64] >> (i % 64)) & uint64_t{1}) {
                    output[i] = 0.0;
                } else {
                    output[i] = self.evaluateScalar(scalar, rhs[i]);
                }
            }
        }
    }

    // ============================================================
    // Op-code accessor (convenience)
    // ============================================================

    static constexpr BinaryOpCode opCode() noexcept {
        return Derived::kOpCode;
    }

    static constexpr const char* opName() noexcept {
        return Derived::name;
    }
};

}  // namespace quantcore
