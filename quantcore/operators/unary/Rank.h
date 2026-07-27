// Rank.h — Rank operators (排名算子)
// Phase: 一期必实现
//
// Three related operators that rank elements within a column using
// pandas-compatible 'average' tie-breaking:
//
//   rank(X)             — raw rank, 1-based, ties averaged
//   rank_pct(X)         — percentile rank: rank / N, in (0, 1]
//   rank_normalized(X)  — normalized rank: (rank-1)/(N-1), in [0, 1]
//
// Reference:  pandas.rank(method='average')
//
// All three share the same core algorithm:
//   1. Collect (value, index) for non-null elements
//   2. Sort by value
//   3. Assign average ranks for ties
//   4. Apply normalization (identity, /N, or (rank-1)/(N-1))
//
// Unlike element-wise unary operators, rank operators require access to
// the full column and therefore shadow the base class evaluate() with a
// custom implementation.
//
// Null propagation: null and NaN inputs are excluded from ranking;
// their output positions are set to 0.0 (caller marks them null).
//
// SIMD notes:
//   - Rank is inherently not vectorizable due to the sorting step.
//     evaluateSimd delegates to evaluate() for interface consistency.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/operators/UnaryOperator.h"

namespace quantcore {
namespace detail {

// ============================================================
// Normalization mode for rank computation
// ============================================================

enum class RankNorm {
    RAW,         // avgRank                  — 1-based average rank
    PCT,         // avgRank / validCount     — percentile in (0, 1]
    NORMALIZED,  // (avgRank-1)/(validCount-1) — normalized in [0, 1]
};

// ============================================================
// Core ranking implementation shared by all rank operators
// ============================================================

inline void computeRank(const double* __restrict__ input,
                        double*       __restrict__ output,
                        std::size_t   n,
                        const uint64_t* __restrict__ nullMask,
                        RankNorm norm) noexcept {
    // ========================================================
    // Pass 1: collect non-null (value, index) pairs
    // ========================================================
    struct IndexedValue {
        double value;
        std::size_t index;
    };

    // Stack allocation for the common small-n case;
    // heap for large columns.
    constexpr std::size_t kStackMax = 4096;
    IndexedValue stackBuf[kStackMax];
    std::vector<IndexedValue> heapBuf;

    IndexedValue* pairs = stackBuf;
    std::size_t validCount = 0;

    if (n <= kStackMax) {
        for (std::size_t i = 0; i < n; ++i) {
            if (nullMask && ((nullMask[i / 64] >> (i % 64)) & uint64_t{1}))
                continue;
            if (std::isnan(input[i]))
                continue;
            pairs[validCount].value = input[i];
            pairs[validCount].index = i;
            ++validCount;
        }
    } else {
        heapBuf.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (nullMask && ((nullMask[i / 64] >> (i % 64)) & uint64_t{1}))
                continue;
            if (std::isnan(input[i]))
                continue;
            heapBuf.push_back({input[i], i});
        }
        pairs = heapBuf.data();
        validCount = heapBuf.size();
    }

    // ========================================================
    // Edge case: no valid elements
    // ========================================================
    if (validCount == 0) {
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = 0.0;
        }
        return;
    }

    // ========================================================
    // Pass 2: sort by value
    // ========================================================
    std::sort(pairs, pairs + validCount,
              [](const IndexedValue& a, const IndexedValue& b) noexcept {
                  return a.value < b.value;
              });

    // ========================================================
    // Pass 3: assign average ranks and normalize
    // ========================================================
    // Initialize all outputs to 0.0 (for null positions)
    for (std::size_t i = 0; i < n; ++i) {
        output[i] = 0.0;
    }

    std::size_t pos = 0;
    while (pos < validCount) {
        // Find the end of the current tie group
        std::size_t groupEnd = pos + 1;
        while (groupEnd < validCount &&
               pairs[groupEnd].value == pairs[pos].value) {
            ++groupEnd;
        }

        // Ranks are 1-based: pos+1 .. groupEnd
        // Average rank = (first + last) / 2
        double avgRank = (static_cast<double>(pos + 1) +
                          static_cast<double>(groupEnd)) / 2.0;

        // Apply normalization
        double result;
        switch (norm) {
            case RankNorm::RAW:
                result = avgRank;
                break;
            case RankNorm::PCT:
                result = avgRank / static_cast<double>(validCount);
                break;
            case RankNorm::NORMALIZED:
                if (validCount > 1) {
                    result = (avgRank - 1.0) /
                             static_cast<double>(validCount - 1);
                } else {
                    result = 0.0;  // single element: (1-1)/(1-1) → 0
                }
                break;
        }

        for (std::size_t k = pos; k < groupEnd; ++k) {
            output[pairs[k].index] = result;
        }

        pos = groupEnd;
    }
}

}  // namespace detail

// ============================================================
// RankOp — raw rank: avgRank (1-based, average for ties)
// ============================================================

struct RankOp : public UnaryOperator<RankOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::RANK;
    static constexpr const char* name = "rank";

    void evaluate(const double* __restrict__ input,
                  double*       __restrict__ output,
                  std::size_t   n,
                  const uint64_t* __restrict__ nullMask) const noexcept {
        detail::computeRank(input, output, n, nullMask, detail::RankNorm::RAW);
    }

    /// Operand dispatch (required by OperatorRegistry)
    void evaluate(const Operand& input,
                  double*       __restrict__ output,
                  std::size_t   n,
                  const uint64_t* __restrict__ nullMask) const noexcept {
        evaluate(input.column(), output, n, nullMask);
    }

    static double evaluateScalar(double /*x*/) noexcept {
        return std::numeric_limits<double>::quiet_NaN();
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        RankOp op;
        op.evaluate(Operand(input.data()), output, n, nullptr);
    }
};

// ============================================================
// RankPctOp — percentile rank: avgRank / N, in (0, 1]
// ============================================================

struct RankPctOp : public UnaryOperator<RankPctOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::RANK_PCT;
    static constexpr const char* name = "rank_pct";

    void evaluate(const double* __restrict__ input,
                  double*       __restrict__ output,
                  std::size_t   n,
                  const uint64_t* __restrict__ nullMask) const noexcept {
        detail::computeRank(input, output, n, nullMask, detail::RankNorm::PCT);
    }

    /// Operand dispatch (required by OperatorRegistry)
    void evaluate(const Operand& input,
                  double*       __restrict__ output,
                  std::size_t   n,
                  const uint64_t* __restrict__ nullMask) const noexcept {
        evaluate(input.column(), output, n, nullMask);
    }

    static double evaluateScalar(double /*x*/) noexcept {
        return std::numeric_limits<double>::quiet_NaN();
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        RankPctOp op;
        op.evaluate(input.data(), output, input.size(), nullptr);
    }
};

// ============================================================
// RankNormalizedOp — normalized rank: (avgRank-1)/(N-1), in [0, 1]
// ============================================================

struct RankNormalizedOp : public UnaryOperator<RankNormalizedOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::RANK_NORMALIZED;
    static constexpr const char* name = "rank_normalized";

    void evaluate(const double* __restrict__ input,
                  double*       __restrict__ output,
                  std::size_t   n,
                  const uint64_t* __restrict__ nullMask) const noexcept {
        detail::computeRank(input, output, n, nullMask, detail::RankNorm::NORMALIZED);
    }

    /// Operand dispatch (required by OperatorRegistry)
    void evaluate(const Operand& input,
                  double*       __restrict__ output,
                  std::size_t   n,
                  const uint64_t* __restrict__ nullMask) const noexcept {
        evaluate(input.column(), output, n, nullMask);
    }

    static double evaluateScalar(double /*x*/) noexcept {
        return std::numeric_limits<double>::quiet_NaN();
    }

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> input,
                             double*         output) noexcept {
        std::size_t n = input.size();
        RankNormalizedOp op;
        op.evaluate(input.data(), output, input.size(), nullptr);
    }
};

}  // namespace quantcore
