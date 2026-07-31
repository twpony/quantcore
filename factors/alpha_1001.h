// alpha_1001.h — 20-day simple moving average
//
// Formula: SMA(close, 20) = mean(close[i-19..i])
//
// Implementation is in alpha_1001.cpp, compiled into the quantcore
// static library so callers get direct (non-PLT) linkage — same
// performance as a hand-rolled loop in the caller's translation unit.
//
// Usage (same pattern as alpha_0001):
//   registerAlpha1001(calc);
//   auto result = calc.evaluate("alpha_1001", md);
//
// Standalone (no FactorCalculator needed):
//   auto sma = computeSma20(md.column<double>(Field::CLOSE));
#pragma once

#include <string>

#include "quantcore/core/FactorCalculator.h"

namespace quantcore {
namespace factors {

// ============================================================
// Constants
// ============================================================

extern const std::string kAlpha1001;
extern const std::string kAlpha1001Desc;

// ============================================================
// Core computation
// ============================================================

/// Compute the 20-day simple moving average of close prices.
///
/// Running-sum sliding window: O(n) time, O(1) extra space.
/// Positions 0..18 are NaN (20-element window boundary).
Column<double> computeSma20(const Column<double>& close);

// ============================================================
// Registration
// ============================================================

void registerAlpha1001(FactorCalculator& calc);

// ============================================================
// Convenience evaluation
// ============================================================

Column<double> evaluateAlpha1001(FactorCalculator& calc,
                                  const MarketData& md);

}  // namespace factors
}  // namespace quantcore
