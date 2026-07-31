// alpha_1001.cpp — 20-day simple moving average
//
// Algorithm:  running-sum sliding window, O(n) time, O(1) extra space.
//
//   sum = close[0] + ... + close[19]
//   result[19] = sum / 20
//   for i = 20 .. n-1:
//       sum += close[i] - close[i-20]
//       result[i] = sum / 20
//
// This file is compiled into the quantcore STATIC library (not the
// quantcorefactors shared library).  Callers link directly against the
// .o — no PLT/GOT indirection, same performance as a hand-rolled loop
// in the caller's translation unit.

#include "factors/alpha_1001.h"

#include <cmath>
#include <limits>

#include "quantcore/core/Types.h"

namespace quantcore {
namespace factors {

// ============================================================
// Constants
// ============================================================

const std::string kAlpha1001 = "alpha_1001";

const std::string kAlpha1001Desc =
    "SMA(close, 20) — 20-day simple moving average (static lib)";

// ============================================================
// Core computation
// ============================================================

Column<double> computeSma20(const Column<double>& close) {
    static constexpr std::size_t kWindow = 20;

    const std::size_t n = close.size();
    Column<double> result(n);

    if (n < kWindow) {
        for (std::size_t i = 0; i < n; ++i)
            result[i] = std::numeric_limits<double>::quiet_NaN();
        return result;
    }

    for (std::size_t i = 0; i < kWindow - 1; ++i)
        result[i] = std::numeric_limits<double>::quiet_NaN();

    double runningSum = 0.0;
    for (std::size_t i = 0; i < kWindow; ++i)
        runningSum += close[i];

    const double invWindow = 1.0 / static_cast<double>(kWindow);
    result[kWindow - 1] = runningSum * invWindow;

    for (std::size_t i = kWindow; i < n; ++i) {
        runningSum += close[i];
        runningSum -= close[i - kWindow];
        result[i] = runningSum * invWindow;
    }

    return result;
}

// ============================================================
// Registration
// ============================================================

void registerAlpha1001(FactorCalculator& calc) {
    calc.registerCustomFactor(
        kAlpha1001,
        [](const MarketData& md) -> Column<double> {
            const auto& close = md.column<double>(Field::CLOSE);
            return computeSma20(close);
        },
        kAlpha1001Desc);
}

// ============================================================
// Convenience evaluation
// ============================================================

Column<double> evaluateAlpha1001(FactorCalculator& calc,
                                  const MarketData& md) {
    return calc.evaluate(kAlpha1001, md);
}

}  // namespace factors
}  // namespace quantcore
