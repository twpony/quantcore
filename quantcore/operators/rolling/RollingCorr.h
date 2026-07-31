// RollingCorr.h — Rolling Pearson correlation (滚动相关系数)
// Phase: 远期扩展
//
// Mathematical form:  rolling_corr(X, Y, n)[i] = corr(window_X, window_Y)
// Reference:  pandas.Series.rolling(n).corr(other)  — verified match
//
// Computes the Pearson correlation coefficient of the most recent `n`
// observations of two series X and Y.  For i < n-1, evaluateScalar returns NaN.
//
// Null propagation: if any input in the window is null, the engine
// marks output[i] as null.  evaluateScalar assumes clean input.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/storage/ColView.h"

namespace quantcore {

struct RollingCorrOp {
    static constexpr RollingOpCode kOpCode = RollingOpCode::ROLLING_CORR;
    static constexpr const char* name = "rolling_corr";

    explicit RollingCorrOp(std::size_t window) noexcept
        : window_(window) {}

    std::size_t window() const noexcept { return window_; }

    // ============================================================
    // Batch evaluation
    // ============================================================

    void evaluate(ColView<double> x,
                  ColView<double> y,
                  double*         output) const noexcept
    {
        const double* __restrict__ dataX = x.data();
        const double* __restrict__ dataY = y.data();
        std::size_t n = x.size();

        for (std::size_t i = 0; i < n; ++i) {
            output[i] = evaluateScalar(dataX, dataY, i, window_);
        }
    }

    // ============================================================
    // Scalar reference implementation — two-pass algorithm
    // ============================================================

    static double evaluateScalar(const double* x,
                                 const double* y,
                                 std::size_t i,
                                 std::size_t window) noexcept
    {
        if (i + 1 < window) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::size_t start = i + 1 - window;
        std::size_t count = window;

        // Pass 1: means
        double sumX = 0.0, sumY = 0.0;
        for (std::size_t j = start; j <= i; ++j) {
            sumX += x[j];
            sumY += y[j];
        }
        double meanX = sumX / static_cast<double>(count);
        double meanY = sumY / static_cast<double>(count);

        // Pass 2: covariance and variances
        double cov = 0.0, varX = 0.0, varY = 0.0;
        for (std::size_t j = start; j <= i; ++j) {
            double dx = x[j] - meanX;
            double dy = y[j] - meanY;
            cov += dx * dy;
            varX += dx * dx;
            varY += dy * dy;
        }
        cov /= static_cast<double>(count);
        varX /= static_cast<double>(count);
        varY /= static_cast<double>(count);

        double denom = std::sqrt(varX * varY);
        if (denom == 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return cov / denom;
    }

    // ============================================================
    // SIMD kernel — O(n) with running sums
    // ============================================================

    template <SimdLevel L>
    static void evaluateSimd(ColView<double> x,
                             ColView<double> y,
                             double*         output,
                             std::size_t     window) noexcept
    {
        std::size_t n = x.size();
        if (n == 0) return;

        double runningSumX = 0.0;
        double runningSumY = 0.0;
        double runningSumXY = 0.0;
        double runningSumXSq = 0.0;
        double runningSumYSq = 0.0;
        double invWindow = 1.0 / static_cast<double>(window);

        for (std::size_t i = 0; i < n; ++i) {
            double xi = x[i];
            double yi = y[i];
            runningSumX += xi;
            runningSumY += yi;
            runningSumXY += xi * yi;
            runningSumXSq += xi * xi;
            runningSumYSq += yi * yi;

            if (i + 1 < window) {
                output[i] = std::numeric_limits<double>::quiet_NaN();
            } else {
                if (i >= window) {
                    double oldX = x[i - window];
                    double oldY = y[i - window];
                    runningSumX -= oldX;
                    runningSumY -= oldY;
                    runningSumXY -= oldX * oldY;
                    runningSumXSq -= oldX * oldX;
                    runningSumYSq -= oldY * oldY;
                }
                double meanX = runningSumX * invWindow;
                double meanY = runningSumY * invWindow;
                double cov = runningSumXY * invWindow - meanX * meanY;
                double varX = runningSumXSq * invWindow - meanX * meanX;
                double varY = runningSumYSq * invWindow - meanY * meanY;

                // Clamp negative variances from floating-point error
                if (varX < 0.0) varX = 0.0;
                if (varY < 0.0) varY = 0.0;

                double denom = std::sqrt(varX * varY);
                if (denom == 0.0) {
                    output[i] = std::numeric_limits<double>::quiet_NaN();
                } else {
                    output[i] = cov / denom;
                }
            }
        }
    }

private:
    std::size_t window_;
};

}  // namespace quantcore
