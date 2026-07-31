// test_rolling_cov.cpp — unit tests for RollingCovOp (滚动协方差, ddof=0)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/rolling/RollingCov.h"

using namespace quantcore;

class RollingCovOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        x_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        y_ = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0};
    }

    // Helper: population covariance (ddof=0) of values in [start, end]
    static double popCov(const double* x, const double* y,
                         std::size_t start, std::size_t end) {
        std::size_t count = end - start + 1;
        double sumX = 0.0, sumY = 0.0;
        for (std::size_t j = start; j <= end; ++j) {
            sumX += x[j];
            sumY += y[j];
        }
        double meanX = sumX / static_cast<double>(count);
        double meanY = sumY / static_cast<double>(count);

        double cov = 0.0;
        for (std::size_t j = start; j <= end; ++j) {
            double dx = x[j] - meanX;
            double dy = y[j] - meanY;
            cov += dx * dy;
        }
        return cov / static_cast<double>(count);
    }

    std::vector<double> x_;
    std::vector<double> y_;
};

TEST_F(RollingCovOpTest, OpCode) {
    RollingCovOp op(3);
    EXPECT_EQ(op.kOpCode, RollingOpCode::ROLLING_COV);
}

TEST_F(RollingCovOpTest, Name) {
    RollingCovOp op(3);
    EXPECT_STREQ(op.name, "rolling_cov");
}

TEST_F(RollingCovOpTest, WindowAccess) {
    RollingCovOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingCovOpTest, ScalarWindow3) {
    RollingCovOp op(3);
    // First 2 positions: insufficient data
    EXPECT_TRUE(std::isnan(op.evaluateScalar(x_.data(), y_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(x_.data(), y_.data(), 1, 3)));

    double expected = popCov(x_.data(), y_.data(), 0, 2);
    double actual = op.evaluateScalar(x_.data(), y_.data(), 2, 3);
    EXPECT_NEAR(actual, expected, 1e-12);

    expected = popCov(x_.data(), y_.data(), 1, 3);
    actual = op.evaluateScalar(x_.data(), y_.data(), 3, 3);
    EXPECT_NEAR(actual, expected, 1e-12);

    expected = popCov(x_.data(), y_.data(), 7, 9);
    actual = op.evaluateScalar(x_.data(), y_.data(), 9, 3);
    EXPECT_NEAR(actual, expected, 1e-12);
}

TEST_F(RollingCovOpTest, ScalarPerfectLinear) {
    // y = 2*x → covariance = 2 * var(x)
    // For x = [1,2,3] (window 3): var(x) = 2/3, so cov = 4/3
    RollingCovOp op(3);
    double actual = op.evaluateScalar(x_.data(), y_.data(), 2, 3);
    double expected = popCov(x_.data(), y_.data(), 0, 2);
    EXPECT_NEAR(actual, expected, 1e-12);
    // Verify: cov(x, 2x) = 2 * var(x)
    double varX = 0.0;
    double meanX = (1.0 + 2.0 + 3.0) / 3.0;
    for (std::size_t j = 0; j < 3; ++j) {
        double d = x_[j] - meanX;
        varX += d * d;
    }
    varX /= 3.0;
    EXPECT_NEAR(actual, 2.0 * varX, 1e-12);
}

TEST_F(RollingCovOpTest, ScalarConstant) {
    // Constant Y → zero covariance
    double a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    RollingCovOp op(3);
    double actual = op.evaluateScalar(a, b, 2, 3);
    EXPECT_NEAR(actual, 0.0, 1e-12);
}

TEST_F(RollingCovOpTest, ScalarWindow1) {
    RollingCovOp op(1);
    // Single point: dx = dy = 0, cov = 0
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(x_.data(), y_.data(), i, 1), 0.0);
    }
}

TEST_F(RollingCovOpTest, ScalarNegativeRelation) {
    double a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    RollingCovOp op(3);
    double actual = op.evaluateScalar(a, b, 2, 3);
    double expected = popCov(a, b, 0, 2);
    EXPECT_NEAR(actual, expected, 1e-12);
    // Should be negative
    EXPECT_LT(actual, 0.0);
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingCovOpTest, BatchNoNulls) {
    RollingCovOp op(3);
    std::vector<double> output(x_.size());
    op.evaluate(ColView<double>(x_.data(), x_.size()),
                ColView<double>(y_.data(), y_.size()),
                output.data());

    for (std::size_t i = 0; i < x_.size(); ++i) {
        double expected = op.evaluateScalar(x_.data(), y_.data(), i, 3);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(output[i])) << "i=" << i;
        else
            EXPECT_NEAR(output[i], expected, 1e-12) << "i=" << i;
    }
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(RollingCovOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 200;
    RollingCovOp op(15);
    std::vector<double> inputX(N), inputY(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) {
        inputX[i] = static_cast<double>(i) * 1.1;
        inputY[i] = static_cast<double>(i) * 0.7 + 3.0;
    }

    op.evaluateSimd<SimdLevel::SCALAR>(
        ColView<double>(inputX.data(), N),
        ColView<double>(inputY.data(), N),
        simdOut.data(), 15);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(inputX.data(), inputY.data(), i, 15);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            // Running sums vs two-pass: use wider tolerance
            EXPECT_NEAR(simdOut[i], expected, 1e-10) << "i=" << i;
    }
}

TEST_F(RollingCovOpTest, SimdPerfectLinear) {
    constexpr std::size_t N = 50;
    std::vector<double> inputX(N), inputY(N), output(N);
    for (std::size_t i = 0; i < N; ++i) {
        inputX[i] = static_cast<double>(i);
        inputY[i] = 2.0 * inputX[i] + 1.0;  // perfect linear: y = 2x + 1
    }

    RollingCovOp op(10);
    op.evaluateSimd<SimdLevel::SCALAR>(
        ColView<double>(inputX.data(), N),
        ColView<double>(inputY.data(), N),
        output.data(), 10);

    // cov(x, 2x+1) = 2 * var(x)
    for (std::size_t i = 9; i < N; ++i) {
        // Compute var of window
        double sumX = 0.0, sumXSq = 0.0;
        std::size_t start = i + 1 - 10;
        for (std::size_t j = start; j <= i; ++j) {
            sumX += inputX[j];
            sumXSq += inputX[j] * inputX[j];
        }
        double varX = sumXSq / 10.0 - (sumX / 10.0) * (sumX / 10.0);
        if (varX < 0.0) varX = 0.0;
        double expected = 2.0 * varX;
        EXPECT_NEAR(output[i], expected, 1e-10) << "i=" << i;
    }
}
