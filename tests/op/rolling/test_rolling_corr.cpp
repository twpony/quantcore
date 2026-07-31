// test_rolling_corr.cpp — unit tests for RollingCorrOp (滚动相关系数)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/rolling/RollingCorr.h"

using namespace quantcore;

class RollingCorrOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        x_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        y_ = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0};
    }

    // Helper: Pearson correlation of values in [start, end]
    static double pearsonCorr(const double* x, const double* y,
                              std::size_t start, std::size_t end) {
        std::size_t count = end - start + 1;
        double sumX = 0.0, sumY = 0.0;
        for (std::size_t j = start; j <= end; ++j) {
            sumX += x[j];
            sumY += y[j];
        }
        double meanX = sumX / static_cast<double>(count);
        double meanY = sumY / static_cast<double>(count);

        double cov = 0.0, varX = 0.0, varY = 0.0;
        for (std::size_t j = start; j <= end; ++j) {
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
        if (denom == 0.0) return std::numeric_limits<double>::quiet_NaN();
        return cov / denom;
    }

    std::vector<double> x_;
    std::vector<double> y_;
};

TEST_F(RollingCorrOpTest, OpCode) {
    RollingCorrOp op(3);
    EXPECT_EQ(op.kOpCode, RollingOpCode::ROLLING_CORR);
}

TEST_F(RollingCorrOpTest, Name) {
    RollingCorrOp op(3);
    EXPECT_STREQ(op.name, "rolling_corr");
}

TEST_F(RollingCorrOpTest, WindowAccess) {
    RollingCorrOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingCorrOpTest, ScalarWindow3) {
    RollingCorrOp op(3);
    // First 2 positions: insufficient data
    EXPECT_TRUE(std::isnan(op.evaluateScalar(x_.data(), y_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(x_.data(), y_.data(), 1, 3)));

    double expected = pearsonCorr(x_.data(), y_.data(), 0, 2);
    double actual = op.evaluateScalar(x_.data(), y_.data(), 2, 3);
    EXPECT_NEAR(actual, expected, 1e-12);

    expected = pearsonCorr(x_.data(), y_.data(), 1, 3);
    actual = op.evaluateScalar(x_.data(), y_.data(), 3, 3);
    EXPECT_NEAR(actual, expected, 1e-12);

    expected = pearsonCorr(x_.data(), y_.data(), 7, 9);
    actual = op.evaluateScalar(x_.data(), y_.data(), 9, 3);
    EXPECT_NEAR(actual, expected, 1e-12);
}

TEST_F(RollingCorrOpTest, ScalarPerfectCorrelation) {
    // y = 2*x → perfect positive correlation
    RollingCorrOp op(3);
    double actual = op.evaluateScalar(x_.data(), y_.data(), 2, 3);
    EXPECT_NEAR(actual, 1.0, 1e-12);
}

TEST_F(RollingCorrOpTest, ScalarPerfectNegativeCorrelation) {
    // y = -x → perfect negative correlation
    double a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b[] = {-1.0, -2.0, -3.0, -4.0, -5.0};
    RollingCorrOp op(3);
    double actual = op.evaluateScalar(a, b, 2, 3);
    EXPECT_NEAR(actual, -1.0, 1e-12);
}

TEST_F(RollingCorrOpTest, ScalarConstantY) {
    // Constant Y → zero variance → NaN correlation
    double a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    RollingCorrOp op(3);
    double actual = op.evaluateScalar(a, b, 2, 3);
    EXPECT_TRUE(std::isnan(actual));
}

TEST_F(RollingCorrOpTest, ScalarWindow1) {
    RollingCorrOp op(1);
    // Single point correlation is undefined (0/0)
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isnan(op.evaluateScalar(x_.data(), y_.data(), i, 1)));
    }
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingCorrOpTest, BatchNoNulls) {
    RollingCorrOp op(3);
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

TEST_F(RollingCorrOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 200;
    RollingCorrOp op(15);
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
            // Running sum-of-squares vs two-pass: use wider tolerance
            EXPECT_NEAR(simdOut[i], expected, 1e-10) << "i=" << i;
    }
}

TEST_F(RollingCorrOpTest, SimdPerfectCorr) {
    constexpr std::size_t N = 50;
    std::vector<double> inputX(N), inputY(N), output(N);
    for (std::size_t i = 0; i < N; ++i) {
        inputX[i] = static_cast<double>(i);
        inputY[i] = 2.0 * inputX[i] + 1.0;  // perfect linear
    }

    RollingCorrOp op(10);
    op.evaluateSimd<SimdLevel::SCALAR>(
        ColView<double>(inputX.data(), N),
        ColView<double>(inputY.data(), N),
        output.data(), 10);

    for (std::size_t i = 9; i < N; ++i) {
        EXPECT_NEAR(output[i], 1.0, 1e-10) << "i=" << i;
    }
}
