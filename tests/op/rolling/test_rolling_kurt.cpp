// test_rolling_kurt.cpp — unit tests for RollingKurtOp (滚动峰度, 有偏)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingKurt.h"

using namespace quantcore;

class RollingKurtOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    }

    // Helper: biased excess kurtosis of values in [start, end]
    // kurt = m4 / m2^2 - 3
    static double kurt(const double* data, std::size_t start, std::size_t end) {
        std::size_t count = end - start + 1;
        double sum = 0.0;
        for (std::size_t j = start; j <= end; ++j) sum += data[j];
        double mean = sum / static_cast<double>(count);

        double m2 = 0.0, m4 = 0.0;
        for (std::size_t j = start; j <= end; ++j) {
            double diff = data[j] - mean;
            double diff2 = diff * diff;
            m2 += diff2;
            m4 += diff2 * diff2;
        }
        m2 /= static_cast<double>(count);
        m4 /= static_cast<double>(count);
        if (m2 == 0.0) return std::numeric_limits<double>::quiet_NaN();
        return m4 / (m2 * m2) - 3.0;
    }

    std::vector<double> input_;
};

TEST_F(RollingKurtOpTest, OpCode) {
    RollingKurtOp op(4);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_KURT);
}

TEST_F(RollingKurtOpTest, Name) {
    RollingKurtOp op(4);
    EXPECT_STREQ(op.opName(), "rolling_kurt");
}

TEST_F(RollingKurtOpTest, WindowAccess) {
    RollingKurtOp op(5);
    EXPECT_EQ(op.window(), 5u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingKurtOpTest, ScalarWindow4) {
    RollingKurtOp op(4);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 4)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 4)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 2, 4)));

    double expected = kurt(input_.data(), 0, 3);
    EXPECT_NEAR(op.evaluateScalar(input_.data(), 3, 4), expected, 1e-12);

    expected = kurt(input_.data(), 1, 4);
    EXPECT_NEAR(op.evaluateScalar(input_.data(), 4, 4), expected, 1e-12);

    expected = kurt(input_.data(), 6, 9);
    EXPECT_NEAR(op.evaluateScalar(input_.data(), 9, 4), expected, 1e-12);
}

TEST_F(RollingKurtOpTest, ScalarConstant) {
    // Constant series: variance = 0 → kurtosis is undefined (returns NaN)
    double data[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    RollingKurtOp op(4);
    for (std::size_t i = 3; i < 5; ++i) {
        EXPECT_TRUE(std::isnan(op.evaluateScalar(data, i, 4)));
    }
}

TEST_F(RollingKurtOpTest, ScalarNormalDistribution) {
    // Approximate normal-ish data: kurtosis should be close to 0 (excess ≈ 0)
    double data[] = {0.5, -0.3, 1.2, 0.8, -1.0, 0.1, -0.5, 0.9, -0.2, 1.5};
    RollingKurtOp op(10);
    double result = op.evaluateScalar(data, 9, 10);
    // For normal data, excess kurtosis ≈ 0. Allow wide tolerance due to small sample.
    EXPECT_GT(result, -3.0);
    EXPECT_LT(result, 3.0);
}

TEST_F(RollingKurtOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, 3.0, nan, 5.0};
    RollingKurtOp op(4);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 4, 4)));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingKurtOpTest, BatchNoNulls) {
    RollingKurtOp op(4);
    std::vector<double> output(input_.size());
    op.evaluate(ColView<double>(input_.data(), input_.size()), output.data());

    for (std::size_t i = 0; i < input_.size(); ++i) {
        double expected = op.evaluateScalar(input_.data(), i, 4);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(output[i])) << "i=" << i;
        else
            EXPECT_NEAR(output[i], expected, 1e-12) << "i=" << i;
    }
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(RollingKurtOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 100;
    RollingKurtOp op(8);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i % 17) * 1.5;

    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 8);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 8);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_NEAR(simdOut[i], expected, 1e-10) << "i=" << i;
    }
}
