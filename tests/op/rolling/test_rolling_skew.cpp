// test_rolling_skew.cpp — unit tests for RollingSkewOp (滚动偏度, 有偏)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingSkew.h"

using namespace quantcore;

class RollingSkewOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    }

    // Helper: biased skewness of values in [start, end]
    // skew = m3 / m2^(3/2)
    static double skew(const double* data, std::size_t start, std::size_t end) {
        std::size_t count = end - start + 1;
        double sum = 0.0;
        for (std::size_t j = start; j <= end; ++j) sum += data[j];
        double mean = sum / static_cast<double>(count);

        double m2 = 0.0, m3 = 0.0;
        for (std::size_t j = start; j <= end; ++j) {
            double diff = data[j] - mean;
            double diff2 = diff * diff;
            m2 += diff2;
            m3 += diff2 * diff;  // diff^2 * diff = diff^3
        }
        m2 /= static_cast<double>(count);
        m3 /= static_cast<double>(count);
        if (m2 == 0.0) return std::numeric_limits<double>::quiet_NaN();
        return m3 / (m2 * std::sqrt(m2));
    }

    std::vector<double> input_;
};

TEST_F(RollingSkewOpTest, OpCode) {
    RollingSkewOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_SKEW);
}

TEST_F(RollingSkewOpTest, Name) {
    RollingSkewOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_skew");
}

TEST_F(RollingSkewOpTest, WindowAccess) {
    RollingSkewOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingSkewOpTest, ScalarWindow3) {
    RollingSkewOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));

    double expected = skew(input_.data(), 0, 2);
    EXPECT_NEAR(op.evaluateScalar(input_.data(), 2, 3), expected, 1e-12);

    expected = skew(input_.data(), 1, 3);
    EXPECT_NEAR(op.evaluateScalar(input_.data(), 3, 3), expected, 1e-12);

    expected = skew(input_.data(), 7, 9);
    EXPECT_NEAR(op.evaluateScalar(input_.data(), 9, 3), expected, 1e-12);
}

TEST_F(RollingSkewOpTest, ScalarConstant) {
    // Constant series: variance = 0 → skewness is undefined (returns NaN)
    double data[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    RollingSkewOp op(3);
    for (std::size_t i = 2; i < 5; ++i) {
        EXPECT_TRUE(std::isnan(op.evaluateScalar(data, i, 3)));
    }
}

TEST_F(RollingSkewOpTest, ScalarSymmetric) {
    // Symmetric around mean: skew should be 0
    double data[] = {1.0, 3.0, 2.0, 3.0, 1.0};
    RollingSkewOp op(5);
    double result = op.evaluateScalar(data, 4, 5);
    EXPECT_NEAR(result, 0.0, 1e-12);
}

TEST_F(RollingSkewOpTest, ScalarRightSkewed) {
    // Right-skewed data: positive skew
    double data[] = {1.0, 1.0, 1.0, 1.0, 10.0};
    RollingSkewOp op(5);
    double result = op.evaluateScalar(data, 4, 5);
    EXPECT_GT(result, 0.0);
}

TEST_F(RollingSkewOpTest, ScalarLeftSkewed) {
    // Left-skewed data: negative skew
    double data[] = {10.0, 10.0, 10.0, 10.0, 1.0};
    RollingSkewOp op(5);
    double result = op.evaluateScalar(data, 4, 5);
    EXPECT_LT(result, 0.0);
}

TEST_F(RollingSkewOpTest, ScalarWindow1) {
    RollingSkewOp op(1);
    // 1 element: variance = 0 → NaN
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), i, 1)));
    }
}

TEST_F(RollingSkewOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, nan, 4.0};
    RollingSkewOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 2, 3)));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingSkewOpTest, BatchNoNulls) {
    RollingSkewOp op(3);
    std::vector<double> output(input_.size());
    op.evaluate(ColView<double>(input_.data(), input_.size()), output.data());

    for (std::size_t i = 0; i < input_.size(); ++i) {
        double expected = op.evaluateScalar(input_.data(), i, 3);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(output[i])) << "i=" << i;
        else
            EXPECT_NEAR(output[i], expected, 1e-12) << "i=" << i;
    }
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(RollingSkewOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 100;
    RollingSkewOp op(6);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i % 11) * 0.9;

    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 6);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 6);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_NEAR(simdOut[i], expected, 1e-10) << "i=" << i;
    }
}
