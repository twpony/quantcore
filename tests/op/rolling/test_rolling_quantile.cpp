// test_rolling_quantile.cpp — unit tests for RollingQuantileOp (滚动分位数)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingQuantile.h"

using namespace quantcore;

class RollingQuantileOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    }

    // Helper: quantile with linear interpolation in [start, end]
    static double quantile(const double* data, std::size_t start, std::size_t end, double q) {
        std::size_t count = end - start + 1;
        std::vector<double> buf(count);
        for (std::size_t j = 0; j < count; ++j) buf[j] = data[start + j];
        std::sort(buf.begin(), buf.end());

        double pos = q * static_cast<double>(count - 1);
        std::size_t lower = static_cast<std::size_t>(pos);
        std::size_t upper = (lower + 1 < count) ? lower + 1 : lower;
        double fraction = pos - static_cast<double>(lower);

        if (lower == upper) return buf[lower];
        return buf[lower] * (1.0 - fraction) + buf[upper] * fraction;
    }

    std::vector<double> input_;
};

TEST_F(RollingQuantileOpTest, OpCode) {
    RollingQuantileOp op(3, 0.5);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_QUANTILE);
}

TEST_F(RollingQuantileOpTest, Name) {
    RollingQuantileOp op(3, 0.5);
    EXPECT_STREQ(op.opName(), "rolling_quantile");
}

TEST_F(RollingQuantileOpTest, WindowAndQAccess) {
    RollingQuantileOp op(4, 0.25);
    EXPECT_EQ(op.window(), 4u);
    EXPECT_DOUBLE_EQ(op.q(), 0.25);
}

// ============================================================
// evaluateScalar — median (q=0.5)
// ============================================================

TEST_F(RollingQuantileOpTest, ScalarMedianWindow3) {
    RollingQuantileOp op(3, 0.5);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3),
                     quantile(input_.data(), 0, 2, 0.5));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3),
                     quantile(input_.data(), 1, 3, 0.5));
}

// ============================================================
// evaluateScalar — minimum (q=0.0)
// ============================================================

TEST_F(RollingQuantileOpTest, ScalarMinWindow3) {
    RollingQuantileOp op(3, 0.0);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), 1.0);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3), 2.0);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 9, 3), 8.0);
}

// ============================================================
// evaluateScalar — maximum (q=1.0)
// ============================================================

TEST_F(RollingQuantileOpTest, ScalarMaxWindow3) {
    RollingQuantileOp op(3, 1.0);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), 3.0);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3), 4.0);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 9, 3), 10.0);
}

// ============================================================
// evaluateScalar — quartile (q=0.25)
// ============================================================

TEST_F(RollingQuantileOpTest, ScalarQuartileWindow4) {
    RollingQuantileOp op(4, 0.25);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 2, 4)));

    double expected = quantile(input_.data(), 0, 3, 0.25);
    EXPECT_NEAR(op.evaluateScalar(input_.data(), 3, 4), expected, 1e-12);

    expected = quantile(input_.data(), 1, 4, 0.25);
    EXPECT_NEAR(op.evaluateScalar(input_.data(), 4, 4), expected, 1e-12);
}

// ============================================================
// evaluateScalar — with linear interpolation
// ============================================================

TEST_F(RollingQuantileOpTest, ScalarLinearInterpolation) {
    // Window of {1, 2, 4, 5}, q=0.5: pos=1.5, interpolate between 2 and 4 → 3
    double data[] = {2.0, 1.0, 5.0, 4.0};
    RollingQuantileOp op(4, 0.5);
    // Sorted: {1, 2, 4, 5}, pos=0.5*3=1.5, lower=1(buf[1]=2), upper=2(buf[2]=4)
    // result = 2*(1-0.5) + 4*0.5 = 1 + 2 = 3
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 4), 3.0);
}

TEST_F(RollingQuantileOpTest, ScalarExactPosition) {
    // Window of {1, 2, 3, 4, 5}, q=0.5: pos=2.0, exact → 3
    RollingQuantileOp op(5, 0.5);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 4, 5), 3.0);
}

TEST_F(RollingQuantileOpTest, ScalarWindow1) {
    RollingQuantileOp op(1, 0.5);
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), input_[i]);
    }
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingQuantileOpTest, BatchNoNulls) {
    RollingQuantileOp op(3, 0.5);
    std::vector<double> output(input_.size());
    op.evaluate(ColView<double>(input_.data(), input_.size()), output.data());

    for (std::size_t i = 0; i < input_.size(); ++i) {
        double expected = op.evaluateScalar(input_.data(), i, 3);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(output[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(output[i], expected) << "i=" << i;
    }
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(RollingQuantileOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 100;
    RollingQuantileOp op(7, 0.3);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i % 13) * 2.1;

    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 7);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 7);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(simdOut[i], expected) << "i=" << i;
    }
}
