// test_rolling_mean.cpp — unit tests for RollingMeanOp (滚动均值)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingMean.h"

using namespace quantcore;

class RollingMeanOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    }

    // Helper: arithmetic mean of values in [start, end]
    static double mean(const double* data, std::size_t start, std::size_t end) {
        std::size_t count = end - start + 1;
        double sum = 0.0;
        for (std::size_t j = start; j <= end; ++j) sum += data[j];
        return sum / static_cast<double>(count);
    }

    std::vector<double> input_;
};

TEST_F(RollingMeanOpTest, OpCode) {
    RollingMeanOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_MEAN);
}

TEST_F(RollingMeanOpTest, Name) {
    RollingMeanOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_mean");
}

TEST_F(RollingMeanOpTest, WindowAccess) {
    RollingMeanOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingMeanOpTest, ScalarWindow3) {
    RollingMeanOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), mean(input_.data(), 0, 2));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3), mean(input_.data(), 1, 3));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 9, 3), mean(input_.data(), 7, 9));
}

TEST_F(RollingMeanOpTest, ScalarWindow1) {
    RollingMeanOp op(1);
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), input_[i]);
    }
}

TEST_F(RollingMeanOpTest, ScalarConstant) {
    double data[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    RollingMeanOp op(3);
    for (std::size_t i = 2; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(data, i, 3), 5.0);
    }
}

TEST_F(RollingMeanOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, nan, 4.0};
    RollingMeanOp op(3);
    double result = op.evaluateScalar(data, 3, 3);
    EXPECT_TRUE(std::isnan(result));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingMeanOpTest, BatchNoNulls) {
    RollingMeanOp op(3);
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

TEST_F(RollingMeanOpTest, BatchWithNulls) {
    double input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double output[6] = {};
    uint64_t mask[1] = {(1ULL << 2)};  // pos 2 null

    RollingMeanOp op(3);
    op.evaluate(ColView<double>(input, 6, mask), output);

    EXPECT_DOUBLE_EQ(output[2], 0.0);  // null
    EXPECT_DOUBLE_EQ(output[3], mean(input, 1, 3));  // non-null
    EXPECT_DOUBLE_EQ(output[4], mean(input, 2, 4));
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(RollingMeanOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 256;
    RollingMeanOp op(12);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i) * 1.7;

    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 12);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 12);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(simdOut[i], expected) << "i=" << i;
    }
}
