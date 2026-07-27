// test_rolling_argmax.cpp — unit tests for RollingArgMaxOp (滚动窗口内最大值位置)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingArgMax.h"

using namespace quantcore;

class RollingArgMaxOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    }

    // Helper: position of first max in [start, end] (0-based within window)
    static std::size_t argmax(const double* data, std::size_t start, std::size_t end) {
        double maxVal = data[start];
        std::size_t pos = 0;
        for (std::size_t j = start + 1; j <= end; ++j) {
            if (data[j] > maxVal) {
                maxVal = data[j];
                pos = j - start;
            }
        }
        return pos;
    }

    std::vector<double> input_;
};

TEST_F(RollingArgMaxOpTest, OpCode) {
    RollingArgMaxOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_ARGMAX);
}

TEST_F(RollingArgMaxOpTest, Name) {
    RollingArgMaxOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_argmax");
}

TEST_F(RollingArgMaxOpTest, WindowAccess) {
    RollingArgMaxOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingArgMaxOpTest, ScalarWindow3) {
    RollingArgMaxOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3),
                     static_cast<double>(argmax(input_.data(), 0, 2)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3),
                     static_cast<double>(argmax(input_.data(), 1, 3)));
}

TEST_F(RollingArgMaxOpTest, ScalarMonotonicIncreasing) {
    // Window is always increasing, so max is always at the last position
    RollingArgMaxOp op(3);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), 2.0);  // max at pos 2
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 5, 3), 2.0);  // max at pos 2
}

TEST_F(RollingArgMaxOpTest, ScalarFirstMax) {
    // Test that first occurrence of max is returned
    double data[] = {5.0, 3.0, 5.0, 2.0};  // max=5 at positions 0 and 2
    RollingArgMaxOp op(3);
    // Window [5,3,5]: max=5, first at pos 0
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 2, 3), 0.0);
    // Window [3,5,2]: max=5, first at pos 1
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 3), 1.0);
}

TEST_F(RollingArgMaxOpTest, ScalarWindow1) {
    RollingArgMaxOp op(1);
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), 0.0);
    }
}

TEST_F(RollingArgMaxOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, nan, 4.0};
    RollingArgMaxOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 2, 3)));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingArgMaxOpTest, BatchNoNulls) {
    RollingArgMaxOp op(3);
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

TEST_F(RollingArgMaxOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 200;
    RollingArgMaxOp op(10);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = std::sin(static_cast<double>(i) * 0.5);

    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 10);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 10);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(simdOut[i], expected) << "i=" << i;
    }
}
