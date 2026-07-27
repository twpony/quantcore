// test_rolling_argmin.cpp — unit tests for RollingArgMinOp (滚动窗口内最小值位置)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingArgMin.h"

using namespace quantcore;

class RollingArgMinOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    }

    // Helper: position of first min in [start, end] (0-based within window)
    static std::size_t argmin(const double* data, std::size_t start, std::size_t end) {
        double minVal = data[start];
        std::size_t pos = 0;
        for (std::size_t j = start + 1; j <= end; ++j) {
            if (data[j] < minVal) {
                minVal = data[j];
                pos = j - start;
            }
        }
        return pos;
    }

    std::vector<double> input_;
};

TEST_F(RollingArgMinOpTest, OpCode) {
    RollingArgMinOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_ARGMIN);
}

TEST_F(RollingArgMinOpTest, Name) {
    RollingArgMinOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_argmin");
}

TEST_F(RollingArgMinOpTest, WindowAccess) {
    RollingArgMinOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingArgMinOpTest, ScalarWindow3) {
    RollingArgMinOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3),
                     static_cast<double>(argmin(input_.data(), 0, 2)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3),
                     static_cast<double>(argmin(input_.data(), 1, 3)));
}

TEST_F(RollingArgMinOpTest, ScalarMonotonicDecreasing) {
    // Window is always decreasing, so min is always at the last position
    RollingArgMinOp op(3);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), 2.0);  // min at pos 2
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 5, 3), 2.0);  // min at pos 2
}

TEST_F(RollingArgMinOpTest, ScalarFirstMin) {
    // Test that first occurrence of min is returned
    double data[] = {2.0, 5.0, 2.0, 3.0};  // min=2 at positions 0 and 2
    RollingArgMinOp op(3);
    // Window [2,5,2]: min=2, first at pos 0
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 2, 3), 0.0);
    // Window [5,2,3]: min=2, first at pos 1
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 3), 1.0);
}

TEST_F(RollingArgMinOpTest, ScalarWindow1) {
    RollingArgMinOp op(1);
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), 0.0);
    }
}

TEST_F(RollingArgMinOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, nan, 4.0};
    RollingArgMinOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 2, 3)));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingArgMinOpTest, BatchNoNulls) {
    RollingArgMinOp op(3);
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

TEST_F(RollingArgMinOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 200;
    RollingArgMinOp op(10);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = std::cos(static_cast<double>(i) * 0.7);

    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 10);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 10);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(simdOut[i], expected) << "i=" << i;
    }
}
