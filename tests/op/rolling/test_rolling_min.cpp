// test_rolling_min.cpp — unit tests for RollingMinOp (滚动最小值)
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingMin.h"

using namespace quantcore;

class RollingMinOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 0.5};
    }
    std::vector<double> input_;
};

TEST_F(RollingMinOpTest, OpCode) {
    RollingMinOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_MIN);
}

TEST_F(RollingMinOpTest, Name) {
    RollingMinOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_min");
}

TEST_F(RollingMinOpTest, WindowAccess) {
    RollingMinOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingMinOpTest, ScalarWindow3) {
    RollingMinOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), 1.0);  // min(3,1,4)
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3), 1.0);  // min(1,4,1)
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 4, 3), 1.0);  // min(4,1,5)
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 5, 3), 1.0);  // min(1,5,9)
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 6, 3), 2.0);  // min(5,9,2)
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 9, 3), 0.5);  // min(6,5,0.5)
}

TEST_F(RollingMinOpTest, ScalarWindow1) {
    RollingMinOp op(1);
    for (std::size_t i = 0; i < input_.size(); ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), input_[i]);
    }
}

TEST_F(RollingMinOpTest, ScalarIncreasing) {
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    RollingMinOp op(3);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 2, 3), 1.0);  // min(1,2,3)
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 3), 2.0);  // min(2,3,4)
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 4, 3), 3.0);  // min(3,4,5)
}

TEST_F(RollingMinOpTest, ScalarNegativeInf) {
    double negInf = -std::numeric_limits<double>::infinity();
    double data[] = {1.0, negInf, 3.0, 4.0};
    RollingMinOp op(3);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 2, 3), negInf);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 3), negInf);
}

TEST_F(RollingMinOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, nan, 4.0};
    RollingMinOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 2, 3)));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingMinOpTest, BatchNoNulls) {
    RollingMinOp op(3);
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

TEST_F(RollingMinOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 200;
    RollingMinOp op(8);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i % 30) * 3.0;

    // Note: NaN/Inf are tested via scalar path; SIMD uses clean data.
    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 8);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 8);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(simdOut[i], expected) << "i=" << i;
    }
}
