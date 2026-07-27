// test_rolling_mul.cpp — unit tests for RollingMulOp (滚动连乘)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingMul.h"

using namespace quantcore;

class RollingMulOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    }

    // Helper: product of values in [start, end]
    static double prod(const double* data, std::size_t start, std::size_t end) {
        double p = 1.0;
        for (std::size_t j = start; j <= end; ++j) p *= data[j];
        return p;
    }

    std::vector<double> input_;
};

TEST_F(RollingMulOpTest, OpCode) {
    RollingMulOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_MUL);
}

TEST_F(RollingMulOpTest, Name) {
    RollingMulOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_mul");
}

TEST_F(RollingMulOpTest, WindowAccess) {
    RollingMulOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingMulOpTest, ScalarWindow3) {
    RollingMulOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), prod(input_.data(), 0, 2));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3), prod(input_.data(), 1, 3));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 9, 3), prod(input_.data(), 7, 9));
}

TEST_F(RollingMulOpTest, ScalarWindow1) {
    RollingMulOp op(1);
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), input_[i]);
    }
}

TEST_F(RollingMulOpTest, ScalarWithZero) {
    double data[] = {2.0, 0.0, 3.0, 4.0};
    RollingMulOp op(3);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 2, 3), 0.0);  // 2*0*3 = 0
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 3), 0.0);  // 0*3*4 = 0
}

TEST_F(RollingMulOpTest, ScalarWithOne) {
    double data[] = {1.0, 5.0, 1.0, 5.0};
    RollingMulOp op(3);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 2, 3), 5.0);   // 1*5*1 = 5
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 3), 25.0);  // 5*1*5 = 25
}

TEST_F(RollingMulOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, nan, 4.0};
    RollingMulOp op(3);
    double result = op.evaluateScalar(data, 3, 3);
    EXPECT_TRUE(std::isnan(result));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingMulOpTest, BatchNoNulls) {
    RollingMulOp op(3);
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

TEST_F(RollingMulOpTest, BatchWithNulls) {
    double input[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    double output[6] = {};
    uint64_t mask[1] = {(1ULL << 2)};  // pos 2 null

    RollingMulOp op(3);
    op.evaluate(ColView<double>(input, 6, mask), output);

    EXPECT_DOUBLE_EQ(output[2], 0.0);  // null at pos 2
    EXPECT_DOUBLE_EQ(output[3], 3.0 * 4.0 * 5.0);  // window: input[1..3]
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(RollingMulOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 100;
    RollingMulOp op(5);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = 1.0 + static_cast<double>(i) * 0.05;

    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 5);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 5);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_NEAR(simdOut[i], expected, 1e-12) << "i=" << i;
    }
}
