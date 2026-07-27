// test_rolling_sum.cpp — unit tests for RollingSumOp (滚动求和)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingSum.h"

using namespace quantcore;

class RollingSumOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    }
    std::vector<double> input_;
};

TEST_F(RollingSumOpTest, OpCode) {
    RollingSumOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_SUM);
}

TEST_F(RollingSumOpTest, Name) {
    RollingSumOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_sum");
    static_assert(RollingSumOp::kOpCode == RollingOpCode::ROLLING_SUM);
}

TEST_F(RollingSumOpTest, WindowAccess) {
    RollingSumOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingSumOpTest, ScalarWindow3) {
    RollingSumOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), 1.0 + 2.0 + 3.0);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3), 2.0 + 3.0 + 4.0);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 9, 3), 8.0 + 9.0 + 10.0);
}

TEST_F(RollingSumOpTest, ScalarWindow1) {
    RollingSumOp op(1);
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), input_[i]);
    }
}

TEST_F(RollingSumOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, nan, 4.0};
    RollingSumOp op(3);
    double result = op.evaluateScalar(data, 3, 3);
    EXPECT_TRUE(std::isnan(result));
}

TEST_F(RollingSumOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    double data[] = {1.0, inf, 3.0};
    RollingSumOp op(3);
    double result = op.evaluateScalar(data, 2, 3);
    EXPECT_TRUE(std::isinf(result));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingSumOpTest, BatchNoNulls) {
    RollingSumOp op(3);
    std::vector<double> output(input_.size());
    op.evaluate(ColView<double>(input_.data(), input_.size()), output.data());

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_DOUBLE_EQ(output[2], 6.0);
    EXPECT_DOUBLE_EQ(output[3], 9.0);
    EXPECT_DOUBLE_EQ(output[9], 27.0);
}

TEST_F(RollingSumOpTest, BatchWithNulls) {
    double input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double output[6] = {};
    uint64_t mask[1] = {(1ULL << 2)};  // pos 2 null

    RollingSumOp op(3);
    op.evaluate(ColView<double>(input, 6, mask), output);

    EXPECT_DOUBLE_EQ(output[2], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[3], 2.0 + 3.0 + 4.0);  // non-null
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(RollingSumOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 256;
    RollingSumOp op(12);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i) * 1.7;

    // Note: NaN/Inf are tested via scalar path; SIMD uses clean data.
    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 12);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 12);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(simdOut[i], expected) << "i=" << i;
    }
}
