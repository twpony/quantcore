// test_rolling_sma.cpp — unit tests for RollingSmaOp (简单移动平均)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingSma.h"

using namespace quantcore;

class RollingRollingSmaOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    }
    std::vector<double> input_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(RollingRollingSmaOpTest, OpCode) {
    RollingSmaOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_SMA);
}

TEST_F(RollingRollingSmaOpTest, Name) {
    RollingSmaOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_sma");
    static_assert(RollingSmaOp::kOpCode == RollingOpCode::ROLLING_SMA);
}

TEST_F(RollingRollingSmaOpTest, WindowAccess) {
    RollingSmaOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar — basic
// ============================================================

TEST_F(RollingRollingSmaOpTest, ScalarWindow3) {
    RollingSmaOp op(3);
    // window=3, need i >= 2
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    // i=2: mean(1,2,3) = 2
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), 2.0);
    // i=3: mean(2,3,4) = 3
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3), 3.0);
    // i=9: mean(8,9,10) = 9
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 9, 3), 9.0);
}

TEST_F(RollingRollingSmaOpTest, ScalarWindow1) {
    RollingSmaOp op(1);
    // With window=1, each value is its own mean
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), input_[i]);
    }
}

TEST_F(RollingRollingSmaOpTest, ScalarWindow5) {
    RollingSmaOp op(5);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 3, 5)));
    // i=4: mean(1,2,3,4,5) = 3
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 4, 5), 3.0);
    // i=5: mean(2,3,4,5,6) = 4
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 5, 5), 4.0);
    // i=9: mean(6,7,8,9,10) = 8
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 9, 5), 8.0);
}

TEST_F(RollingRollingSmaOpTest, ScalarConstantSeries) {
    double data[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    RollingSmaOp op(3);
    for (std::size_t i = 2; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(data, i, 3), 5.0);
    }
}

// ============================================================
// evaluateScalar — special values
// ============================================================

TEST_F(RollingRollingSmaOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, nan, 4.0};
    RollingSmaOp op(3);
    double result = op.evaluateScalar(data, 3, 3);  // window over {2.0, nan, 4.0}
    EXPECT_TRUE(std::isnan(result));
}

TEST_F(RollingRollingSmaOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    double data[] = {1.0, 2.0, inf, 4.0};
    RollingSmaOp op(3);
    double result = op.evaluateScalar(data, 3, 3);  // window over {2.0, inf, 4.0}
    EXPECT_TRUE(std::isinf(result));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingRollingSmaOpTest, BatchNoNulls) {
    RollingSmaOp op(3);
    std::size_t n = input_.size();
    std::vector<double> output(n);
    op.evaluate(ColView<double>(input_.data(), input_.size()), output.data());

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_DOUBLE_EQ(output[2], (1.0 + 2.0 + 3.0) / 3.0);
    EXPECT_DOUBLE_EQ(output[3], (2.0 + 3.0 + 4.0) / 3.0);
    EXPECT_DOUBLE_EQ(output[9], (8.0 + 9.0 + 10.0) / 3.0);
}

TEST_F(RollingRollingSmaOpTest, BatchWithNulls) {
    double input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double output[6] = {};
    uint64_t mask[1] = {(1ULL << 2) | (1ULL << 4)};  // pos 2,4 null

    RollingSmaOp op(3);
    op.evaluate(ColView<double>(input, 6, mask), output);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_DOUBLE_EQ(output[2], 0.0);    // null
    EXPECT_DOUBLE_EQ(output[3], (2.0 + 3.0 + 4.0) / 3.0);  // non-null (but 3 is null)
    EXPECT_DOUBLE_EQ(output[4], 0.0);    // null
    EXPECT_DOUBLE_EQ(output[5], (4.0 + 5.0 + 6.0) / 3.0);  // non-null (but 4 is null)
}

TEST_F(RollingRollingSmaOpTest, BatchLargeArray) {
    constexpr std::size_t N = 1000;
    RollingSmaOp op(10);
    std::vector<double> input(N), output(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i);

    op.evaluate(ColView<double>(input.data(), input.size()), output.data());

    for (std::size_t i = 0; i < 9; ++i) {
        EXPECT_TRUE(std::isnan(output[i])) << "i=" << i;
    }
    // i=9: mean of 0..9 = 4.5
    EXPECT_DOUBLE_EQ(output[9], 4.5);
    // i=99: mean of 90..99 = 94.5
    EXPECT_DOUBLE_EQ(output[99], 94.5);
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(RollingRollingSmaOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 256;
    RollingSmaOp op(15);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i) * 1.5;

    // Note: NaN/Inf are tested via scalar path; SIMD cross-validation uses
    // clean data since NaN handling requires special care in running-sum SIMD paths.
    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 15);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 15);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(simdOut[i], expected) << "i=" << i;
    }
}

TEST_F(RollingRollingSmaOpTest, SimdSmallArray) {
    RollingSmaOp op(2);
    double input[] = {2.0, 4.0, 6.0, 8.0};
    double output[4] = {};
    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input, 4), output, 2);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_DOUBLE_EQ(output[1], 3.0);
    EXPECT_DOUBLE_EQ(output[2], 5.0);
    EXPECT_DOUBLE_EQ(output[3], 7.0);
}

// ============================================================
// Statelessness
// ============================================================

TEST_F(RollingRollingSmaOpTest, MultipleInstancesIdentical) {
    RollingSmaOp op1(3), op2(3);
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    for (std::size_t i = 0; i < 5; ++i) {
        double v1 = op1.evaluateScalar(data, i, 3);
        double v2 = op2.evaluateScalar(data, i, 3);
        if (std::isnan(v1))
            EXPECT_TRUE(std::isnan(v2));
        else
            EXPECT_DOUBLE_EQ(v1, v2);
    }
}
