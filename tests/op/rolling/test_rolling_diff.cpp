// test_rolling_diff.cpp — unit tests for RollingDiffOp (差分)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingDiff.h"

using namespace quantcore;

class RollingRollingDiffOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0};
    }
    std::vector<double> input_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(RollingRollingDiffOpTest, OpCode) {
    RollingDiffOp op(1);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_DIFF);
}

TEST_F(RollingRollingDiffOpTest, Name) {
    RollingDiffOp op(1);
    EXPECT_STREQ(op.opName(), "rolling_diff");
    static_assert(RollingDiffOp::kOpCode == RollingOpCode::ROLLING_DIFF);
}

TEST_F(RollingRollingDiffOpTest, WindowAccess) {
    RollingDiffOp op(5);
    EXPECT_EQ(op.window(), 5u);
}

// ============================================================
// evaluateScalar — basic
// ============================================================

TEST_F(RollingRollingDiffOpTest, ScalarDiff1) {
    // diff(x, 1) = x[i] - x[i-1]
    RollingDiffOp op(1);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 1)));  // i < 1
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 1, 1), 3.0 - 1.0);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 1), 6.0 - 3.0);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 6, 1), 28.0 - 21.0);
}

TEST_F(RollingRollingDiffOpTest, ScalarDiff3) {
    // diff(x, 3) = x[i] - x[i-3]
    RollingDiffOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 2, 3)));
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3), 10.0 - 1.0);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 6, 3), 28.0 - 10.0);
}

TEST_F(RollingRollingDiffOpTest, ScalarConstantSeries) {
    double data[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    RollingDiffOp op(2);
    for (std::size_t i = 2; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(data, i, 2), 0.0);
    }
}

// ============================================================
// evaluateScalar — special values
// ============================================================

TEST_F(RollingRollingDiffOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, nan, 3.0};
    RollingDiffOp op(1);
    double result = op.evaluateScalar(data, 2, 1);
    EXPECT_TRUE(std::isnan(result));
}

TEST_F(RollingRollingDiffOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    double data[] = {1.0, inf, 3.0};
    RollingDiffOp op(1);
    double result = op.evaluateScalar(data, 2, 1);
    EXPECT_DOUBLE_EQ(result, 3.0 - inf);  // -Inf
    EXPECT_TRUE(std::isinf(result));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingRollingDiffOpTest, BatchNoNulls) {
    RollingDiffOp op(1);
    std::size_t n = input_.size();
    std::vector<double> output(n);
    op.evaluate(ColView<double>(input_.data(), input_.size()), output.data());

    EXPECT_TRUE(std::isnan(output[0]));
    for (std::size_t i = 1; i < n; ++i) {
        EXPECT_DOUBLE_EQ(output[i], input_[i] - input_[i - 1]) << "i=" << i;
    }
}

TEST_F(RollingRollingDiffOpTest, BatchWithNulls) {
    double input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double output[6] = {};
    // Mark positions 3 and 4 as null
    uint64_t mask[1] = {(1ULL << 3) | (1ULL << 4)};

    RollingDiffOp op(2);
    op.evaluate(ColView<double>(input, 6, mask), output);

    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_DOUBLE_EQ(output[2], 3.0 - 1.0);  // non-null
    EXPECT_DOUBLE_EQ(output[3], 0.0);         // null
    EXPECT_DOUBLE_EQ(output[4], 0.0);         // null
    EXPECT_DOUBLE_EQ(output[5], 6.0 - 4.0);  // non-null
}

TEST_F(RollingRollingDiffOpTest, BatchLargeArray) {
    constexpr std::size_t N = 1000;
    RollingDiffOp op(5);
    std::vector<double> input(N), output(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i);

    op.evaluate(ColView<double>(input.data(), input.size()), output.data());

    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_TRUE(std::isnan(output[i])) << "i=" << i;
    }
    for (std::size_t i = 5; i < N; ++i) {
        EXPECT_DOUBLE_EQ(output[i], input[i] - input[i - 5]) << "i=" << i;
    }
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(RollingRollingDiffOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 256;
    RollingDiffOp op(7);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i) * 1.5 - 50.0;

    input[50] = std::numeric_limits<double>::quiet_NaN();
    input[150] = std::numeric_limits<double>::infinity();

    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 7);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 7);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(simdOut[i], expected) << "i=" << i;
    }
}

// ============================================================
// Pandas cross-validation
// ============================================================

TEST_F(RollingRollingDiffOpTest, MatchesPandasDiff1) {
    // pd.Series([10,20,30,40,50,60,70]).diff(1)
    // → [NaN, 10, 10, 10, 10, 10, 10]
    double data[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0};
    RollingDiffOp op(1);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 0, 1)));
    for (std::size_t i = 1; i < 7; ++i)
        EXPECT_DOUBLE_EQ(op.evaluateScalar(data, i, 1), 10.0) << "i=" << i;
}

TEST_F(RollingRollingDiffOpTest, MatchesPandasDiff3) {
    // pd.Series([10,20,30,40,50,60,70]).diff(3)
    // → [NaN, NaN, NaN, 30, 30, 30, 30]
    double data[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0};
    RollingDiffOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 1, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 2, 3)));
    for (std::size_t i = 3; i < 7; ++i)
        EXPECT_DOUBLE_EQ(op.evaluateScalar(data, i, 3), 30.0) << "i=" << i;
}

TEST_F(RollingRollingDiffOpTest, MatchesPandasDiff0) {
    // pd.Series([1,2,3]).diff(0) → [0, 0, 0]
    double data[] = {1.0, 2.0, 3.0};
    RollingDiffOp op(0);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(op.evaluateScalar(data, i, 0), 0.0) << "i=" << i;
}

// ============================================================
// Statelessness
// ============================================================

TEST_F(RollingRollingDiffOpTest, MultipleInstancesIdentical) {
    RollingDiffOp op1(3), op2(3);
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
