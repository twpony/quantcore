// test_rolling_rank.cpp — unit tests for RollingRankOp (滚动排名, 窗口内百分位)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingRank.h"

using namespace quantcore;

class RollingRankOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Mixed values for rank testing
        input_ = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0};
    }
    std::vector<double> input_;
};

TEST_F(RollingRankOpTest, OpCode) {
    RollingRankOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_RANK);
}

TEST_F(RollingRankOpTest, Name) {
    RollingRankOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_rank");
}

TEST_F(RollingRankOpTest, WindowAccess) {
    RollingRankOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingRankOpTest, ScalarWindow3) {
    RollingRankOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));

    // i=2: window = {3, 1, 4}, current = 4 → rank 3, pct = 3/3 = 1.0
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), 1.0);

    // i=3: window = {1, 4, 1}, current = 1 (tie with another 1)
    // sorted: 1,1,4. Ties at rank (1+2)/2 = 1.5. pct = 1.5/3 = 0.5
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3), 1.5 / 3.0);

    // i=4: window = {4, 1, 5}, current = 5 → rank 3, pct = 3/3 = 1.0
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 4, 3), 1.0);

    // i=6: window = {5, 9, 2}, current = 2 → rank 1, pct = 1/3
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 6, 3), 1.0 / 3.0);
}

TEST_F(RollingRankOpTest, ScalarAllSame) {
    double data[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    RollingRankOp op(4);
    // All 4 values equal → average rank for ties = (1+2+3+4)/4 = 2.5
    // pct = 2.5/4 = 0.625
    for (std::size_t i = 3; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(data, i, 4), 2.5 / 4.0);
    }
}

TEST_F(RollingRankOpTest, ScalarIncreasing) {
    // Strictly increasing: each current value gets the highest rank
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    RollingRankOp op(3);

    // i=2: window {1,2,3}, current=3 → rank 3, pct=1.0
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 2, 3), 1.0);
    // i=3: window {2,3,4}, current=4 → rank 3, pct=1.0
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 3), 1.0);
    // i=4: window {3,4,5}, current=5 → rank 3, pct=1.0
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 4, 3), 1.0);
}

TEST_F(RollingRankOpTest, ScalarDecreasing) {
    // Strictly decreasing: each current value gets the lowest rank
    double data[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    RollingRankOp op(3);

    // i=2: window {5,4,3}, current=3 → rank 1, pct=1/3
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 2, 3), 1.0 / 3.0);
    // i=3: window {4,3,2}, current=2 → rank 1, pct=1/3
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 3), 1.0 / 3.0);
    // i=4: window {3,2,1}, current=1 → rank 1, pct=1/3
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 4, 3), 1.0 / 3.0);
}

TEST_F(RollingRankOpTest, ScalarWindow1) {
    RollingRankOp op(1);
    // Single element in window → always rank 1, pct = 1.0
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), 1.0);
    }
}

TEST_F(RollingRankOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, nan, 4.0};
    RollingRankOp op(3);
    // NaN comparisons: current=NaN, all comparisons with NaN are false
    // lessCount=0, equalCount=0 → avgRank = 0.5, pct = 0.5/window
    double result = op.evaluateScalar(data, 2, 3);
    // Actually: input[2]=NaN. NaN < nan is false, NaN == NaN is true (but in IEEE754 NaN != NaN)
    // So: lessCount=0, equalCount=1 → avgRank = 0 + (1+1)/2 = 1.0, pct = 1/3
    // Wait, NaN == NaN is false in IEEE754. So equalCount=0.
    // lessCount=0, equalCount=0 → avgRank = 0 + 1/2 = 0.5, pct = 0.5/3
    EXPECT_DOUBLE_EQ(result, 0.5 / 3.0);
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingRankOpTest, BatchNoNulls) {
    RollingRankOp op(3);
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

TEST_F(RollingRankOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 100;
    RollingRankOp op(10);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i % 20) * 1.5;

    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 10);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 10);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(simdOut[i], expected) << "i=" << i;
    }
}

// ============================================================
// Statelessness
// ============================================================

TEST_F(RollingRankOpTest, MultipleInstancesIdentical) {
    RollingRankOp op1(3), op2(3);
    double data[] = {1.0, 3.0, 2.0, 5.0, 4.0};
    for (std::size_t i = 0; i < 5; ++i) {
        double v1 = op1.evaluateScalar(data, i, 3);
        double v2 = op2.evaluateScalar(data, i, 3);
        if (std::isnan(v1))
            EXPECT_TRUE(std::isnan(v2));
        else
            EXPECT_DOUBLE_EQ(v1, v2);
    }
}
