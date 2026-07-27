// test_rolling_max.cpp — unit tests for RollingMaxOp (滚动最大值)
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingMax.h"

using namespace quantcore;

class RollingMaxOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0};
    }
    std::vector<double> input_;
};

TEST_F(RollingMaxOpTest, OpCode) {
    RollingMaxOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_MAX);
}

TEST_F(RollingMaxOpTest, Name) {
    RollingMaxOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_max");
}

TEST_F(RollingMaxOpTest, WindowAccess) {
    RollingMaxOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingMaxOpTest, ScalarWindow3) {
    RollingMaxOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));
    // i=2: max(3,1,4) = 4
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 2, 3), 4.0);
    // i=3: max(1,4,1) = 4
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 3, 3), 4.0);
    // i=4: max(4,1,5) = 5
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 4, 3), 5.0);
    // i=5: max(1,5,9) = 9
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 5, 3), 9.0);
    // i=6: max(5,9,2) = 9
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 6, 3), 9.0);
    // i=9: max(6,5,3) = 6
    EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), 9, 3), 6.0);
}

TEST_F(RollingMaxOpTest, ScalarWindow1) {
    RollingMaxOp op(1);
    for (std::size_t i = 0; i < input_.size(); ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), input_[i]);
    }
}

TEST_F(RollingMaxOpTest, ScalarAllSame) {
    double data[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    RollingMaxOp op(3);
    for (std::size_t i = 2; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(data, i, 3), 5.0);
    }
}

TEST_F(RollingMaxOpTest, ScalarDecreasing) {
    double data[] = {5.0, 4.0, 3.0, 2.0, 1.0};
    RollingMaxOp op(3);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 2, 3), 5.0);  // max(5,4,3)
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 3), 4.0);  // max(4,3,2)
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 4, 3), 3.0);  // max(3,2,1)
}

TEST_F(RollingMaxOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    double data[] = {1.0, inf, 3.0, 4.0};
    RollingMaxOp op(3);
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 2, 3), inf);  // max(1,inf,3)
    EXPECT_DOUBLE_EQ(op.evaluateScalar(data, 3, 3), inf);  // max(inf,3,4)
}

TEST_F(RollingMaxOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, nan, 4.0};
    RollingMaxOp op(3);
    // max with NaN is NaN
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 2, 3)));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingMaxOpTest, BatchNoNulls) {
    RollingMaxOp op(3);
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

TEST_F(RollingMaxOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 200;
    RollingMaxOp op(10);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i % 50) * 2.0;

    // Note: NaN/Inf are tested via scalar path; SIMD uses clean data.
    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 10);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 10);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(simdOut[i], expected) << "i=" << i;
    }
}
