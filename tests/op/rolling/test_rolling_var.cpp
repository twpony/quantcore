// test_rolling_var.cpp — unit tests for RollingVarOp (滚动方差, ddof=0)
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RollingOperator.h"
#include "quantcore/operators/rolling/RollingVar.h"

using namespace quantcore;

class RollingVarOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        input_ = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    }

    // Helper: population variance (ddof=0) of values in [start, end]
    static double popVar(const double* data, std::size_t start, std::size_t end) {
        std::size_t count = end - start + 1;
        double sum = 0.0;
        for (std::size_t j = start; j <= end; ++j) sum += data[j];
        double mean = sum / static_cast<double>(count);
        double sumSq = 0.0;
        for (std::size_t j = start; j <= end; ++j) {
            double d = data[j] - mean;
            sumSq += d * d;
        }
        return sumSq / static_cast<double>(count);
    }

    std::vector<double> input_;
};

TEST_F(RollingVarOpTest, OpCode) {
    RollingVarOp op(3);
    EXPECT_EQ(op.opCode(), RollingOpCode::ROLLING_VAR);
}

TEST_F(RollingVarOpTest, Name) {
    RollingVarOp op(3);
    EXPECT_STREQ(op.opName(), "rolling_var");
}

TEST_F(RollingVarOpTest, WindowAccess) {
    RollingVarOp op(4);
    EXPECT_EQ(op.window(), 4u);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(RollingVarOpTest, ScalarWindow3) {
    RollingVarOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 0, 3)));
    EXPECT_TRUE(std::isnan(op.evaluateScalar(input_.data(), 1, 3)));

    double expected = popVar(input_.data(), 0, 2);
    double actual = op.evaluateScalar(input_.data(), 2, 3);
    EXPECT_NEAR(actual, expected, 1e-12);

    expected = popVar(input_.data(), 1, 3);
    actual = op.evaluateScalar(input_.data(), 3, 3);
    EXPECT_NEAR(actual, expected, 1e-12);

    expected = popVar(input_.data(), 7, 9);
    actual = op.evaluateScalar(input_.data(), 9, 3);
    EXPECT_NEAR(actual, expected, 1e-12);
}

TEST_F(RollingVarOpTest, ScalarConstant) {
    double data[] = {5.0, 5.0, 5.0, 5.0, 5.0};
    RollingVarOp op(3);
    for (std::size_t i = 2; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(data, i, 3), 0.0);
    }
}

TEST_F(RollingVarOpTest, ScalarWindow1) {
    RollingVarOp op(1);
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(op.evaluateScalar(input_.data(), i, 1), 0.0);
    }
}

TEST_F(RollingVarOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double data[] = {1.0, 2.0, nan, 4.0};
    RollingVarOp op(3);
    EXPECT_TRUE(std::isnan(op.evaluateScalar(data, 2, 3)));
}

// ============================================================
// Batch evaluate
// ============================================================

TEST_F(RollingVarOpTest, BatchNoNulls) {
    RollingVarOp op(3);
    std::vector<double> output(input_.size());
    op.evaluate(ColView<double>(input_.data(), input_.size()), output.data());

    for (std::size_t i = 0; i < input_.size(); ++i) {
        double expected = op.evaluateScalar(input_.data(), i, 3);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(output[i])) << "i=" << i;
        else
            EXPECT_NEAR(output[i], expected, 1e-12) << "i=" << i;
    }
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(RollingVarOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 200;
    RollingVarOp op(15);
    std::vector<double> input(N), simdOut(N);
    for (std::size_t i = 0; i < N; ++i) input[i] = static_cast<double>(i) * 1.1;

    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data(), 15);

    for (std::size_t i = 0; i < N; ++i) {
        double expected = op.evaluateScalar(input.data(), i, 15);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            // Running sum-of-squares vs two-pass: use wider tolerance
            EXPECT_NEAR(simdOut[i], expected, 1e-10) << "i=" << i;
    }
}
