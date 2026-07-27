// test_not.cpp — unit tests for NotOp (logical NOT)
// Reference: numpy.logical_not
//
// Coverage:
//   Operator identity, evaluateScalar (zero/non-zero/negative/NaN/Inf),
//   batch evaluate without nulls, batch evaluate with null mask,
//   SIMD kernel cross-validation

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/UnaryOperator.h"
#include "quantcore/operators/unary/Not.h"

using namespace quantcore;

class NotOpTest : public ::testing::Test {
protected:
    NotOp op_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(NotOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::NOT);
}

TEST_F(NotOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "not");
    static_assert(NotOp::kOpCode == UnaryOpCode::NOT);
}

// ============================================================
// evaluateScalar — logic
// ============================================================

TEST_F(NotOpTest, ScalarZero) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0), 1.0);   // NOT(0) = 1
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-0.0), 1.0);  // NOT(-0) = 1
}

TEST_F(NotOpTest, ScalarNonZero) {
    // Non-zero values (truthy) -> 0.0 (false)
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(3.14), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.001), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-0.001), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(100.0), 0.0);
}

// ============================================================
// evaluateScalar — special values
// ============================================================

TEST_F(NotOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf), 0.0);   // NOT(Inf) = 0
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-inf), 0.0);  // NOT(-Inf) = 0
}

TEST_F(NotOpTest, ScalarNaN) {
    // NaN != 0, so it's truthy -> NOT(NaN) = 0
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(
        std::numeric_limits<double>::quiet_NaN()), 0.0);
}

// ============================================================
// evaluateScalar — comprehensive
// ============================================================

TEST_F(NotOpTest, ScalarComprehensive) {
    double vals[] = {0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 100.0,
                     std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()};
    for (double x : vals) {
        double result = op_.evaluateScalar(x);
        double expected = (x == 0.0) ? 1.0 : 0.0;
        EXPECT_DOUBLE_EQ(result, expected) << "x=" << x;
    }
}

// ============================================================
// Batch evaluate without nulls
// ============================================================

TEST_F(NotOpTest, BatchNoNulls) {
    double input[]  = {0.0, 1.0, -0.0, 3.14, -2.0};
    double expected[] = {1.0, 0.0, 1.0, 0.0, 0.0};
    double output[5] = {};
    op_.evaluate(input, output, 5, nullptr);

    for (int i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], expected[i]) << "i=" << i;
}

TEST_F(NotOpTest, BatchWithNulls) {
    double input[]  = {0.0, 5.0, 0.0, -3.0};
    double output[4] = {};
    uint64_t nullMask[1] = {(1ULL << 1) | (1ULL << 3)};

    op_.evaluate(input, output, 4, nullMask);
    EXPECT_DOUBLE_EQ(output[0], 1.0);   // NOT(0) = 1
    EXPECT_DOUBLE_EQ(output[1], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[2], 1.0);   // NOT(0) = 1
    EXPECT_DOUBLE_EQ(output[3], 0.0);   // null
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(NotOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> input(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i)
        input[i] = (i % 3 == 0) ? 0.0 : static_cast<double>(i) - 256.0;

    op_.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data());
    for (std::size_t i = 0; i < N; ++i)
        scalarOut[i] = op_.evaluateScalar(input[i]);

    for (std::size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(simdOut[i], scalarOut[i]) << "i=" << i;
}
