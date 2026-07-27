// test_inv.cpp — unit tests for InvOp (reciprocal)
// Reference: numpy.reciprocal
//
// Coverage:
//   Operator identity, evaluateScalar (normal/zero/Inf/NaN),
//   batch evaluate without nulls, batch evaluate with null mask,
//   SIMD kernel cross-validation

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/UnaryOperator.h"
#include "quantcore/operators/unary/Inv.h"

using namespace quantcore;

class InvOpTest : public ::testing::Test {
protected:
    InvOp op_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(InvOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::INV);
}

TEST_F(InvOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "inv");
    static_assert(InvOp::kOpCode == UnaryOpCode::INV);
}

// ============================================================
// evaluateScalar — normal values
// ============================================================

TEST_F(InvOpTest, ScalarNormal) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(2.0), 0.5);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(4.0), 0.25);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.5), 2.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-2.0), -0.5);
}

// ============================================================
// evaluateScalar — boundary: zero -> ±Inf
// ============================================================

TEST_F(InvOpTest, ScalarZero) {
    double inf = std::numeric_limits<double>::infinity();
    double r = op_.evaluateScalar(0.0);
    EXPECT_TRUE(std::isinf(r) && r > 0);  // 1/0 = +Inf
}

TEST_F(InvOpTest, ScalarNegativeZero) {
    double r = op_.evaluateScalar(-0.0);
    EXPECT_TRUE(std::isinf(r) && r < 0);  // 1/(-0) = -Inf
}

// ============================================================
// evaluateScalar — special values
// ============================================================

TEST_F(InvOpTest, ScalarInfNaN) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf), 0.0);     // 1/Inf = 0
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-inf), -0.0);   // 1/(-Inf) = -0
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(
        std::numeric_limits<double>::quiet_NaN())));    // 1/NaN = NaN
}

// ============================================================
// evaluateScalar — matches 1/x (numpy.reciprocal reference)
// ============================================================

TEST_F(InvOpTest, ScalarMatchesReciprocal) {
    double vals[] = {-10.0, -1.0, -0.5, 0.5, 1.0, 10.0,
                     std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity()};
    for (double x : vals) {
        double result = op_.evaluateScalar(x);
        double expected = 1.0 / x;
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(result)) << "x=" << x;
        else
            EXPECT_DOUBLE_EQ(result, expected) << "x=" << x;
    }
}

// ============================================================
// Batch evaluate without nulls
// ============================================================

TEST_F(InvOpTest, BatchNoNulls) {
    double input[]  = {1.0, 2.0, 4.0, 0.5, -2.0};
    double output[5] = {};
    op_.evaluate(input, output, 5, nullptr);

    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], 1.0 / input[i]) << "i=" << i;
}

TEST_F(InvOpTest, BatchWithNulls) {
    double input[]  = {2.0, 4.0, 8.0, 1.0};
    double output[4] = {};
    uint64_t nullMask[1] = {(1ULL << 0) | (1ULL << 2)};

    op_.evaluate(input, output, 4, nullMask);
    EXPECT_DOUBLE_EQ(output[0], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[1], 0.25);  // 1/4
    EXPECT_DOUBLE_EQ(output[2], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[3], 1.0);   // 1/1
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(InvOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> input(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i)
        input[i] = static_cast<double>(i) + 1.0;  // no zeros

    op_.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOut.data());
    for (std::size_t i = 0; i < N; ++i)
        scalarOut[i] = op_.evaluateScalar(input[i]);

    for (std::size_t i = 0; i < N; ++i) {
        if (std::isnan(scalarOut[i]))
            EXPECT_TRUE(std::isnan(simdOut[i])) << "i=" << i;
        else
            EXPECT_DOUBLE_EQ(simdOut[i], scalarOut[i]) << "i=" << i;
    }
}
