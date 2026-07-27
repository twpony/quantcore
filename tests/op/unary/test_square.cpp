// test_square.cpp — unit tests for SquareOp (element-wise square)
// Reference: numpy.square
//
// Coverage:
//   Operator identity, evaluateScalar (normal/zero/Inf/NaN/overflow),
//   batch evaluate without nulls, batch evaluate with null mask,
//   SIMD kernel cross-validation

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/UnaryOperator.h"
#include "quantcore/operators/unary/Square.h"

using namespace quantcore;

class SquareOpTest : public ::testing::Test {
protected:
    SquareOp op_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(SquareOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::SQUARE);
}

TEST_F(SquareOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "square");
    static_assert(SquareOp::kOpCode == UnaryOpCode::SQUARE);
}

// ============================================================
// evaluateScalar — normal values
// ============================================================

TEST_F(SquareOpTest, ScalarPositive) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(2.0), 4.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(10.0), 100.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.5), 0.25);
}

TEST_F(SquareOpTest, ScalarNegative) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-1.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-2.0), 4.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-10.0), 100.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-0.5), 0.25);
}

// ============================================================
// evaluateScalar — zero
// ============================================================

TEST_F(SquareOpTest, ScalarZero) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-0.0), 0.0);
    EXPECT_FALSE(std::signbit(op_.evaluateScalar(-0.0)));  // (-0)^2 = +0
}

// ============================================================
// evaluateScalar — special values
// ============================================================

TEST_F(SquareOpTest, ScalarInfNaN) {
    double inf = std::numeric_limits<double>::infinity();
    // (±Inf)^2 = +Inf
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(inf)) && op_.evaluateScalar(inf) > 0);
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(-inf)) && op_.evaluateScalar(-inf) > 0);
    // NaN^2 = NaN
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(
        std::numeric_limits<double>::quiet_NaN())));
}

// ============================================================
// evaluateScalar — matches x*x (numpy.square reference)
// ============================================================

TEST_F(SquareOpTest, ScalarMatchesXSquared) {
    double vals[] = {-100.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 100.0,
                     std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::min(),
                     std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()};
    for (double x : vals) {
        double result = op_.evaluateScalar(x);
        double expected = x * x;
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(result)) << "x=" << x;
        else
            EXPECT_DOUBLE_EQ(result, expected) << "x=" << x;
    }
}

// ============================================================
// Batch evaluate without nulls
// ============================================================

TEST_F(SquareOpTest, BatchNoNulls) {
    double input[]  = {0.0, 1.0, -2.0, 3.0, -4.0};
    double output[5] = {};
    op_.evaluate(input, output, 5, nullptr);

    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], input[i] * input[i]) << "i=" << i;
}

TEST_F(SquareOpTest, BatchLargeArray) {
    constexpr std::size_t N = 10000;
    std::vector<double> input(N), output(N);
    for (std::size_t i = 0; i < N; ++i)
        input[i] = static_cast<double>(i) - 5000.0;

    op_.evaluate(input.data(), output.data(), N, nullptr);
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(output[i], input[i] * input[i]) << "i=" << i;
}

// ============================================================
// Batch evaluate with null mask
// ============================================================

TEST_F(SquareOpTest, BatchWithNulls) {
    double input[]  = {-3.0, 2.0, -1.0, 0.0};
    double output[4] = {};
    uint64_t nullMask[1] = {(1ULL << 0) | (1ULL << 2)};

    op_.evaluate(input, output, 4, nullMask);
    EXPECT_DOUBLE_EQ(output[0], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[1], 4.0);   // 2^2=4
    EXPECT_DOUBLE_EQ(output[2], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[3], 0.0);   // 0^2=0
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(SquareOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> input(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i)
        input[i] = static_cast<double>(i) - 256.0;

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

// ============================================================
// Stateless
// ============================================================

TEST_F(SquareOpTest, MultipleInstancesIdentical) {
    SquareOp a, b;
    for (double x : {-5.0, 0.0, 3.14})
        EXPECT_DOUBLE_EQ(a.evaluateScalar(x), b.evaluateScalar(x));
}
