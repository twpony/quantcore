// test_sqrt.cpp — unit tests for SqrtOp (square root)
// Reference: numpy.sqrt
//
// Coverage:
//   Operator identity, evaluateScalar (normal/zero/negative/Inf/NaN),
//   batch evaluate without nulls, batch evaluate with null mask,
//   SIMD kernel cross-validation

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/UnaryOperator.h"
#include "quantcore/operators/unary/Sqrt.h"

using namespace quantcore;

class SqrtOpTest : public ::testing::Test {
protected:
    SqrtOp op_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(SqrtOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::SQRT);
}

TEST_F(SqrtOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "sqrt");
    static_assert(SqrtOp::kOpCode == UnaryOpCode::SQRT);
}

// ============================================================
// evaluateScalar — normal values
// ============================================================

TEST_F(SqrtOpTest, ScalarPerfectSquares) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(4.0), 2.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(9.0), 3.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(100.0), 10.0);
}

TEST_F(SqrtOpTest, ScalarIrregular) {
    EXPECT_NEAR(op_.evaluateScalar(2.0), 1.4142135623730951, 1e-14);
    EXPECT_NEAR(op_.evaluateScalar(0.25), 0.5, 1e-14);
}

// ============================================================
// evaluateScalar — domain boundary: x < 0
// ============================================================

TEST_F(SqrtOpTest, ScalarNegative) {
    // sqrt(negative) = NaN (matches numpy.sqrt)
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(-1.0)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(-100.0)));
}

// ============================================================
// evaluateScalar — zero
// ============================================================

TEST_F(SqrtOpTest, ScalarZero) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0), 0.0);
    // sqrt(-0.0) = -0.0 (IEEE 754 sign-preserving)
    double r = op_.evaluateScalar(-0.0);
    EXPECT_EQ(r, 0.0);
    EXPECT_TRUE(std::signbit(r));  // preserves negative zero sign
}

// ============================================================
// evaluateScalar — special values
// ============================================================

TEST_F(SqrtOpTest, ScalarInfNaN) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(inf)) && op_.evaluateScalar(inf) > 0);
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(-inf)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(
        std::numeric_limits<double>::quiet_NaN())));
}

// ============================================================
// evaluateScalar — matches std::sqrt (numpy.sqrt reference)
// ============================================================

TEST_F(SqrtOpTest, ScalarMatchesStdSqrt) {
    double vals[] = {0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 1e10,
                     std::numeric_limits<double>::infinity()};
    for (double x : vals) {
        EXPECT_DOUBLE_EQ(op_.evaluateScalar(x), std::sqrt(x)) << "x=" << x;
    }
}

// ============================================================
// Batch evaluate without nulls
// ============================================================

TEST_F(SqrtOpTest, BatchNoNulls) {
    double input[]  = {0.0, 1.0, 4.0, 9.0, 16.0};
    double output[5] = {};
    op_.evaluate(input, output, 5, nullptr);

    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], std::sqrt(input[i])) << "i=" << i;
}

// ============================================================
// Batch evaluate with null mask
// ============================================================

TEST_F(SqrtOpTest, BatchWithNulls) {
    double input[]  = {1.0, 4.0, 9.0, 16.0};
    double output[4] = {};
    uint64_t nullMask[1] = {(1ULL << 0) | (1ULL << 2)};

    op_.evaluate(input, output, 4, nullMask);
    EXPECT_DOUBLE_EQ(output[0], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[1], 2.0);   // sqrt(4)=2
    EXPECT_DOUBLE_EQ(output[2], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[3], 4.0);   // sqrt(16)=4
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(SqrtOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> input(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i)
        input[i] = static_cast<double>(i) * 100.0 / N;  // all >= 0

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
