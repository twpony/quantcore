// test_sign.cpp — unit tests for SignOp (sign function)
// Reference: numpy.sign
//
// Coverage:
//   Operator identity, evaluateScalar (positive/negative/zero/NaN/Inf),
//   batch evaluate without nulls, batch evaluate with null mask,
//   SIMD kernel cross-validation

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/UnaryOperator.h"
#include "quantcore/operators/unary/Sign.h"

using namespace quantcore;

class SignOpTest : public ::testing::Test {
protected:
    SignOp op_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(SignOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::SIGN);
}

TEST_F(SignOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "sign");
    static_assert(SignOp::kOpCode == UnaryOpCode::SIGN);
}

// ============================================================
// evaluateScalar — positive, negative, zero
// ============================================================

TEST_F(SignOpTest, ScalarPositive) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.001), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(100.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(3.14), 1.0);
}

TEST_F(SignOpTest, ScalarNegative) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-1.0), -1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-0.001), -1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-100.0), -1.0);
}

TEST_F(SignOpTest, ScalarZero) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-0.0), 0.0);
}

// ============================================================
// evaluateScalar — special values
// ============================================================

TEST_F(SignOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf), 1.0);   // sign(+Inf) = 1
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-inf), -1.0); // sign(-Inf) = -1
}

TEST_F(SignOpTest, ScalarNaN) {
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(
        std::numeric_limits<double>::quiet_NaN())));  // sign(NaN) = NaN
}

// ============================================================
// evaluateScalar — matches numpy.sign
// ============================================================

TEST_F(SignOpTest, ScalarComprehensive) {
    struct { double input; double expected; } cases[] = {
        {5.0, 1.0}, {-5.0, -1.0}, {0.0, 0.0}, {-0.0, 0.0},
        {1e-10, 1.0}, {-1e-10, -1.0},
        {1e308, 1.0}, {-1e308, -1.0},
        {1e-307, 1.0},
    };
    for (auto& c : cases)
        EXPECT_DOUBLE_EQ(op_.evaluateScalar(c.input), c.expected)
            << "input=" << c.input;
}

// ============================================================
// Batch evaluate without nulls
// ============================================================

TEST_F(SignOpTest, BatchNoNulls) {
    double input[]  = {3.0, -4.0, 0.0, -0.0, 42.0};
    double expected[] = {1.0, -1.0, 0.0, 0.0, 1.0};
    double output[5] = {};
    op_.evaluate(input, output, 5, nullptr);

    for (int i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], expected[i]) << "i=" << i;
}

TEST_F(SignOpTest, BatchWithNulls) {
    double input[]  = {5.0, -3.0, 0.0, -7.0};
    double output[4] = {};
    uint64_t nullMask[1] = {(1ULL << 0) | (1ULL << 3)};

    op_.evaluate(input, output, 4, nullMask);
    EXPECT_DOUBLE_EQ(output[0], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[1], -1.0);  // sign(-3)
    EXPECT_DOUBLE_EQ(output[2], 0.0);   // sign(0)
    EXPECT_DOUBLE_EQ(output[3], 0.0);   // null
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(SignOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> input(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i)
        input[i] = static_cast<double>(i) - 256.0;  // mix of signs

    input[100] = 0.0;
    input[200] = -0.0;
    input[300] = std::numeric_limits<double>::quiet_NaN();

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
