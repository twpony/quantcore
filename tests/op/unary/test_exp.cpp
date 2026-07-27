// test_exp.cpp — unit tests for ExpOp
// Reference: numpy.exp
//
// Coverage:
//   Operator identity, evaluateScalar (normal/zero/NaN/Inf),
//   batch evaluate without nulls, batch evaluate with null mask,
//   SIMD kernel cross-validation against scalar reference

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/UnaryOperator.h"
#include "quantcore/operators/unary/Exp.h"

using namespace quantcore;

class ExpOpTest : public ::testing::Test {
protected:
    ExpOp op_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(ExpOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::EXP);
}

TEST_F(ExpOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "exp");
    static_assert(ExpOp::kOpCode == UnaryOpCode::EXP);
}

// ============================================================
// evaluateScalar — normal values
// ============================================================

TEST_F(ExpOpTest, ScalarPositive) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0), 1.0);   // e^0 = 1
    EXPECT_NEAR(op_.evaluateScalar(1.0), M_E, 1e-14);
}

TEST_F(ExpOpTest, ScalarNegative) {
    EXPECT_NEAR(op_.evaluateScalar(-1.0), 1.0 / M_E, 1e-14);
}

TEST_F(ExpOpTest, ScalarSpecialValues) {
    double inf = std::numeric_limits<double>::infinity();
    double nan = std::numeric_limits<double>::quiet_NaN();

    // exp(-Inf) = 0
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-inf), 0.0);
    // exp(+Inf) = +Inf
    double r = op_.evaluateScalar(inf);
    EXPECT_TRUE(std::isinf(r) && r > 0);
    // exp(NaN) = NaN
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(nan)));
}

// ============================================================
// evaluateScalar — matches std::exp (numpy.exp reference)
// ============================================================

TEST_F(ExpOpTest, ScalarMatchesStdExp) {
    double vals[] = {-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0, 100.0,
                     -std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()};
    for (double x : vals) {
        double result = op_.evaluateScalar(x);
        double expected = std::exp(x);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(result)) << "x=" << x;
        else
            EXPECT_DOUBLE_EQ(result, expected) << "x=" << x;
    }
}

// ============================================================
// Batch evaluate without nulls
// ============================================================

TEST_F(ExpOpTest, BatchNoNulls) {
    double input[]  = {0.0, 1.0, -1.0, 2.0, -2.0};
    double output[5] = {};
    op_.evaluate(input, output, 5, nullptr);

    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], std::exp(input[i])) << "i=" << i;
}

// ============================================================
// Batch evaluate with null mask
// ============================================================

TEST_F(ExpOpTest, BatchWithNulls) {
    double input[]  = {1.0, 0.0, -1.0, 2.0};
    double output[4] = {};
    uint64_t nullMask[1] = {(1ULL << 1) | (1ULL << 3)};  // pos 1,3 null

    op_.evaluate(input, output, 4, nullMask);
    EXPECT_DOUBLE_EQ(output[0], std::exp(1.0));
    EXPECT_DOUBLE_EQ(output[1], 0.0);  // null
    EXPECT_DOUBLE_EQ(output[2], std::exp(-1.0));
    EXPECT_DOUBLE_EQ(output[3], 0.0);  // null
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(ExpOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> input(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i)
        input[i] = (static_cast<double>(i) - 256.0) / 50.0;

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
