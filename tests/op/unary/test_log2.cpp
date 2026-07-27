// test_log2.cpp — unit tests for Log2Op (base-2 logarithm)
// Reference: numpy.log2
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
#include "quantcore/operators/unary/Log2.h"

using namespace quantcore;

class Log2OpTest : public ::testing::Test {
protected:
    Log2Op op_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(Log2OpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::LOG2);
}

TEST_F(Log2OpTest, Name) {
    EXPECT_STREQ(op_.opName(), "log2");
    static_assert(Log2Op::kOpCode == UnaryOpCode::LOG2);
}

// ============================================================
// evaluateScalar — normal values
// ============================================================

TEST_F(Log2OpTest, ScalarPowersOfTwo) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0), 0.0);    // log2(1) = 0
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(2.0), 1.0);    // log2(2) = 1
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(4.0), 2.0);    // log2(4) = 2
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(8.0), 3.0);    // log2(8) = 3
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.5), -1.0);   // log2(1/2) = -1
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.25), -2.0);  // log2(1/4) = -2
}

// ============================================================
// evaluateScalar — domain boundary: x <= 0
// ============================================================

TEST_F(Log2OpTest, ScalarZero) {
    double result = op_.evaluateScalar(0.0);
    EXPECT_TRUE(std::isinf(result) && result < 0);  // -Inf
}

TEST_F(Log2OpTest, ScalarNegative) {
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(-1.0)));  // NaN
}

// ============================================================
// evaluateScalar — special values
// ============================================================

TEST_F(Log2OpTest, ScalarInfNaN) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(inf)) && op_.evaluateScalar(inf) > 0);
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(-inf)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(
        std::numeric_limits<double>::quiet_NaN())));
}

// ============================================================
// evaluateScalar — matches std::log2 (numpy.log2 reference)
// ============================================================

TEST_F(Log2OpTest, ScalarMatchesStdLog2) {
    double vals[] = {0.001, 0.5, 1.0, 2.0, 10.0, 1024.0,
                     std::numeric_limits<double>::infinity()};
    for (double x : vals) {
        EXPECT_DOUBLE_EQ(op_.evaluateScalar(x), std::log2(x)) << "x=" << x;
    }
}

// ============================================================
// Batch evaluate without nulls
// ============================================================

TEST_F(Log2OpTest, BatchNoNulls) {
    double input[]  = {1.0, 2.0, 4.0, 8.0, 0.5};
    double output[5] = {};
    op_.evaluate(input, output, 5, nullptr);

    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], std::log2(input[i])) << "i=" << i;
}

// ============================================================
// Batch evaluate with null mask
// ============================================================

TEST_F(Log2OpTest, BatchWithNulls) {
    double input[]  = {2.0, 4.0, 8.0, 16.0};
    double output[4] = {};
    uint64_t nullMask[1] = {(1ULL << 1) | (1ULL << 3)};

    op_.evaluate(input, output, 4, nullMask);
    EXPECT_DOUBLE_EQ(output[0], 1.0);   // log2(2)=1
    EXPECT_DOUBLE_EQ(output[1], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[2], 3.0);   // log2(8)=3
    EXPECT_DOUBLE_EQ(output[3], 0.0);   // null
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(Log2OpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> input(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i)
        input[i] = static_cast<double>(i + 1) / 50.0 + 0.01;  // all > 0

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
