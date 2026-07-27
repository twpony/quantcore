// test_log10.cpp — unit tests for Log10Op (base-10 logarithm)
// Reference: numpy.log10
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
#include "quantcore/operators/unary/Log10.h"

using namespace quantcore;

class Log10OpTest : public ::testing::Test {
protected:
    Log10Op op_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(Log10OpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::LOG10);
}

TEST_F(Log10OpTest, Name) {
    EXPECT_STREQ(op_.opName(), "log10");
    static_assert(Log10Op::kOpCode == UnaryOpCode::LOG10);
}

// ============================================================
// evaluateScalar — normal values
// ============================================================

TEST_F(Log10OpTest, ScalarPowersOfTen) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0), 0.0);     // log10(1)=0
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(10.0), 1.0);    // log10(10)=1
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(100.0), 2.0);   // log10(100)=2
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.1), -1.0);    // log10(0.1)=-1
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.01), -2.0);   // log10(0.01)=-2
}

// ============================================================
// evaluateScalar — domain boundary: x <= 0
// ============================================================

TEST_F(Log10OpTest, ScalarZero) {
    double r = op_.evaluateScalar(0.0);
    EXPECT_TRUE(std::isinf(r) && r < 0);  // -Inf
}

TEST_F(Log10OpTest, ScalarNegative) {
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(-1.0)));  // NaN
}

// ============================================================
// evaluateScalar — special values
// ============================================================

TEST_F(Log10OpTest, ScalarInfNaN) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(inf)) && op_.evaluateScalar(inf) > 0);
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(-inf)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(
        std::numeric_limits<double>::quiet_NaN())));
}

// ============================================================
// evaluateScalar — matches std::log10 (numpy.log10 reference)
// ============================================================

TEST_F(Log10OpTest, ScalarMatchesStdLog10) {
    double vals[] = {0.001, 0.5, 1.0, 10.0, 100.0, 1e6,
                     std::numeric_limits<double>::infinity()};
    for (double x : vals)
        EXPECT_DOUBLE_EQ(op_.evaluateScalar(x), std::log10(x)) << "x=" << x;
}

// ============================================================
// Batch evaluate without nulls
// ============================================================

TEST_F(Log10OpTest, BatchNoNulls) {
    double input[]  = {1.0, 10.0, 100.0, 0.1, 1000.0};
    double output[5] = {};
    op_.evaluate(input, output, 5, nullptr);

    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], std::log10(input[i])) << "i=" << i;
}

TEST_F(Log10OpTest, BatchWithNulls) {
    double input[]  = {10.0, 100.0, 1.0, 1000.0};
    double output[4] = {};
    uint64_t nullMask[1] = {(1ULL << 1) | (1ULL << 3)};

    op_.evaluate(input, output, 4, nullMask);
    EXPECT_DOUBLE_EQ(output[0], 1.0);  // log10(10)
    EXPECT_DOUBLE_EQ(output[1], 0.0);  // null
    EXPECT_DOUBLE_EQ(output[2], 0.0);  // log10(1)
    EXPECT_DOUBLE_EQ(output[3], 0.0);  // null
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(Log10OpTest, SimdMatchesScalar) {
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
