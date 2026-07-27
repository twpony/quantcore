// test_log.cpp — unit tests for LogOp (natural logarithm)
// Reference: numpy.log
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
#include "quantcore/operators/unary/Log.h"

using namespace quantcore;

class LogOpTest : public ::testing::Test {
protected:
    LogOp op_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(LogOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::LOG);
}

TEST_F(LogOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "log");
    static_assert(LogOp::kOpCode == UnaryOpCode::LOG);
}

// ============================================================
// evaluateScalar — normal values
// ============================================================

TEST_F(LogOpTest, ScalarPositive) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0), 0.0);   // ln(1) = 0
    EXPECT_NEAR(op_.evaluateScalar(M_E), 1.0, 1e-14); // ln(e) = 1
}

TEST_F(LogOpTest, ScalarFractional) {
    double result = op_.evaluateScalar(0.5);
    EXPECT_NEAR(result, -0.6931471805599453, 1e-14);  // ln(0.5)
}

// ============================================================
// evaluateScalar — domain boundary: x <= 0
// ============================================================

TEST_F(LogOpTest, ScalarZero) {
    double result = op_.evaluateScalar(0.0);
    // log(0) = -Inf (matches numpy.log)
    EXPECT_TRUE(std::isinf(result) && result < 0);
}

TEST_F(LogOpTest, ScalarNegative) {
    double result = op_.evaluateScalar(-1.0);
    // log(negative) = NaN (matches numpy.log)
    EXPECT_TRUE(std::isnan(result));
}

TEST_F(LogOpTest, ScalarNegativeZero) {
    double result = op_.evaluateScalar(-0.0);
    EXPECT_TRUE(std::isinf(result) && result < 0);
}

// ============================================================
// evaluateScalar — special values
// ============================================================

TEST_F(LogOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    // log(+Inf) = +Inf
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(inf)) && op_.evaluateScalar(inf) > 0);
    // log(-Inf) = NaN
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(-inf)));
}

TEST_F(LogOpTest, ScalarNaN) {
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(
        std::numeric_limits<double>::quiet_NaN())));
}

// ============================================================
// evaluateScalar — matches std::log (numpy.log reference)
// ============================================================

TEST_F(LogOpTest, ScalarMatchesStdLog) {
    double vals[] = {0.001, 0.5, 1.0, 2.0, M_E, 10.0, 100.0,
                     std::numeric_limits<double>::infinity()};
    for (double x : vals) {
        double result = op_.evaluateScalar(x);
        double expected = std::log(x);
        if (std::isnan(expected))
            EXPECT_TRUE(std::isnan(result)) << "x=" << x;
        else
            EXPECT_DOUBLE_EQ(result, expected) << "x=" << x;
    }
}

// ============================================================
// Batch evaluate without nulls
// ============================================================

TEST_F(LogOpTest, BatchNoNulls) {
    double input[]  = {1.0, M_E, 10.0, 0.5, 100.0};
    double output[5] = {};
    op_.evaluate(input, output, 5, nullptr);

    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], std::log(input[i])) << "i=" << i;
}

// ============================================================
// Batch evaluate with null mask
// ============================================================

TEST_F(LogOpTest, BatchWithNulls) {
    double input[]  = {2.0, 0.0, 8.0, 1.0};
    double output[4] = {};
    uint64_t nullMask[1] = {(1ULL << 0) | (1ULL << 2)};  // pos 0,2 null

    op_.evaluate(input, output, 4, nullMask);
    EXPECT_DOUBLE_EQ(output[0], 0.0);  // null
    EXPECT_DOUBLE_EQ(output[1], std::log(0.0));  // -Inf (not null — caller handles)
    EXPECT_DOUBLE_EQ(output[2], 0.0);  // null
    EXPECT_DOUBLE_EQ(output[3], 0.0);  // ln(1)=0
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(LogOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> input(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i)
        input[i] = static_cast<double>(i + 1) / 50.0;  // all > 0

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
