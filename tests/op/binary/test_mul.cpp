// test_mul.cpp — unit tests for MulOp (element-wise multiplication)
// Phase: 一期必实现
// Reference: numpy.multiply
//
// Coverage:
//   MulOp: operator identity, evaluateScalar (normal/NaN/Inf/zero*Inf),
//          batch evaluate col-vs-col, col-vs-scalar, scalar-vs-col,
//          batch evaluate with null mask,
//          SIMD kernel cross-validation

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/BinaryOperator.h"
#include "quantcore/operators/binary/Mul.h"
#include "quantcore/storage/ColView.h"

using namespace quantcore;

class MulOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        lhsSmall_  = {2.0, -3.0, 4.0, 0.0, -1.0};
        rhsSmall_  = {3.0,  2.0, 0.5, 5.0, -6.0};
        expectedSmall_ = {6.0, -6.0, 2.0, 0.0, 6.0};
    }

    MulOp op_;
    std::vector<double> lhsSmall_;
    std::vector<double> rhsSmall_;
    std::vector<double> expectedSmall_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(MulOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), BinaryOpCode::MUL);
}

TEST_F(MulOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "mul");
    static_assert(MulOp::kOpCode == BinaryOpCode::MUL);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(MulOpTest, ScalarBasic) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(3.0, 4.0), 12.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-2.0, 5.0), -10.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-3.0, -7.0), 21.0);
}

TEST_F(MulOpTest, ScalarZero) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(5.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, 5.0), 0.0);
}

TEST_F(MulOpTest, ScalarOne) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(5.0, 1.0), 5.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0, -3.0), -3.0);
}

TEST_F(MulOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(nan, 1.0)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(1.0, nan)));
}

TEST_F(MulOpTest, ScalarZeroTimesInf) {
    double inf = std::numeric_limits<double>::infinity();
    // 0 * Inf = NaN (IEEE 754)
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(0.0, inf)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(inf, 0.0)));
}

TEST_F(MulOpTest, ScalarInfTimesInf) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(inf, inf)));
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(inf, -inf)));
}

TEST_F(MulOpTest, ScalarCommutative) {
    double vals[] = {-5.0, 0.0, 5.0, 3.14};
    for (double a : vals)
        for (double b : vals)
            EXPECT_DOUBLE_EQ(op_.evaluateScalar(a, b), op_.evaluateScalar(b, a));
}

// ============================================================
// Batch evaluate — unified interface
// ============================================================

TEST_F(MulOpTest, BatchColVsCol) {
    const std::size_t n = lhsSmall_.size();
    std::vector<double> output(n);
    op_.evaluate(lhsSmall_.data(), rhsSmall_.data(), output.data(), n, nullptr);
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(output[i], expectedSmall_[i]) << "i=" << i;
}

TEST_F(MulOpTest, BatchColVsScalar) {
    double lhs[] = {1.0, 2.0, 3.0};
    double out[3] = {};
    op_.evaluate(lhs, 10.0, out, 3, nullptr);
    EXPECT_DOUBLE_EQ(out[0], 10.0);
    EXPECT_DOUBLE_EQ(out[1], 20.0);
    EXPECT_DOUBLE_EQ(out[2], 30.0);
}

TEST_F(MulOpTest, BatchScalarVsCol) {
    double rhs[] = {1.0, 2.0, 3.0};
    double out[3] = {};
    op_.evaluate(10.0, rhs, out, 3, nullptr);
    EXPECT_DOUBLE_EQ(out[0], 10.0);
    EXPECT_DOUBLE_EQ(out[1], 20.0);
    EXPECT_DOUBLE_EQ(out[2], 30.0);
}

TEST_F(MulOpTest, BatchScalarVsScalar) {
    double out[4] = {};
    op_.evaluate(6.0, 7.0, out, 4, nullptr);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(out[i], 42.0);
}

// ============================================================
// Null mask
// ============================================================

TEST_F(MulOpTest, BatchWithNulls) {
    double lhs[] = {2.0, 3.0, 4.0, 5.0};
    double rhs[] = {10.0, 20.0, 30.0, 40.0};
    double out[4] = {};
    uint64_t mask[1] = {(1ULL << 0) | (1ULL << 2)};

    op_.evaluate(lhs, rhs, out, 4, mask);
    EXPECT_DOUBLE_EQ(out[0], 0.0);
    EXPECT_DOUBLE_EQ(out[1], 60.0);
    EXPECT_DOUBLE_EQ(out[2], 0.0);
    EXPECT_DOUBLE_EQ(out[3], 200.0);
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(MulOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> lhs(N), rhs(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i) {
        lhs[i] = static_cast<double>(i) * 0.5 + 1.0;
        rhs[i] = static_cast<double>(i) * 0.1 - 10.0;
    }
    lhs[100] = 0.0;
    rhs[100] = std::numeric_limits<double>::infinity();  // 0*Inf = NaN
    lhs[200] = std::numeric_limits<double>::quiet_NaN();

    op_.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(lhs.data(), N), ColView<double>(rhs.data(), N), simdOut.data());
    for (std::size_t i = 0; i < N; ++i)
        scalarOut[i] = op_.evaluateScalar(lhs[i], rhs[i]);

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

TEST_F(MulOpTest, MultipleInstancesIdentical) {
    MulOp op1, op2;
    for (double a : {-5.0, 0.0, 5.0})
        for (double b : {-3.0, 1.0, 4.0})
            EXPECT_DOUBLE_EQ(op1.evaluateScalar(a, b), op2.evaluateScalar(a, b));
}
