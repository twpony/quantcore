// test_div.cpp — unit tests for DivOp (element-wise division)
// Phase: 一期必实现
// Reference: numpy.divide
//
// Coverage:
//   DivOp: operator identity, evaluateScalar (normal/NaN/Inf/div-by-zero),
//          batch evaluate col-vs-col, col-vs-scalar, scalar-vs-col
//          (verifying non-commutative behaviour),
//          batch evaluate with null mask,
//          SIMD kernel cross-validation

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/BinaryOperator.h"
#include "quantcore/operators/binary/Div.h"
#include "quantcore/storage/ColView.h"

using namespace quantcore;

class DivOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        lhsSmall_  = {6.0, 10.0, -8.0, 0.0, 1.0};
        rhsSmall_  = {2.0,  4.0,  2.0, 5.0, 2.0};
        expectedSmall_ = {3.0, 2.5, -4.0, 0.0, 0.5};
    }

    DivOp op_;
    std::vector<double> lhsSmall_;
    std::vector<double> rhsSmall_;
    std::vector<double> expectedSmall_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(DivOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), BinaryOpCode::DIV);
}

TEST_F(DivOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "div");
    static_assert(DivOp::kOpCode == BinaryOpCode::DIV);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(DivOpTest, ScalarBasic) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(10.0, 2.0), 5.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-8.0, 4.0), -2.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0, 3.0), 1.0 / 3.0);
}

TEST_F(DivOpTest, ScalarZeroNumerator) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, 5.0), 0.0);
}

TEST_F(DivOpTest, ScalarDivideByZero) {
    double inf = std::numeric_limits<double>::infinity();
    // x / 0 = ±Inf (x ≠ 0)
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(1.0, 0.0)));
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(-1.0, 0.0)));
    // 0 / 0 = NaN
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(0.0, 0.0)));
}

TEST_F(DivOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(nan, 1.0)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(1.0, nan)));
}

TEST_F(DivOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf, 2.0), inf);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0, inf), 0.0);
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(inf, inf)));  // Inf / Inf = NaN
}

// ============================================================
// Non-commutative: a / b ≠ b / a  (unless a == b)
// ============================================================

TEST_F(DivOpTest, ScalarNonCommutative) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(10.0, 2.0), 5.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(2.0, 10.0), 0.2);
    EXPECT_NE(op_.evaluateScalar(10.0, 2.0), op_.evaluateScalar(2.0, 10.0));
}

TEST_F(DivOpTest, ScalarVsColDiffersFromColVsScalar) {
    double col[] = {12.0, 6.0, 3.0};
    double scalar = 3.0;
    double out1[3] = {}, out2[3] = {};

    op_.evaluate(col, scalar, out1, 3, nullptr);  // [4, 2, 1]
    op_.evaluate(scalar, col, out2, 3, nullptr);  // [0.25, 0.5, 1]

    EXPECT_DOUBLE_EQ(out1[0], 4.0);
    EXPECT_DOUBLE_EQ(out1[1], 2.0);
    EXPECT_DOUBLE_EQ(out1[2], 1.0);

    EXPECT_DOUBLE_EQ(out2[0], 0.25);
    EXPECT_DOUBLE_EQ(out2[1], 0.5);
    EXPECT_DOUBLE_EQ(out2[2], 1.0);
}

// ============================================================
// Batch evaluate — unified interface
// ============================================================

TEST_F(DivOpTest, BatchColVsCol) {
    const std::size_t n = lhsSmall_.size();
    std::vector<double> output(n);
    op_.evaluate(lhsSmall_.data(), rhsSmall_.data(), output.data(), n, nullptr);
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(output[i], expectedSmall_[i]) << "i=" << i;
}

TEST_F(DivOpTest, BatchColVsScalar) {
    double lhs[] = {10.0, 20.0, 30.0};
    double out[3] = {};
    op_.evaluate(lhs, 2.0, out, 3, nullptr);
    EXPECT_DOUBLE_EQ(out[0], 5.0);
    EXPECT_DOUBLE_EQ(out[1], 10.0);
    EXPECT_DOUBLE_EQ(out[2], 15.0);
}

TEST_F(DivOpTest, BatchScalarVsCol) {
    double rhs[] = {2.0, 4.0, 8.0};
    double out[3] = {};
    op_.evaluate(16.0, rhs, out, 3, nullptr);
    EXPECT_DOUBLE_EQ(out[0], 8.0);  // 16/2
    EXPECT_DOUBLE_EQ(out[1], 4.0);  // 16/4
    EXPECT_DOUBLE_EQ(out[2], 2.0);  // 16/8
}

TEST_F(DivOpTest, BatchScalarVsScalar) {
    double out[4] = {};
    op_.evaluate(10.0, 2.0, out, 4, nullptr);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(out[i], 5.0);
}

// ============================================================
// Null mask
// ============================================================

TEST_F(DivOpTest, BatchWithNulls) {
    double lhs[] = {10.0, 20.0, 30.0, 40.0};
    double rhs[] = {2.0,  4.0,  5.0,  8.0};
    double out[4] = {};
    uint64_t mask[1] = {(1ULL << 1) | (1ULL << 3)};

    op_.evaluate(lhs, rhs, out, 4, mask);
    EXPECT_DOUBLE_EQ(out[0], 5.0);
    EXPECT_DOUBLE_EQ(out[1], 0.0);   // null
    EXPECT_DOUBLE_EQ(out[2], 6.0);
    EXPECT_DOUBLE_EQ(out[3], 0.0);   // null
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(DivOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> lhs(N), rhs(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i) {
        lhs[i] = static_cast<double>(i) * 2.0 + 1.0;
        rhs[i] = static_cast<double>(i) * 0.5 + 1.0;  // never zero
    }
    lhs[100] = 1.0; rhs[100] = 0.0;  // div-by-zero → Inf
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

TEST_F(DivOpTest, MultipleInstancesIdentical) {
    DivOp op1, op2;
    for (double a : {-5.0, 0.0, 5.0})
        for (double b : {-3.0, 1.0, 4.0})
            EXPECT_DOUBLE_EQ(op1.evaluateScalar(a, b), op2.evaluateScalar(a, b));
}
