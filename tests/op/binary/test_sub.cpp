// test_sub.cpp — unit tests for SubOp (element-wise subtraction)
// Phase: 一期必实现
// Reference: numpy.subtract
//
// Coverage:
//   SubOp: operator identity, evaluateScalar (normal/NaN/Inf),
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
#include "quantcore/operators/binary/Sub.h"
#include "quantcore/storage/ColView.h"

using namespace quantcore;

class SubOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        lhsSmall_  = {10.0, 5.0, 3.14, 0.0, -5.0};
        rhsSmall_  = {3.0,  7.0, 1.0,  2.0, -2.0};
        expectedSmall_ = {7.0, -2.0, 2.14, -2.0, -3.0};
    }

    SubOp op_;
    std::vector<double> lhsSmall_;
    std::vector<double> rhsSmall_;
    std::vector<double> expectedSmall_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(SubOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), BinaryOpCode::SUB);
}

TEST_F(SubOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "sub");
    static_assert(SubOp::kOpCode == BinaryOpCode::SUB);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(SubOpTest, ScalarBasic) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(10.0, 3.0), 7.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(5.0, -2.0), 7.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-5.0, -3.0), -2.0);
}

TEST_F(SubOpTest, ScalarZero) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(5.0, 0.0), 5.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, 5.0), -5.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, 0.0), 0.0);
}

TEST_F(SubOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(nan, 1.0)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(1.0, nan)));
}

TEST_F(SubOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(inf, 1.0)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(inf, inf)));   // Inf - Inf = NaN
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(inf, -inf)));  // Inf - (-Inf) = Inf
}

// ============================================================
// Non-commutative: a - b ≠ b - a  (unless a == b)
// ============================================================

TEST_F(SubOpTest, ScalarNonCommutative) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(10.0, 3.0), 7.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(3.0, 10.0), -7.0);
    EXPECT_NE(op_.evaluateScalar(10.0, 3.0), op_.evaluateScalar(3.0, 10.0));
}

TEST_F(SubOpTest, ScalarVsColDiffersFromColVsScalar) {
    // col - scalar ≠ scalar - col
    double col[] = {10.0, 5.0, 0.0};
    double scalar = 3.0;
    double out1[3] = {}, out2[3] = {};

    op_.evaluate(col, scalar, out1, 3, nullptr);  // [7, 2, -3]
    op_.evaluate(scalar, col, out2, 3, nullptr);  // [-7, -2, 3]

    EXPECT_DOUBLE_EQ(out1[0], 7.0);
    EXPECT_DOUBLE_EQ(out1[1], 2.0);
    EXPECT_DOUBLE_EQ(out1[2], -3.0);

    EXPECT_DOUBLE_EQ(out2[0], -7.0);
    EXPECT_DOUBLE_EQ(out2[1], -2.0);
    EXPECT_DOUBLE_EQ(out2[2], 3.0);
}

// ============================================================
// Batch evaluate — unified interface
// ============================================================

TEST_F(SubOpTest, BatchColVsCol) {
    const std::size_t n = lhsSmall_.size();
    std::vector<double> output(n);
    op_.evaluate(lhsSmall_.data(), rhsSmall_.data(), output.data(), n, nullptr);
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(output[i], expectedSmall_[i]) << "i=" << i;
}

TEST_F(SubOpTest, BatchColVsScalar) {
    double lhs[] = {10.0, 20.0, 30.0};
    double out[3] = {};
    op_.evaluate(lhs, 3.0, out, 3, nullptr);
    EXPECT_DOUBLE_EQ(out[0], 7.0);
    EXPECT_DOUBLE_EQ(out[1], 17.0);
    EXPECT_DOUBLE_EQ(out[2], 27.0);
}

TEST_F(SubOpTest, BatchScalarVsCol) {
    double rhs[] = {1.0, 2.0, 3.0};
    double out[3] = {};
    op_.evaluate(10.0, rhs, out, 3, nullptr);
    EXPECT_DOUBLE_EQ(out[0], 9.0);   // 10 - 1
    EXPECT_DOUBLE_EQ(out[1], 8.0);   // 10 - 2
    EXPECT_DOUBLE_EQ(out[2], 7.0);   // 10 - 3
}

TEST_F(SubOpTest, BatchScalarVsScalar) {
    double out[4] = {};
    op_.evaluate(10.0, 3.0, out, 4, nullptr);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(out[i], 7.0);
}

// ============================================================
// Null mask
// ============================================================

TEST_F(SubOpTest, BatchWithNulls) {
    double lhs[] = {10.0, 20.0, 30.0, 40.0};
    double rhs[] = {1.0,  2.0,  3.0,  4.0};
    double out[4] = {};
    uint64_t mask[1] = {(1ULL << 1) | (1ULL << 3)};

    op_.evaluate(lhs, rhs, out, 4, mask);
    EXPECT_DOUBLE_EQ(out[0], 9.0);
    EXPECT_DOUBLE_EQ(out[1], 0.0);   // null
    EXPECT_DOUBLE_EQ(out[2], 27.0);
    EXPECT_DOUBLE_EQ(out[3], 0.0);   // null
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(SubOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> lhs(N), rhs(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i) {
        lhs[i] = static_cast<double>(i) * 2.0;
        rhs[i] = static_cast<double>(i) * 0.5;
    }
    lhs[100] = std::numeric_limits<double>::quiet_NaN();
    lhs[200] = std::numeric_limits<double>::infinity();

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

TEST_F(SubOpTest, MultipleInstancesIdentical) {
    SubOp op1, op2;
    for (double a : {-5.0, 0.0, 5.0})
        for (double b : {-3.0, 1.0, 4.0})
            EXPECT_DOUBLE_EQ(op1.evaluateScalar(a, b), op2.evaluateScalar(a, b));
}
