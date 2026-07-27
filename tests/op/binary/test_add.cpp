// test_add.cpp — unit tests for AddOp (element-wise addition)
// Phase: 一期必实现
// Reference: numpy.add
//
// Coverage:
//   AddOp: operator identity, evaluateScalar (normal/NaN/Inf),
//          batch evaluate col-vs-col, col-vs-scalar, scalar-vs-col,
//          batch evaluate with null mask,
//          SIMD kernel cross-validation against scalar reference

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/BinaryOperator.h"
#include "quantcore/operators/binary/Add.h"
#include "quantcore/storage/ColView.h"

using namespace quantcore;

class AddOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        lhsSmall_  = {1.0, -2.5, 3.14, -100.0, 0.0};
        rhsSmall_  = {0.5, -1.0, 5.0,   -50.0,  2.0};
        expectedSmall_ = {1.5, -3.5, 8.14, -150.0, 2.0};
    }

    AddOp op_;
    std::vector<double> lhsSmall_;
    std::vector<double> rhsSmall_;
    std::vector<double> expectedSmall_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(AddOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), BinaryOpCode::ADD);
}

TEST_F(AddOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "add");
    static_assert(AddOp::kOpCode == BinaryOpCode::ADD);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(AddOpTest, ScalarBasic) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(2.0, 3.0), 5.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-1.0, 5.0), 4.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-3.0, -7.0), -10.0);
}

TEST_F(AddOpTest, ScalarZero) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(5.0, 0.0), 5.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-0.0, 0.0), 0.0);
}

TEST_F(AddOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(nan, 1.0)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(1.0, nan)));
}

TEST_F(AddOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(inf, 1.0)));
    EXPECT_TRUE(std::isinf(op_.evaluateScalar(inf, inf)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(inf, -inf)));  // Inf - Inf = NaN
}

TEST_F(AddOpTest, ScalarCommutative) {
    double vals[] = {-5.0, 0.0, 5.0, 3.14};
    for (double a : vals)
        for (double b : vals)
            EXPECT_DOUBLE_EQ(op_.evaluateScalar(a, b), op_.evaluateScalar(b, a));
}

// ============================================================
// Batch evaluate — unified interface
// ============================================================

TEST_F(AddOpTest, BatchColVsCol) {
    const std::size_t n = lhsSmall_.size();
    std::vector<double> output(n);
    op_.evaluate(lhsSmall_.data(), rhsSmall_.data(), output.data(), n, nullptr);
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(output[i], expectedSmall_[i]) << "i=" << i;
}

TEST_F(AddOpTest, BatchColVsScalar) {
    double lhs[] = {1.0, 2.0, 3.0};
    double out[3] = {};
    op_.evaluate(lhs, 10.0, out, 3, nullptr);
    EXPECT_DOUBLE_EQ(out[0], 11.0);
    EXPECT_DOUBLE_EQ(out[1], 12.0);
    EXPECT_DOUBLE_EQ(out[2], 13.0);
}

TEST_F(AddOpTest, BatchScalarVsCol) {
    double rhs[] = {1.0, 2.0, 3.0};
    double out[3] = {};
    op_.evaluate(10.0, rhs, out, 3, nullptr);
    EXPECT_DOUBLE_EQ(out[0], 11.0);
    EXPECT_DOUBLE_EQ(out[1], 12.0);
    EXPECT_DOUBLE_EQ(out[2], 13.0);
}

TEST_F(AddOpTest, BatchScalarVsScalar) {
    double out[4] = {};
    op_.evaluate(3.0, 7.0, out, 4, nullptr);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(out[i], 10.0);
}

// ============================================================
// Null mask
// ============================================================

TEST_F(AddOpTest, BatchWithNulls) {
    double lhs[] = {1.0, 2.0, 3.0, 4.0};
    double rhs[] = {5.0, 6.0, 7.0, 8.0};
    double out[4] = {};
    uint64_t mask[1] = {(1ULL << 0) | (1ULL << 2)};  // pos 0,2 null

    op_.evaluate(lhs, rhs, out, 4, mask);
    EXPECT_DOUBLE_EQ(out[0], 0.0);
    EXPECT_DOUBLE_EQ(out[1], 8.0);
    EXPECT_DOUBLE_EQ(out[2], 0.0);
    EXPECT_DOUBLE_EQ(out[3], 12.0);
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(AddOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> lhs(N), rhs(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i) {
        lhs[i] = static_cast<double>(i) - 256.0;
        rhs[i] = static_cast<double>(N - i) - 256.0;
    }
    // Special values
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

TEST_F(AddOpTest, MultipleInstancesIdentical) {
    AddOp op1, op2;
    for (double a : {-5.0, 0.0, 5.0})
        for (double b : {-3.0, 1.0, 4.0})
            EXPECT_DOUBLE_EQ(op1.evaluateScalar(a, b), op2.evaluateScalar(a, b));
}
