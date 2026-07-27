// test_neq.cpp — unit tests for NeqOp (element-wise not-equal: x != y)
// Phase: 一期必实现
// Reference: numpy.not_equal (as 1.0/0.0 float)
//
// Coverage:
//   NeqOp: operator identity, evaluateScalar (normal/NaN/Inf/equal),
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
#include "quantcore/operators/binary/Neq.h"
#include "quantcore/storage/ColView.h"

using namespace quantcore;

class NeqOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        lhsSmall_  = {1.0, 5.0, 3.0, -1.0, 0.0};
        rhsSmall_  = {0.0, 5.0, 3.0, -2.0, 0.0};
        // lhs != rhs ? 1.0 : 0.0
        expectedSmall_ = {1.0, 0.0, 0.0, 1.0, 0.0};
    }

    NeqOp op_;
    std::vector<double> lhsSmall_;
    std::vector<double> rhsSmall_;
    std::vector<double> expectedSmall_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(NeqOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), BinaryOpCode::NEQ);
}

TEST_F(NeqOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "neq");
    static_assert(NeqOp::kOpCode == BinaryOpCode::NEQ);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(NeqOpTest, ScalarNotEqual) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(5.0, 3.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, -1.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-3.0, 3.0), 1.0);
}

TEST_F(NeqOpTest, ScalarEqual) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(5.0, 5.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-3.0, -3.0), 0.0);
}

TEST_F(NeqOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    // NaN != any is always true
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(nan, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0, nan), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(nan, nan), 1.0);
}

TEST_F(NeqOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf, inf), 0.0);       // Inf == Inf → false
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-inf, -inf), 0.0);     // -Inf == -Inf → false
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf, -inf), 1.0);      // Inf != -Inf
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-inf, inf), 1.0);      // -Inf != Inf
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf, 1.0), 1.0);       // Inf != n
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0, inf), 1.0);       // n != Inf
}

TEST_F(NeqOpTest, ScalarSignedZero) {
    // +0.0 == -0.0 per IEEE 754, so != is false
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, -0.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-0.0, 0.0), 0.0);
}

// ============================================================
// Batch evaluate — unified interface
// ============================================================

TEST_F(NeqOpTest, BatchColVsCol) {
    const std::size_t n = lhsSmall_.size();
    std::vector<double> output(n);
    op_.evaluate(lhsSmall_.data(), rhsSmall_.data(), output.data(), n, nullptr);
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(output[i], expectedSmall_[i]) << "i=" << i;
}

TEST_F(NeqOpTest, BatchColVsScalar) {
    double lhs[] = {1.0, 2.0, 3.0};
    double out[3] = {};
    op_.evaluate(lhs, 2.0, out, 3, nullptr);
    EXPECT_DOUBLE_EQ(out[0], 1.0);  // 1 != 2 = true
    EXPECT_DOUBLE_EQ(out[1], 0.0);  // 2 != 2 = false
    EXPECT_DOUBLE_EQ(out[2], 1.0);  // 3 != 2 = true
}

TEST_F(NeqOpTest, BatchScalarVsCol) {
    double rhs[] = {1.0, 2.0, 3.0};
    double out[3] = {};
    op_.evaluate(2.0, rhs, out, 3, nullptr);
    EXPECT_DOUBLE_EQ(out[0], 1.0);  // 2 != 1 = true
    EXPECT_DOUBLE_EQ(out[1], 0.0);  // 2 != 2 = false
    EXPECT_DOUBLE_EQ(out[2], 1.0);  // 2 != 3 = true
}

TEST_F(NeqOpTest, BatchScalarVsScalar) {
    double out[4] = {};
    op_.evaluate(5.0, 3.0, out, 4, nullptr);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(out[i], 1.0);
}

// ============================================================
// Null mask
// ============================================================

TEST_F(NeqOpTest, BatchWithNulls) {
    double lhs[] = {5.0, 2.0, 4.0, 3.0};
    double rhs[] = {3.0, 1.0, 5.0, 3.0};
    double out[4] = {};
    uint64_t mask[1] = {(1ULL << 0) | (1ULL << 2)};

    op_.evaluate(lhs, rhs, out, 4, mask);
    EXPECT_DOUBLE_EQ(out[0], 0.0);  // null
    EXPECT_DOUBLE_EQ(out[1], 1.0);  // 2 != 1 → true
    EXPECT_DOUBLE_EQ(out[2], 0.0);  // null
    EXPECT_DOUBLE_EQ(out[3], 0.0);  // 3 != 3 → false
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(NeqOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> lhs(N), rhs(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i) {
        lhs[i] = static_cast<double>(i % 10);
        rhs[i] = static_cast<double>((i / 10) % 10);
    }
    lhs[50] = std::numeric_limits<double>::quiet_NaN();
    lhs[150] = std::numeric_limits<double>::infinity();
    lhs[250] = -std::numeric_limits<double>::infinity();
    rhs[150] = std::numeric_limits<double>::infinity();
    rhs[250] = -std::numeric_limits<double>::infinity();

    op_.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(lhs.data(), N), ColView<double>(rhs.data(), N), simdOut.data());
    for (std::size_t i = 0; i < N; ++i)
        scalarOut[i] = op_.evaluateScalar(lhs[i], rhs[i]);

    for (std::size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(simdOut[i], scalarOut[i]) << "i=" << i;
}

// ============================================================
// Stateless
// ============================================================

TEST_F(NeqOpTest, MultipleInstancesIdentical) {
    NeqOp op1, op2;
    for (double a : {-5.0, 0.0, 5.0})
        for (double b : {-3.0, 1.0, 4.0})
            EXPECT_DOUBLE_EQ(op1.evaluateScalar(a, b), op2.evaluateScalar(a, b));
}
