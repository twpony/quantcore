// test_lt.cpp — unit tests for LtOp (element-wise less-than: x < y)
// Phase: 一期必实现
// Reference: numpy.less (as 1.0/0.0 float)
//
// Coverage:
//   LtOp: operator identity, evaluateScalar (normal/NaN/Inf/equal),
//         batch evaluate col-vs-col, col-vs-scalar, scalar-vs-col,
//         batch evaluate with null mask,
//         SIMD kernel cross-validation

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/BinaryOperator.h"
#include "quantcore/operators/binary/Lt.h"
#include "quantcore/storage/ColView.h"

using namespace quantcore;

class LtOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        lhsSmall_  = {1.0, 5.0, 3.0, -1.0, 0.0};
        rhsSmall_  = {0.0, 3.0, 5.0, -2.0, 0.0};
        // lhs < rhs ? 1.0 : 0.0
        expectedSmall_ = {0.0, 0.0, 1.0, 0.0, 0.0};
    }

    LtOp op_;
    std::vector<double> lhsSmall_;
    std::vector<double> rhsSmall_;
    std::vector<double> expectedSmall_;
};

// ============================================================
// Operator identity
// ============================================================

TEST_F(LtOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), BinaryOpCode::LT);
}

TEST_F(LtOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "lt");
    static_assert(LtOp::kOpCode == BinaryOpCode::LT);
}

// ============================================================
// evaluateScalar
// ============================================================

TEST_F(LtOpTest, ScalarLessThan) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(3.0, 5.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-1.0, 0.0), 1.0);
}

TEST_F(LtOpTest, ScalarGreaterThan) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(5.0, 3.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, -1.0), 0.0);
}

TEST_F(LtOpTest, ScalarEqual) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(5.0, 5.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-3.0, -3.0), 0.0);
}

TEST_F(LtOpTest, ScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    // NaN comparisons always return false (0.0)
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(nan, 1.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0, nan), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(nan, nan), 0.0);
}

TEST_F(LtOpTest, ScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0, inf), 1.0);      // n < Inf
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf, 1.0), 0.0);      // Inf > n
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf, inf), 0.0);      // Inf == Inf → 0
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-inf, inf), 1.0);     // -Inf < Inf
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf, -inf), 0.0);     // Inf > -Inf
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-inf, -inf), 0.0);    // -Inf == -Inf → 0
}

// ============================================================
// Batch evaluate — unified interface
// ============================================================

TEST_F(LtOpTest, BatchColVsCol) {
    const std::size_t n = lhsSmall_.size();
    std::vector<double> output(n);
    op_.evaluate(lhsSmall_.data(), rhsSmall_.data(), output.data(), n, nullptr);
    for (std::size_t i = 0; i < n; ++i)
        EXPECT_DOUBLE_EQ(output[i], expectedSmall_[i]) << "i=" << i;
}

TEST_F(LtOpTest, BatchColVsScalar) {
    double lhs[] = {1.0, 2.0, 3.0};
    double out[3] = {};
    op_.evaluate(lhs, 2.0, out, 3, nullptr);
    EXPECT_DOUBLE_EQ(out[0], 1.0);  // 1 < 2 = true
    EXPECT_DOUBLE_EQ(out[1], 0.0);  // 2 < 2 = false
    EXPECT_DOUBLE_EQ(out[2], 0.0);  // 3 < 2 = false
}

TEST_F(LtOpTest, BatchScalarVsCol) {
    double rhs[] = {1.0, 2.0, 3.0};
    double out[3] = {};
    op_.evaluate(2.0, rhs, out, 3, nullptr);
    EXPECT_DOUBLE_EQ(out[0], 0.0);  // 2 < 1 = false
    EXPECT_DOUBLE_EQ(out[1], 0.0);  // 2 < 2 = false
    EXPECT_DOUBLE_EQ(out[2], 1.0);  // 2 < 3 = true
}

TEST_F(LtOpTest, BatchScalarVsScalar) {
    double out[4] = {};
    op_.evaluate(3.0, 5.0, out, 4, nullptr);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(out[i], 1.0);
}

// ============================================================
// Null mask
// ============================================================

TEST_F(LtOpTest, BatchWithNulls) {
    double lhs[] = {5.0, 2.0, 4.0, 3.0};
    double rhs[] = {3.0, 5.0, 5.0, 5.0};
    double out[4] = {};
    uint64_t mask[1] = {(1ULL << 0) | (1ULL << 2)};

    op_.evaluate(lhs, rhs, out, 4, mask);
    EXPECT_DOUBLE_EQ(out[0], 0.0);  // null
    EXPECT_DOUBLE_EQ(out[1], 1.0);  // 2 < 5 = true
    EXPECT_DOUBLE_EQ(out[2], 0.0);  // null
    EXPECT_DOUBLE_EQ(out[3], 1.0);  // 3 < 5 = true
}

// ============================================================
// SIMD cross-validation
// ============================================================

TEST_F(LtOpTest, SimdMatchesScalar) {
    constexpr std::size_t N = 512;
    std::vector<double> lhs(N), rhs(N), simdOut(N), scalarOut(N);
    for (std::size_t i = 0; i < N; ++i) {
        lhs[i] = static_cast<double>(i) - 256.0;
        rhs[i] = 0.0;
    }
    lhs[50] = std::numeric_limits<double>::quiet_NaN();
    lhs[150] = std::numeric_limits<double>::infinity();
    lhs[250] = -std::numeric_limits<double>::infinity();

    op_.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(lhs.data(), N), ColView<double>(rhs.data(), N), simdOut.data());
    for (std::size_t i = 0; i < N; ++i)
        scalarOut[i] = op_.evaluateScalar(lhs[i], rhs[i]);

    for (std::size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(simdOut[i], scalarOut[i]) << "i=" << i;
}

// ============================================================
// Stateless
// ============================================================

TEST_F(LtOpTest, MultipleInstancesIdentical) {
    LtOp op1, op2;
    for (double a : {-5.0, 0.0, 5.0})
        for (double b : {-3.0, 1.0, 4.0})
            EXPECT_DOUBLE_EQ(op1.evaluateScalar(a, b), op2.evaluateScalar(a, b));
}
