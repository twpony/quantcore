// test_min.cpp — unit tests for MinOp (element-wise minimum)
// Phase: 一期必实现
// Reference: numpy.minimum
//
// Coverage:
//   MinOp: operator identity, evaluateScalar (normal/NaN/Inf),
//          batch evaluate col-vs-col, col-vs-scalar, scalar-vs-col,
//          batch evaluate with null mask,
//          SIMD kernel cross-validation against scalar reference

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/BinaryOperator.h"
#include "quantcore/operators/binary/Min.h"
#include "quantcore/storage/ColView.h"

using namespace quantcore;

// ============================================================
// Test fixture
// ============================================================

class MinOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        lhsSmall_  = {1.0, -2.5, 3.14, -100.0, 0.0};
        rhsSmall_  = {0.5, -1.0, 5.0,   -50.0,  2.0};
        // min(lhs, rhs) element-wise
        expectedSmall_ = {0.5, -2.5, 3.14, -100.0, 0.0};
    }

    MinOp op_;

    std::vector<double> lhsSmall_;
    std::vector<double> rhsSmall_;
    std::vector<double> expectedSmall_;
};

// ============================================================
// MinOp — Operator identity
// ============================================================

TEST_F(MinOpTest, OpCodeIsMin) {
    EXPECT_EQ(op_.opCode(), BinaryOpCode::MIN);
}

TEST_F(MinOpTest, NameIsMin) {
    EXPECT_STREQ(op_.opName(), "min");
}

TEST_F(MinOpTest, CompileTimeConstants) {
    static_assert(MinOp::kOpCode == BinaryOpCode::MIN);
    EXPECT_STREQ(MinOp::name, "min");
}

// ============================================================
// MinOp — evaluateScalar: normal values
// ============================================================

TEST_F(MinOpTest, EvaluateScalarBasic) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(3.0, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0, 5.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(2.0, 2.0), 2.0);  // equal
}

TEST_F(MinOpTest, EvaluateScalarNegative) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-1.0, -5.0), -5.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-10.0, -3.0), -10.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-7.0, -7.0), -7.0);
}

TEST_F(MinOpTest, EvaluateScalarMixedSign) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0, -5.0), -5.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-3.0, 2.0), -3.0);
}

TEST_F(MinOpTest, EvaluateScalarZero) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, -1.0), -1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-0.0, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0, 1.0), 0.0);
}

// ============================================================
// MinOp — evaluateScalar: special values (NaN, Inf)
// ============================================================

TEST_F(MinOpTest, EvaluateScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    // NaN propagates: std::min(NaN, x) = NaN, std::min(x, NaN) = NaN
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(nan, 1.0)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(1.0, nan)));
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(nan, nan)));
}

TEST_F(MinOpTest, EvaluateScalarInf) {
    double inf = std::numeric_limits<double>::infinity();
    // Finite values are smaller than +Inf
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0, inf), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf, inf), inf);
    // -Inf dominates min
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(inf, -inf), -inf);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-inf, inf), -inf);
}

TEST_F(MinOpTest, EvaluateScalarNegativeInf) {
    double negInf = -std::numeric_limits<double>::infinity();
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(negInf, 1.0), negInf);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0, negInf), negInf);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(negInf, negInf), negInf);
}

// ============================================================
// MinOp — evaluateScalar: consistency with std::min
// ============================================================

TEST_F(MinOpTest, EvaluateScalarMatchesStdMin) {
    double testVals[] = {
        -1e10, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1e10,
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(),
    };

    for (double a : testVals) {
        for (double b : testVals) {
            double result = op_.evaluateScalar(a, b);
            // Our operator propagates NaN (unlike std::min)
            if (std::isnan(a) || std::isnan(b)) {
                EXPECT_TRUE(std::isnan(result)) << "a=" << a << " b=" << b;
            } else {
                EXPECT_DOUBLE_EQ(result, std::min(a, b)) << "a=" << a << " b=" << b;
            }
        }
    }
}

// ============================================================
// MinOp — Batch evaluate: Column vs Column (no nulls)
// ============================================================

TEST_F(MinOpTest, BatchEvaluateColumnVsColumn) {
    const std::size_t n = lhsSmall_.size();
    std::vector<double> output(n);

    op_.evaluate(lhsSmall_.data(), rhsSmall_.data(), output.data(), n, nullptr);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(output[i], expectedSmall_[i]) << "i=" << i;
    }
}

TEST_F(MinOpTest, BatchEvaluateLargeArray) {
    constexpr std::size_t N = 10000;
    std::vector<double> lhs(N), rhs(N), output(N);

    for (std::size_t i = 0; i < N; ++i) {
        lhs[i] = static_cast<double>(i);
        rhs[i] = static_cast<double>(N - i);
    }

    op_.evaluate(lhs.data(), rhs.data(), output.data(), N, nullptr);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(output[i], std::min(lhs[i], rhs[i])) << "i=" << i;
    }
}

// ============================================================
// MinOp — Batch evaluate: Column vs Scalar
// ============================================================

TEST_F(MinOpTest, BatchEvaluateColumnVsScalar) {
    constexpr std::size_t N = 5;
    double lhs[N] = {-3.0, 0.0, 5.0, -1.0, 2.0};
    double output[N] = {};
    double scalar = 1.0;

    op_.evaluate(lhs, scalar, output, N, nullptr);

    // min(lhs[i], 1.0)
    EXPECT_DOUBLE_EQ(output[0], -3.0);  // min(-3, 1)
    EXPECT_DOUBLE_EQ(output[1], 0.0);   // min(0, 1)
    EXPECT_DOUBLE_EQ(output[2], 1.0);   // min(5, 1)
    EXPECT_DOUBLE_EQ(output[3], -1.0);  // min(-1, 1)
    EXPECT_DOUBLE_EQ(output[4], 1.0);   // min(2, 1)
}

// ============================================================
// MinOp — Batch evaluate: Scalar vs Column
// ============================================================

TEST_F(MinOpTest, BatchEvaluateScalarVsColumn) {
    constexpr std::size_t N = 5;
    double rhs[N] = {-3.0, 0.0, 5.0, -1.0, 2.0};
    double output[N] = {};
    double scalar = 1.0;

    op_.evaluate(scalar, rhs, output, N, nullptr);

    // min(1.0, rhs[i]) — symmetric, same result
    EXPECT_DOUBLE_EQ(output[0], -3.0);
    EXPECT_DOUBLE_EQ(output[1], 0.0);
    EXPECT_DOUBLE_EQ(output[2], 1.0);
    EXPECT_DOUBLE_EQ(output[3], -1.0);
    EXPECT_DOUBLE_EQ(output[4], 1.0);
}

// ============================================================
// MinOp — Batch evaluate with null mask
// ============================================================

TEST_F(MinOpTest, BatchEvaluateWithNulls) {
    constexpr std::size_t N = 8;
    double lhs[N] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    double rhs[N] = {8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
    double output[N] = {};

    // Mark positions 1, 3, 5 as null
    uint64_t nullMask[1] = {
        (uint64_t{1} << 1) | (uint64_t{1} << 3) | (uint64_t{1} << 5)
    };

    op_.evaluate(lhs, rhs, output, N, nullMask);

    EXPECT_DOUBLE_EQ(output[0], 1.0);  // min(1, 8) non-null
    EXPECT_DOUBLE_EQ(output[1], 0.0);  // null
    EXPECT_DOUBLE_EQ(output[2], 3.0);  // min(3, 6) non-null
    EXPECT_DOUBLE_EQ(output[3], 0.0);  // null
    EXPECT_DOUBLE_EQ(output[4], 4.0);  // min(5, 4) non-null
    EXPECT_DOUBLE_EQ(output[5], 0.0);  // null
    EXPECT_DOUBLE_EQ(output[6], 2.0);  // min(7, 2) non-null
    EXPECT_DOUBLE_EQ(output[7], 1.0);  // min(8, 1) non-null
}

TEST_F(MinOpTest, BatchEvaluateAllNull) {
    constexpr std::size_t N = 4;
    double lhs[N] = {1.0, 2.0, 3.0, 4.0};
    double rhs[N] = {5.0, 6.0, 7.0, 8.0};
    double output[N] = {};

    uint64_t nullMask[1] = {0xF};  // all 4 positions null

    op_.evaluate(lhs, rhs, output, N, nullMask);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(output[i], 0.0) << "i=" << i;
    }
}

TEST_F(MinOpTest, BatchEvaluateScalarRhsWithNulls) {
    constexpr std::size_t N = 4;
    double lhs[N] = {-5.0, 3.0, -2.0, 10.0};
    double output[N] = {};
    double scalar = 0.0;

    // Mark position 2 as null
    uint64_t nullMask[1] = {uint64_t{1} << 2};

    op_.evaluate(lhs, scalar, output, N, nullMask);

    EXPECT_DOUBLE_EQ(output[0], -5.0);  // min(-5, 0) non-null
    EXPECT_DOUBLE_EQ(output[1], 0.0);   // min(3, 0) non-null
    EXPECT_DOUBLE_EQ(output[2], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[3], 0.0);   // min(10, 0) non-null
}

// ============================================================
// MinOp — SIMD kernel cross-validation
// ============================================================

TEST_F(MinOpTest, SimdMatchesScalarReference) {
    constexpr std::size_t N = 1024;
    std::vector<double> lhs(N), rhs(N);
    std::vector<double> simdOutput(N), scalarOutput(N);

    for (std::size_t i = 0; i < N; ++i) {
        double a = static_cast<double>(i) * 1.5 - 500.0;
        double b = static_cast<double>(N - i) * 1.3 - 400.0;
        // Special values at known positions
        if (i == 100) { a = 0.0; b = -0.0; }
        if (i == 200) { a = std::numeric_limits<double>::quiet_NaN(); }
        if (i == 300) { a = std::numeric_limits<double>::infinity(); }
        if (i == 400) { a = -std::numeric_limits<double>::infinity(); }
        if (i == 500) { b = std::numeric_limits<double>::quiet_NaN(); }
        lhs[i] = a;
        rhs[i] = b;
    }

    // SIMD kernel
    op_.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(lhs.data(), N), ColView<double>(rhs.data(), N),
                                         simdOutput.data());

    // Scalar reference
    for (std::size_t i = 0; i < N; ++i) {
        scalarOutput[i] = op_.evaluateScalar(lhs[i], rhs[i]);
    }

    // Cross-validate
    for (std::size_t i = 0; i < N; ++i) {
        if (std::isnan(scalarOutput[i])) {
            EXPECT_TRUE(std::isnan(simdOutput[i])) << "Expected NaN at i=" << i;
        } else {
            EXPECT_DOUBLE_EQ(simdOutput[i], scalarOutput[i]) << "i=" << i;
        }
    }
}

TEST_F(MinOpTest, SimdSmallArray) {
    double lhs[]     = {3.0, -4.0, 0.0, 5.0};
    double rhs[]     = {1.0, -2.0, 2.0, 5.0};
    double output[4] = {};

    op_.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(lhs, 4), ColView<double>(rhs, 4), output);

    EXPECT_DOUBLE_EQ(output[0], 1.0);   // min(3, 1)
    EXPECT_DOUBLE_EQ(output[1], -4.0);  // min(-4, -2)
    EXPECT_DOUBLE_EQ(output[2], 0.0);   // min(0, 2)
    EXPECT_DOUBLE_EQ(output[3], 5.0);   // min(5, 5)
}

// ============================================================
// MinOp — CRTP dispatch verification
// ============================================================

TEST_F(MinOpTest, CrtpDispatchMatchesDirectCall) {
    constexpr std::size_t N = 256;
    std::vector<double> lhs(N), rhs(N), output(N);

    for (std::size_t i = 0; i < N; ++i) {
        lhs[i] = static_cast<double>(i) - 128.0;
        rhs[i] = static_cast<double>(N - i) - 128.0;
    }

    op_.evaluate(lhs.data(), rhs.data(), output.data(), N, nullptr);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(output[i], op_.evaluateScalar(lhs[i], rhs[i])) << "i=" << i;
    }
}

// ============================================================
// MinOp — Stateless (multiple instances produce same results)
// ============================================================

TEST_F(MinOpTest, MultipleInstancesIdentical) {
    MinOp op1, op2;

    double vals[] = {-5.0, 0.0, 5.0, -3.14, 1.0};
    for (double a : vals) {
        for (double b : vals) {
            EXPECT_DOUBLE_EQ(op1.evaluateScalar(a, b), op2.evaluateScalar(a, b))
                << "a=" << a << " b=" << b;
        }
    }
}
