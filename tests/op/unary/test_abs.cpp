// test_unary_ops.cpp — unit tests for unary element-wise operators
// Phase: 一期必实现
//
// Coverage:
//   AbsOp: operator identity, evaluateScalar (normal/zero/NaN/Inf),
//          batch evaluate without nulls, batch evaluate with null mask,
//          SIMD kernel cross-validation against scalar reference

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/UnaryOperator.h"
#include "quantcore/operators/unary/Abs.h"

using namespace quantcore;

// ============================================================
// Test fixture: provides pre-built test data
// ============================================================

class AbsOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Mixed-sign input data of varying lengths
        inputSmall_  = {1.0, -2.5, 3.14, -100.0, 0.0};
        expectedSmall_ = {1.0, 2.5, 3.14, 100.0, 0.0};
    }

    AbsOp op_;

    std::vector<double> inputSmall_;
    std::vector<double> expectedSmall_;
};

// ============================================================
// AbsOp — Operator identity
// ============================================================

TEST_F(AbsOpTest, OpCodeIsAbs) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::ABS);
    EXPECT_EQ(static_cast<int>(op_.opCode()), static_cast<int>(UnaryOpCode::ABS));
}

TEST_F(AbsOpTest, NameIsAbs) {
    EXPECT_STREQ(op_.opName(), "abs");
}

// ============================================================
// AbsOp — evaluateScalar: normal values
// ============================================================

TEST_F(AbsOpTest, EvaluateScalarPositive) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(3.14), 3.14);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(1000.5), 1000.5);
}

TEST_F(AbsOpTest, EvaluateScalarNegative) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-3.14), 3.14);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-1.0), 1.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-1000.5), 1000.5);
}

TEST_F(AbsOpTest, EvaluateScalarZero) {
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(0.0), 0.0);
    EXPECT_DOUBLE_EQ(op_.evaluateScalar(-0.0), 0.0);
    // Sign of zero: std::abs(-0.0) returns 0.0 (positive zero per IEEE 754)
    EXPECT_EQ(std::signbit(op_.evaluateScalar(-0.0)), false);
}

// ============================================================
// AbsOp — evaluateScalar: special values (NaN, Inf)
// ============================================================

TEST_F(AbsOpTest, EvaluateScalarNaN) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    double result = op_.evaluateScalar(nan);
    EXPECT_TRUE(std::isnan(result));
}

TEST_F(AbsOpTest, EvaluateScalarPositiveInf) {
    double inf = std::numeric_limits<double>::infinity();
    double result = op_.evaluateScalar(inf);
    EXPECT_TRUE(std::isinf(result));
    EXPECT_GT(result, 0.0);
}

TEST_F(AbsOpTest, EvaluateScalarNegativeInf) {
    double negInf = -std::numeric_limits<double>::infinity();
    double result = op_.evaluateScalar(negInf);
    EXPECT_TRUE(std::isinf(result));
    EXPECT_GT(result, 0.0);  // abs(-Inf) = +Inf
}

// ============================================================
// AbsOp — evaluateScalar: consistency with std::abs
// ============================================================

TEST_F(AbsOpTest, EvaluateScalarMatchesStdAbs) {
    // Verify that evaluateScalar produces the same result as std::abs
    // across a range of values, matching the numpy.abs reference.
    double testValues[] = {
        -1e10, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1e10,
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::epsilon(),
        -std::numeric_limits<double>::epsilon(),
    };

    for (double x : testValues) {
        double result = op_.evaluateScalar(x);
        double expected = std::abs(x);
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(result)) << "x=" << x;
        } else {
            EXPECT_DOUBLE_EQ(result, expected) << "x=" << x;
        }
    }
}

// ============================================================
// AbsOp — Batch evaluate without nulls
// ============================================================

TEST_F(AbsOpTest, BatchEvaluateNoNulls) {
    const std::size_t n = inputSmall_.size();
    std::vector<double> output(n);

    op_.evaluate(inputSmall_.data(), output.data(), n, nullptr);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(output[i], expectedSmall_[i]) << "i=" << i;
    }
}

TEST_F(AbsOpTest, BatchEvaluateLargeArray) {
    constexpr std::size_t N = 10000;
    std::vector<double> input(N);
    std::vector<double> output(N);

    for (std::size_t i = 0; i < N; ++i) {
        input[i] = (i % 2 == 0) ? static_cast<double>(i) : -static_cast<double>(i);
    }

    op_.evaluate(input.data(), output.data(), N, nullptr);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(output[i], std::abs(input[i])) << "i=" << i;
    }
}

TEST_F(AbsOpTest, BatchEvaluateAllPositive) {
    std::vector<double> input  = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> output(5);

    op_.evaluate(input.data(), output.data(), 5, nullptr);

    // All positive: output should equal input
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(output[i], input[i]) << "i=" << i;
    }
}

TEST_F(AbsOpTest, BatchEvaluateAllNegative) {
    std::vector<double> input  = {-1.0, -2.0, -3.0, -4.0, -5.0};
    std::vector<double> output(5);

    op_.evaluate(input.data(), output.data(), 5, nullptr);

    // All negative: output should be the negation of input
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(output[i], -input[i]) << "i=" << i;
    }
}

TEST_F(AbsOpTest, BatchEvaluateSingleElement) {
    double input  = -42.0;
    double output = 0.0;

    op_.evaluate(&input, &output, 1, nullptr);

    EXPECT_DOUBLE_EQ(output, 42.0);
}

// ============================================================
// AbsOp — Batch evaluate with null mask
// ============================================================

TEST_F(AbsOpTest, BatchEvaluateWithNulls) {
    constexpr std::size_t N = 8;
    double input[N]  = {1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0};
    double output[N] = {};

    // Mark positions 1, 3, 5 as null
    // nullMask[i/64] >> (i%64) & 1 => bit 1, 3, 5 of first uint64
    uint64_t nullMask[1] = { (uint64_t{1} << 1) | (uint64_t{1} << 3) | (uint64_t{1} << 5) };

    op_.evaluate(input, output, N, nullMask);

    // Non-null positions: should have correct abs values
    EXPECT_DOUBLE_EQ(output[0], 1.0);   // non-null
    EXPECT_DOUBLE_EQ(output[1], 0.0);   // null → 0.0
    EXPECT_DOUBLE_EQ(output[2], 3.0);   // non-null
    EXPECT_DOUBLE_EQ(output[3], 0.0);   // null → 0.0
    EXPECT_DOUBLE_EQ(output[4], 5.0);   // non-null
    EXPECT_DOUBLE_EQ(output[5], 0.0);   // null → 0.0
    EXPECT_DOUBLE_EQ(output[6], 7.0);   // non-null
    EXPECT_DOUBLE_EQ(output[7], 8.0);   // non-null
}

TEST_F(AbsOpTest, BatchEvaluateAllNull) {
    constexpr std::size_t N = 4;
    double input[N]  = {-1.0, -2.0, -3.0, -4.0};
    double output[N] = {};

    // Mark all positions as null
    uint64_t nullMask[1] = {
        (uint64_t{1} << 0) | (uint64_t{1} << 1) |
        (uint64_t{1} << 2) | (uint64_t{1} << 3)
    };

    op_.evaluate(input, output, N, nullMask);

    // All positions null → all output 0.0
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(output[i], 0.0) << "i=" << i;
    }
}

TEST_F(AbsOpTest, NullMaskAcrossWordBoundary) {
    constexpr std::size_t N = 128;
    std::vector<double> input(N);
    std::vector<double> output(N);

    // Fill with alternating positive/negative
    for (std::size_t i = 0; i < N; ++i) {
        input[i] = (i % 2 == 0) ? static_cast<double>(i) : -static_cast<double>(i);
    }

    // Mark positions 63, 64, 65 as null (crosses uint64 word boundary)
    uint64_t nullMask[2] = {};
    nullMask[0] |= (uint64_t{1} << 63);  // bit 63 in first word
    nullMask[1] |= (uint64_t{1} << 0);   // bit 64 → bit 0 of second word
    nullMask[1] |= (uint64_t{1} << 1);   // bit 65 → bit 1 of second word

    op_.evaluate(input.data(), output.data(), N, nullMask);

    // Non-null positions should have correct abs values
    for (std::size_t i = 0; i < N; ++i) {
        if (i == 63 || i == 64 || i == 65) {
            EXPECT_DOUBLE_EQ(output[i], 0.0) << "null position i=" << i;
        } else {
            EXPECT_DOUBLE_EQ(output[i], std::abs(input[i])) << "i=" << i;
        }
    }
}

TEST_F(AbsOpTest, NullMaskBoundaryStart) {
    double  input  = -99.0;
    double  output = -1.0;

    // Position 0 is null
    uint64_t nullMask[1] = {uint64_t{1} << 0};

    op_.evaluate(&input, &output, 1, nullMask);

    EXPECT_DOUBLE_EQ(output, 0.0);
}

TEST_F(AbsOpTest, NullMaskBoundaryEnd) {
    constexpr std::size_t N = 65;
    std::vector<double> input(N, -5.0);
    std::vector<double> output(N);

    // Position 64 is null (bit 0 of second uint64 word)
    uint64_t nullMask[2] = {};
    nullMask[1] |= (uint64_t{1} << 0);

    op_.evaluate(input.data(), output.data(), N, nullMask);

    EXPECT_DOUBLE_EQ(output[63], 5.0);  // non-null
    EXPECT_DOUBLE_EQ(output[64], 0.0);  // null
}

// ============================================================
// AbsOp — SIMD kernel cross-validation
// ============================================================

TEST_F(AbsOpTest, SimdMatchesScalarReference) {
    constexpr std::size_t N = 1024;
    std::vector<double> input(N);
    std::vector<double> simdOutput(N);
    std::vector<double> scalarOutput(N);

    // Generate pseudo-random values with sign variation
    for (std::size_t i = 0; i < N; ++i) {
        double val = static_cast<double>(i) * 1.5 - 500.0;
        // Add some special values at known positions
        if (i == 100) val = 0.0;
        if (i == 200) val = -0.0;
        if (i == 300) val = std::numeric_limits<double>::quiet_NaN();
        if (i == 400) val = std::numeric_limits<double>::infinity();
        if (i == 500) val = -std::numeric_limits<double>::infinity();
        input[i] = val;
    }

    // SIMD kernel (scalar fallback level)
    op_.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input.data(), N), simdOutput.data());

    // Scalar reference
    for (std::size_t i = 0; i < N; ++i) {
        scalarOutput[i] = op_.evaluateScalar(input[i]);
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

TEST_F(AbsOpTest, SimdSmallArray) {
    double input[]  = {3.0, -4.0, 0.0, -0.0};
    double output[4] = {};

    op_.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input, 4), output);

    EXPECT_DOUBLE_EQ(output[0], 3.0);
    EXPECT_DOUBLE_EQ(output[1], 4.0);
    EXPECT_DOUBLE_EQ(output[2], 0.0);
    EXPECT_DOUBLE_EQ(output[3], 0.0);
}

// ============================================================
// AbsOp — CRTP dispatch verification
// ============================================================

TEST_F(AbsOpTest, CrtpDispatchMatchesDirectCall) {
    // Verify that calling evaluate() through the base class produces
    // the same results as calling evaluateScalar() directly.
    constexpr std::size_t N = 256;
    std::vector<double> input(N);
    std::vector<double> output(N);

    for (std::size_t i = 0; i < N; ++i) {
        input[i] = static_cast<double>(i) - 128.0;
    }

    op_.evaluate(input.data(), output.data(), N, nullptr);

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(output[i], op_.evaluateScalar(input[i])) << "i=" << i;
    }
}

// ============================================================
// AbsOp — Compile-time constants
// ============================================================

TEST_F(AbsOpTest, CompileTimeConstants) {
    // Verify kOpCode and name are compile-time accessible (static constexpr)
    static_assert(AbsOp::kOpCode == UnaryOpCode::ABS);
    EXPECT_STREQ(AbsOp::name, "abs");
}

// ============================================================
// AbsOp — Stateless (multiple instances produce same results)
// ============================================================

TEST_F(AbsOpTest, MultipleInstancesIdentical) {
    AbsOp op1, op2;

    double testVals[] = {-5.0, 0.0, 5.0, -3.14};
    for (double x : testVals) {
        EXPECT_DOUBLE_EQ(op1.evaluateScalar(x), op2.evaluateScalar(x)) << "x=" << x;
    }
}
