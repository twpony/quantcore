// test_cs_ops.cpp — unit tests for red (cross-section) operators
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include "quantcore/core/Types.h"
#include "quantcore/operators/RedOperator.h"
#include "quantcore/operators/red/RedMax.h"
#include "quantcore/operators/red/RedMean.h"
#include "quantcore/operators/red/RedMedian.h"
#include "quantcore/operators/red/RedMin.h"
#include "quantcore/operators/red/RedMul.h"
#include "quantcore/operators/red/RedQuantile.h"
#include "quantcore/operators/red/RedStd.h"
#include "quantcore/operators/red/RedSum.h"
#include "quantcore/operators/red/RedVar.h"
#include "quantcore/operators/red/RedZScore.h"
#include "quantcore/storage/ColView.h"

using namespace quantcore;

static const double kValues[] = {10.0, 12.0, 8.0, 15.0, 11.0};
static constexpr std::size_t kN = 5;

class RedOpTest : public ::testing::Test {};

// ============================================================
// RedSum
// ============================================================
TEST_F(RedOpTest, Sum) {
    RedSumOp op;
    EXPECT_NEAR(op.reduce(ColView<double>(kValues, kN)), 56.0, 1e-12);
}

TEST_F(RedOpTest, SumWithNaN) {
    double vals[] = {10.0, 12.0, std::numeric_limits<double>::quiet_NaN(), 15.0, 11.0};
    RedSumOp op;
    EXPECT_NEAR(op.reduce(ColView<double>(vals, kN)), 48.0, 1e-12);
}

TEST_F(RedOpTest, SumOpCode) {
    RedSumOp op;
    EXPECT_EQ(op.opCode(), RedOpCode::RED_SUM);
    EXPECT_STREQ(op.opName(), "red_sum");
}

// ============================================================
// RedMean
// ============================================================
TEST_F(RedOpTest, Mean) {
    RedMeanOp op;
    EXPECT_NEAR(op.reduce(ColView<double>(kValues, kN)), 11.2, 1e-12);
}

TEST_F(RedOpTest, MeanWithNaN) {
    double vals[] = {10.0, 12.0, std::numeric_limits<double>::quiet_NaN(), 15.0, 11.0};
    RedMeanOp op;
    EXPECT_NEAR(op.reduce(ColView<double>(vals, kN)), 12.0, 1e-12);
}

// ============================================================
// RedStd / RedVar
// ============================================================
TEST_F(RedOpTest, Std) {
    RedStdOp op;
    EXPECT_NEAR(op.reduce(ColView<double>(kValues, kN)), 2.31516738055804, 1e-12);
}

TEST_F(RedOpTest, StdConstant) {
    double vals[] = {5.0, 5.0, 5.0};
    RedStdOp op;
    EXPECT_NEAR(op.reduce(ColView<double>(vals, 3)), 0.0, 1e-12);
}

TEST_F(RedOpTest, Var) {
    RedVarOp op;
    EXPECT_NEAR(op.reduce(ColView<double>(kValues, kN)), 5.36, 1e-12);
}

// ============================================================
// RedMin / RedMax / RedMul
// ============================================================
TEST_F(RedOpTest, Min) {
    RedMinOp op;
    EXPECT_DOUBLE_EQ(op.reduce(ColView<double>(kValues, kN)), 8.0);
}

TEST_F(RedOpTest, Max) {
    RedMaxOp op;
    EXPECT_DOUBLE_EQ(op.reduce(ColView<double>(kValues, kN)), 15.0);
}

TEST_F(RedOpTest, Prod) {
    RedMulOp op;
    EXPECT_DOUBLE_EQ(op.reduce(ColView<double>(kValues, kN)), 10.0 * 12.0 * 8.0 * 15.0 * 11.0);
}

// ============================================================
// RedMedian
// ============================================================
TEST_F(RedOpTest, Median) {
    RedMedianOp op;
    EXPECT_DOUBLE_EQ(op.reduce(ColView<double>(kValues, kN)), 11.0);
}

TEST_F(RedOpTest, MedianEven) {
    double vals[] = {3.0, 7.0, 2.0, 9.0};  // sorted: 2,3,7,9 → median = (3+7)/2 = 5
    RedMedianOp op;
    EXPECT_DOUBLE_EQ(op.reduce(ColView<double>(vals, 4)), 5.0);
}

TEST_F(RedOpTest, MedianSingle) {
    double vals[] = {42.0};
    RedMedianOp op;
    EXPECT_DOUBLE_EQ(op.reduce(ColView<double>(vals, 1)), 42.0);
}

TEST_F(RedOpTest, MedianWithNaN) {
    double vals[] = {10.0, 12.0, std::numeric_limits<double>::quiet_NaN(), 15.0, 11.0};
    RedMedianOp op;
    EXPECT_NEAR(op.reduce(ColView<double>(vals, kN)), 11.5, 1e-12);  // sorted: 10,11,12,15 → (11+12)/2
}

TEST_F(RedOpTest, MedianAllNaN) {
    double vals[] = {std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::quiet_NaN()};
    RedMedianOp op;
    EXPECT_TRUE(std::isnan(op.reduce(ColView<double>(vals, 2))));
}

TEST_F(RedOpTest, MedianBroadcast) {
    RedMedianOp op;
    double output[kN];
    op.evaluate(ColView<double>(kValues, kN), output);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(output[i], 11.0) << "i=" << i;
}

// ============================================================
// RedZScore
// ============================================================
TEST_F(RedOpTest, ZScore) {
    RedZScoreOp op;
    double output[kN];
    op.evaluate(ColView<double>(kValues, kN), output);

    double mean = 11.2, stdv = 2.31516738055804;
    for (std::size_t i = 0; i < kN; ++i) {
        double expected = (kValues[i] - mean) / stdv;
        EXPECT_NEAR(output[i], expected, 1e-12) << "i=" << i;
    }
}

TEST_F(RedOpTest, ZScoreWithNaN) {
    double vals[] = {10.0, 12.0, std::numeric_limits<double>::quiet_NaN(), 15.0, 11.0};
    RedZScoreOp op;
    double output[5];
    op.evaluate(ColView<double>(vals, 5), output);
    EXPECT_TRUE(std::isnan(output[2]));
    // mean of non-NaN values = 12.0
    double mean_exp = 12.0;
    double sumSq = 4.0 + 0.0 + 9.0 + 1.0;  // (10-12)^2, (12-12)^2, (15-12)^2, (11-12)^2
    double stdv_exp = std::sqrt(sumSq / 4.0);
    for (std::size_t i = 0; i < 5; ++i) {
        if (i == 2) continue;
        double expected = (vals[i] - mean_exp) / stdv_exp;
        EXPECT_NEAR(output[i], expected, 1e-12) << "i=" << i;
    }
}

// ============================================================
// RedQuantile
// ============================================================
TEST_F(RedOpTest, Quantile) {
    RedQuantileOp op;
    double output[kN];
    op.evaluate(ColView<double>(kValues, kN), output);
    double expected[] = {0.4, 0.8, 0.2, 1.0, 0.6};
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_NEAR(output[i], expected[i], 1e-12) << "i=" << i;
}

TEST_F(RedOpTest, QuantileWithTies) {
    double vals[] = {10.0, 10.0, 20.0};
    RedQuantileOp op;
    double output[3];
    op.evaluate(ColView<double>(vals, 3), output);
    EXPECT_NEAR(output[0], 0.5, 1e-12);
    EXPECT_NEAR(output[1], 0.5, 1e-12);
    EXPECT_NEAR(output[2], 1.0, 1e-12);
}

// ============================================================
// evaluate() — broadcast behavior for reduction ops
// ============================================================
TEST_F(RedOpTest, EvaluateBroadcastsScalar) {
    RedMeanOp op;
    double output[kN];
    op.evaluate(ColView<double>(kValues, kN), output);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_NEAR(output[i], 11.2, 1e-12) << "i=" << i;
}

// ============================================================
// SIMD cross-validation
// ============================================================
TEST_F(RedOpTest, SimdMatchesScalarMean) {
    double vals[] = {3.0, 7.0, 2.0, 9.0, 5.0, 1.0, 8.0, 6.0, 4.0, 10.0};
    constexpr std::size_t N = 10;
    RedMeanOp op;
    double scalarOut[N], simdOut[N];
    op.evaluate(ColView<double>(vals, N), scalarOut);
    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(vals, N), simdOut);
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(simdOut[i], scalarOut[i]) << "i=" << i;
}

TEST_F(RedOpTest, SimdMatchesScalarZScore) {
    double vals[] = {3.0, 7.0, 2.0, 9.0, 5.0, 1.0, 8.0, 6.0, 4.0, 10.0};
    constexpr std::size_t N = 10;
    RedZScoreOp op;
    double scalarOut[N], simdOut[N];
    op.evaluate(ColView<double>(vals, N), scalarOut);
    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(vals, N), simdOut);
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(simdOut[i], scalarOut[i]) << "i=" << i;
}
