// test_cs_ops.cpp — unit tests for CS (cross-section transform) operators
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "quantcore/core/Types.h"
#include "quantcore/operators/CsOperator.h"
#include "quantcore/operators/cs/CsClip.h"
#include "quantcore/operators/cs/CsDemean.h"
#include "quantcore/operators/cs/CsNormalizeL1.h"
#include "quantcore/operators/cs/CsNormalizeL2.h"
#include "quantcore/operators/cs/CsQuantile.h"
#include "quantcore/operators/cs/CsRank.h"
#include "quantcore/operators/cs/CsNormalize.h"
#include "quantcore/operators/cs/CsWinsorize.h"
#include "quantcore/operators/cs/CsWinsorizeMAD.h"
#include "quantcore/operators/cs/CsZScore.h"
#include "quantcore/storage/ColView.h"

using namespace quantcore;

static const double kValues[] = {10.0, 12.0, 8.0, 15.0, 11.0};
static constexpr std::size_t kN = 5;

class CsOpTest : public ::testing::Test {};

// ============================================================
// CsRank
// ============================================================
TEST_F(CsOpTest, Rank) {
    CsRankOp op;
    double output[kN];
    op.evaluate(ColView<double>(kValues, kN), output);
    // Sorted: 8,10,11,12,15 → ranks should be average for ties
    // 8→1, 10→2, 11→3, 12→4, 15→5
    double expected[] = {2.0, 4.0, 1.0, 5.0, 3.0};
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(output[i], expected[i]) << "i=" << i;
}

TEST_F(CsOpTest, RankWithTies) {
    double vals[] = {10.0, 10.0, 20.0};
    CsRankOp op;
    double output[3];
    op.evaluate(ColView<double>(vals, 3), output);
    // Sorted: 10,10,20 → (1+2)/2=1.5 for both 10s, 3 for 20
    EXPECT_DOUBLE_EQ(output[0], 1.5);
    EXPECT_DOUBLE_EQ(output[1], 1.5);
    EXPECT_DOUBLE_EQ(output[2], 3.0);
}

TEST_F(CsOpTest, RankWithNaN) {
    double vals[] = {10.0, std::numeric_limits<double>::quiet_NaN(), 20.0};
    CsRankOp op;
    double output[3];
    op.evaluate(ColView<double>(vals, 3), output);
    EXPECT_DOUBLE_EQ(output[0], 1.0);
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_DOUBLE_EQ(output[2], 2.0);
}

// ============================================================
// CsQuantile
// ============================================================
TEST_F(CsOpTest, Quantile) {
    CsQuantileOp op;
    double output[kN];
    op.evaluate(ColView<double>(kValues, kN), output);
    double expected[] = {0.4, 0.8, 0.2, 1.0, 0.6};
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_NEAR(output[i], expected[i], 1e-12) << "i=" << i;
}

TEST_F(CsOpTest, QuantileWithNaN) {
    double vals[] = {10.0, std::numeric_limits<double>::quiet_NaN(), 20.0};
    CsQuantileOp op;
    double output[3];
    op.evaluate(ColView<double>(vals, 3), output);
    EXPECT_NEAR(output[0], 0.5, 1e-12);
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_NEAR(output[2], 1.0, 1e-12);
}

// ============================================================
// CsZScore
// ============================================================
TEST_F(CsOpTest, ZScore) {
    CsZScoreOp op;
    double output[kN];
    op.evaluate(ColView<double>(kValues, kN), output);
    double mean = 11.2, stdv = 2.31516738055804;
    for (std::size_t i = 0; i < kN; ++i) {
        double expected = (kValues[i] - mean) / stdv;
        EXPECT_NEAR(output[i], expected, 1e-12) << "i=" << i;
    }
}

TEST_F(CsOpTest, ZScoreConstant) {
    double vals[] = {5.0, 5.0, 5.0};
    CsZScoreOp op;
    double output[3];
    op.evaluate(ColView<double>(vals, 3), output);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(output[i], 0.0);
}

// ============================================================
// CsNormalize
// ============================================================
TEST_F(CsOpTest, Normalize) {
    CsNormalizeOp op;
    double output[kN];
    op.evaluate(ColView<double>(kValues, kN), output);
    double minV = 8.0, maxV = 15.0;
    for (std::size_t i = 0; i < kN; ++i) {
        double expected = (kValues[i] - minV) / (maxV - minV);
        EXPECT_NEAR(output[i], expected, 1e-12) << "i=" << i;
    }
}

TEST_F(CsOpTest, NormalizeConstant) {
    double vals[] = {3.0, 3.0, 3.0};
    CsNormalizeOp op;
    double output[3];
    op.evaluate(ColView<double>(vals, 3), output);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(output[i], 0.0);
}

// ============================================================
// CsNormalizeL2
// ============================================================
TEST_F(CsOpTest, NormalizeL2) {
    double vals[] = {3.0, 4.0};  // L2 norm = 5
    CsNormalizeL2Op op;
    double output[2];
    op.evaluate(ColView<double>(vals, 2), output);
    EXPECT_DOUBLE_EQ(output[0], 3.0 / 5.0);
    EXPECT_DOUBLE_EQ(output[1], 4.0 / 5.0);
}

TEST_F(CsOpTest, NormalizeL2ZeroNorm) {
    double vals[] = {0.0, 0.0};
    CsNormalizeL2Op op;
    double output[2];
    op.evaluate(ColView<double>(vals, 2), output);
    EXPECT_DOUBLE_EQ(output[0], 0.0);
    EXPECT_DOUBLE_EQ(output[1], 0.0);
}

// ============================================================
// CsNormalizeL1
// ============================================================
TEST_F(CsOpTest, NormalizeL1) {
    double vals[] = {3.0, -4.0, 0.0};  // sum(|x|) = 7
    CsNormalizeL1Op op;
    double output[3];
    op.evaluate(ColView<double>(vals, 3), output);
    EXPECT_DOUBLE_EQ(output[0], 3.0 / 7.0);
    EXPECT_DOUBLE_EQ(output[1], -4.0 / 7.0);
    EXPECT_DOUBLE_EQ(output[2], 0.0);
}

TEST_F(CsOpTest, NormalizeL1ZeroNorm) {
    double vals[] = {0.0, 0.0, 0.0};
    CsNormalizeL1Op op;
    double output[3];
    op.evaluate(ColView<double>(vals, 3), output);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(output[i], 0.0);
}

TEST_F(CsOpTest, NormalizeL1WithNaN) {
    double vals[] = {1.0, std::numeric_limits<double>::quiet_NaN(), 2.0};
    CsNormalizeL1Op op;
    double output[3];
    op.evaluate(ColView<double>(vals, 3), output);
    EXPECT_DOUBLE_EQ(output[0], 1.0 / 3.0);
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_DOUBLE_EQ(output[2], 2.0 / 3.0);
}

// ============================================================
// CsDemean
// ============================================================
TEST_F(CsOpTest, Demean) {
    CsDemeanOp op;
    double output[kN];
    op.evaluate(ColView<double>(kValues, kN), output);
    double mean = 11.2;
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_NEAR(output[i], kValues[i] - mean, 1e-12) << "i=" << i;
}

TEST_F(CsOpTest, DemeanAllNaN) {
    double vals[] = {std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::quiet_NaN()};
    CsDemeanOp op;
    double output[2];
    op.evaluate(ColView<double>(vals, 2), output);
    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
}

// ============================================================
// CsWinsorize
// ============================================================
TEST_F(CsOpTest, WinsorizeDefault) {
    // 10 values — 1% and 99% percentile should clip the extremes
    double vals[] = {-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 200.0};
    CsWinsorizeOp op;
    double output[10];
    op.evaluate(ColView<double>(vals, 10), output, 0.01, 0.99);  // default 1% / 99%
    // The 1% and 99% should clip the extremes
    EXPECT_GT(output[0], -100.0);  // was -100, should be raised
    EXPECT_LT(output[9], 200.0);   // was 200, should be lowered
    // Middle values unchanged
    for (std::size_t i = 1; i < 9; ++i)
        EXPECT_DOUBLE_EQ(output[i], vals[i]) << "i=" << i;
}

TEST_F(CsOpTest, WinsorizeAllInside) {
    double vals[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    CsWinsorizeOp op;
    double output[5];
    op.evaluate(ColView<double>(vals, 5), output, 0.0, 1.0);  // 0% to 100% — no clipping
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], vals[i]);
}

// ============================================================
// CsWinsorizeMAD
// ============================================================
TEST_F(CsOpTest, WinsorizeMAD) {
    // Values centered at 100: 90, 95, 100, 105, 110
    // median = 100, MAD = median(|90-100|,|95-100|,0,|105-100|,|110-100|)
    //                   = median(10, 5, 0, 5, 10) = 5
    // n=3 → lo=85, hi=115 — no clipping
    double vals[] = {90.0, 95.0, 100.0, 105.0, 110.0};
    CsWinsorizeMADOp op;
    double output[5];
    op.evaluate(ColView<double>(vals, 5), output, 3.0);
    for (std::size_t i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], vals[i]);
}

TEST_F(CsOpTest, WinsorizeMADClip) {
    // Values centered at 100: 70, 98, 100, 102, 150
    // median = 100, MAD = median(30, 2, 0, 2, 50) = 2
    // n=5 → lo=90, hi=110 — clips 70→90, 150→110
    double vals[] = {70.0, 98.0, 100.0, 102.0, 150.0};
    CsWinsorizeMADOp op;
    double output[5];
    op.evaluate(ColView<double>(vals, 5), output, 5.0);
    EXPECT_DOUBLE_EQ(output[0], 90.0);   // 70 → 90
    EXPECT_DOUBLE_EQ(output[1], 98.0);   // unchanged
    EXPECT_DOUBLE_EQ(output[2], 100.0);  // unchanged
    EXPECT_DOUBLE_EQ(output[3], 102.0);  // unchanged
    EXPECT_DOUBLE_EQ(output[4], 110.0);  // 150 → 110
}

TEST_F(CsOpTest, WinsorizeMADZeroMAD) {
    // All equal → MAD = 0 → no clipping applied
    double vals[] = {5.0, 5.0, 5.0};
    CsWinsorizeMADOp op;
    double output[3];
    op.evaluate(ColView<double>(vals, 3), output, 3.0);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(output[i], 5.0);
}

TEST_F(CsOpTest, WinsorizeMADWithNaN) {
    // Non-NaN: 0, 2, 5, 20 → median=(2+5)/2=3.5, MAD=2.5
    // n=3 → lo=3.5-7.5=-4, hi=3.5+7.5=11 → clips 20→11
    double vals[] = {0.0, std::numeric_limits<double>::quiet_NaN(), 2.0, 5.0, 20.0};
    CsWinsorizeMADOp op;
    double output[5];
    op.evaluate(ColView<double>(vals, 5), output, 3.0);
    EXPECT_DOUBLE_EQ(output[0], 0.0);
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_DOUBLE_EQ(output[2], 2.0);
    EXPECT_DOUBLE_EQ(output[3], 5.0);
    EXPECT_DOUBLE_EQ(output[4], 11.0);
}

// ============================================================
// CsClip
// ============================================================
TEST_F(CsOpTest, Clip) {
    double vals[] = {-10.0, 0.0, 5.0, 10.0, 20.0};
    CsClipOp op;
    double output[5];
    op.evaluate(ColView<double>(vals, 5), output, 0.0, 10.0);
    EXPECT_DOUBLE_EQ(output[0], 0.0);
    EXPECT_DOUBLE_EQ(output[1], 0.0);
    EXPECT_DOUBLE_EQ(output[2], 5.0);
    EXPECT_DOUBLE_EQ(output[3], 10.0);
    EXPECT_DOUBLE_EQ(output[4], 10.0);
}

TEST_F(CsOpTest, ClipNoChange) {
    double vals[] = {2.0, 3.0, 4.0};
    CsClipOp op;
    double output[3];
    op.evaluate(ColView<double>(vals, 3), output, 0.0, 10.0);
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(output[i], vals[i]);
}

// ============================================================
// SIMD cross-validation
// ============================================================
TEST_F(CsOpTest, SimdMatchesScalar) {
    double vals[] = {3.0, 7.0, 2.0, 9.0, 5.0, 1.0, 8.0, 6.0, 4.0, 10.0};
    constexpr std::size_t N = 10;
    CsZScoreOp op;
    double scalarOut[N], simdOut[N];
    op.evaluate(ColView<double>(vals, N), scalarOut);
    op.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(vals, N), simdOut);
    for (std::size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(simdOut[i], scalarOut[i]) << "i=" << i;
}

// ============================================================
// Operator identity
// ============================================================
TEST_F(CsOpTest, OpCodes) {
    EXPECT_EQ(CsRankOp::opCode(), CsOpCode::CS_RANK);
    EXPECT_EQ(CsZScoreOp::opCode(), CsOpCode::CS_ZSCORE);
    EXPECT_EQ(CsNormalizeOp::opCode(), CsOpCode::CS_NORMALIZE);
    EXPECT_EQ(CsNormalizeL1Op::opCode(), CsOpCode::CS_NORMALIZE_L1);
    EXPECT_STREQ(CsRankOp::opName(), "cs_rank");
    EXPECT_STREQ(CsDemeanOp::opName(), "cs_demean");
    EXPECT_STREQ(CsClipOp::opName(), "cs_clip");
    EXPECT_STREQ(CsNormalizeL1Op::opName(), "cs_normalize_l1");
    EXPECT_STREQ(CsWinsorizeMADOp::opName(), "cs_winsorize_mad");
}
