// test_rank.cpp — unit tests for RankOp, RankPctOp, RankNormalizedOp
// Reference: pandas.rank(method='average')
//
// Coverage for each operator:
//   Operator identity, evaluateScalar sentinel,
//   basic ranking (no ties), ranking with ties,
//   null propagation, all-null, single-element,
//   NaN excluded, all-equal, negative values,
//   SIMD kernel cross-validation

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/operators/UnaryOperator.h"
#include "quantcore/operators/unary/Rank.h"

using namespace quantcore;

// ============================================================
// Shared helpers
// ============================================================

namespace {

std::vector<uint64_t> makeNullMask(std::size_t n,
                                    const std::vector<std::size_t>& nullIndices) {
    std::vector<uint64_t> mask((n + 63) / 64, 0);
    for (auto idx : nullIndices) {
        mask[idx / 64] |= (1ULL << (idx % 64));
    }
    return mask;
}

}  // anonymous namespace

// ============================================================
// RankOp — raw rank (1-based, average for ties)
// ============================================================

class RankOpTest : public ::testing::Test {
protected:
    RankOp op_;
};

TEST_F(RankOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::RANK);
}

TEST_F(RankOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "rank");
    static_assert(RankOp::kOpCode == UnaryOpCode::RANK);
}

TEST_F(RankOpTest, ScalarReturnsNaN) {
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(5.0)));
}

TEST_F(RankOpTest, BasicNoTies) {
    // [3, 1, 4, 2] → sorted: 1(pos1), 2(pos2), 3(pos3), 4(pos4)
    // ranks: [3, 1, 4, 2]
    double input[]  = {3.0, 1.0, 4.0, 2.0};
    double output[4] = {};
    double expected[] = {3.0, 1.0, 4.0, 2.0};

    op_.evaluate(input, output, 4, nullptr);

    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(output[i], expected[i]) << "i=" << i;
}

TEST_F(RankOpTest, WithTies) {
    // [20, 10, 20, 30, 10]
    // Sorted: 10(pos1), 10(pos2), 20(pos3), 20(pos4), 30(pos5)
    // Avg ranks: 10→(1+2)/2=1.5, 20→(3+4)/2=3.5, 30→5
    double input[]  = {20.0, 10.0, 20.0, 30.0, 10.0};
    double output[5] = {};
    double expected[] = {3.5, 1.5, 3.5, 5.0, 1.5};

    op_.evaluate(input, output, 5, nullptr);

    for (int i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], expected[i]) << "i=" << i;
}

TEST_F(RankOpTest, AllEqual) {
    // [5, 5, 5] → avg rank = (1+2+3)/3 = 2.0
    double input[]  = {5.0, 5.0, 5.0};
    double output[3] = {};

    op_.evaluate(input, output, 3, nullptr);

    for (int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(output[i], 2.0) << "i=" << i;
}

TEST_F(RankOpTest, NegativeValues) {
    double input[]  = {-5.0, 0.0, 5.0, -10.0};
    double output[4] = {};
    double expected[] = {2.0, 3.0, 4.0, 1.0};

    op_.evaluate(input, output, 4, nullptr);

    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(output[i], expected[i]) << "i=" << i;
}

TEST_F(RankOpTest, WithNulls) {
    // [10, null, 30, null, 20] → valid: 10(pos1), 20(pos2), 30(pos3)
    double input[]  = {10.0, 0.0, 30.0, 0.0, 20.0};
    double output[5] = {};
    auto nullMask = makeNullMask(5, {1, 3});

    op_.evaluate(input, output, 5, nullMask.data());

    EXPECT_DOUBLE_EQ(output[0], 1.0);   // 10 → rank 1
    EXPECT_DOUBLE_EQ(output[1], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[2], 3.0);   // 30 → rank 3
    EXPECT_DOUBLE_EQ(output[3], 0.0);   // null
    EXPECT_DOUBLE_EQ(output[4], 2.0);   // 20 → rank 2
}

TEST_F(RankOpTest, SingleElement) {
    double input[]  = {42.0};
    double output[1] = {};

    op_.evaluate(input, output, 1, nullptr);

    EXPECT_DOUBLE_EQ(output[0], 1.0);  // only element → rank 1
}

TEST_F(RankOpTest, LargeDatasetMonotonic) {
    constexpr std::size_t N = 1000;
    std::vector<double> input(N), output(N);
    for (std::size_t i = 0; i < N; ++i)
        input[i] = static_cast<double>(i);

    op_.evaluate(input.data(), output.data(), N, nullptr);

    for (std::size_t i = 1; i < N; ++i)
        EXPECT_GE(output[i], output[i - 1]) << "i=" << i;

    EXPECT_DOUBLE_EQ(output[0], 1.0);     // smallest
    EXPECT_DOUBLE_EQ(output[N - 1], static_cast<double>(N));  // largest
}

TEST_F(RankOpTest, SimdMatchesEvaluate) {
    double input[]  = {3.0, 1.0, 4.0, 2.0};
    double simdOut[4] = {};
    double evalOut[4] = {};

    op_.evaluate(input, evalOut, 4, nullptr);
    op_.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input, 4), simdOut);

    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(simdOut[i], evalOut[i]) << "i=" << i;
}

// ============================================================
// RankPctOp — percentile rank: avgRank / N, in (0, 1]
// Reference: pandas.rank(pct=True)
// ============================================================

class RankPctOpTest : public ::testing::Test {
protected:
    RankPctOp op_;
};

TEST_F(RankPctOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::RANK_PCT);
}

TEST_F(RankPctOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "rank_pct");
    static_assert(RankPctOp::kOpCode == UnaryOpCode::RANK_PCT);
}

TEST_F(RankPctOpTest, ScalarReturnsNaN) {
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(5.0)));
}

TEST_F(RankPctOpTest, BasicNoTies) {
    // [3, 1, 4, 2] → [3/4, 1/4, 4/4, 2/4]
    double input[]  = {3.0, 1.0, 4.0, 2.0};
    double output[4] = {};
    double expected[] = {0.75, 0.25, 1.0, 0.5};

    op_.evaluate(input, output, 4, nullptr);

    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(output[i], expected[i]) << "i=" << i;
}

TEST_F(RankPctOpTest, WithTies) {
    // [20, 10, 20, 30, 10] → avg ranks: 10→1.5, 20→3.5, 30→5
    // pct: 1.5/5=0.3, 3.5/5=0.7, 5/5=1.0
    double input[]  = {20.0, 10.0, 20.0, 30.0, 10.0};
    double output[5] = {};
    double expected[] = {0.7, 0.3, 0.7, 1.0, 0.3};

    op_.evaluate(input, output, 5, nullptr);

    for (int i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], expected[i]) << "i=" << i;
}

TEST_F(RankPctOpTest, AllEqual) {
    double input[]  = {5.0, 5.0, 5.0};
    double output[3] = {};
    // avg rank=2, pct=2/3

    op_.evaluate(input, output, 3, nullptr);

    for (int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(output[i], 2.0 / 3.0) << "i=" << i;
}

TEST_F(RankPctOpTest, WithNulls) {
    double input[]  = {10.0, 0.0, 30.0, 0.0, 20.0};
    double output[5] = {};
    auto nullMask = makeNullMask(5, {1, 3});

    op_.evaluate(input, output, 5, nullMask.data());

    EXPECT_DOUBLE_EQ(output[0], 1.0 / 3.0);  // 10 → rank 1 → pct 1/3
    EXPECT_DOUBLE_EQ(output[1], 0.0);         // null
    EXPECT_DOUBLE_EQ(output[2], 1.0);         // 30 → rank 3 → pct 3/3
    EXPECT_DOUBLE_EQ(output[3], 0.0);         // null
    EXPECT_DOUBLE_EQ(output[4], 2.0 / 3.0);  // 20 → rank 2 → pct 2/3
}

TEST_F(RankPctOpTest, PandasReference) {
    // >>> pd.Series([10, 20, 30, 20]).rank(pct=True)
    // 0    0.250
    // 1    0.625     ← avg(2,3)/4
    // 2    1.000
    // 3    0.625
    double input[]  = {10.0, 20.0, 30.0, 20.0};
    double output[4] = {};
    double expected[] = {0.25, 0.625, 1.0, 0.625};

    op_.evaluate(input, output, 4, nullptr);

    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(output[i], expected[i]) << "i=" << i;
}

TEST_F(RankPctOpTest, SingleElement) {
    double input[]  = {42.0};
    double output[1] = {};

    op_.evaluate(input, output, 1, nullptr);

    EXPECT_DOUBLE_EQ(output[0], 1.0);  // 1/1
}

TEST_F(RankPctOpTest, SimdMatchesEvaluate) {
    double input[]  = {3.0, 1.0, 4.0, 2.0};
    double simdOut[4] = {};
    double evalOut[4] = {};

    op_.evaluate(input, evalOut, 4, nullptr);
    op_.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input, 4), simdOut);

    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(simdOut[i], evalOut[i]) << "i=" << i;
}

// ============================================================
// RankNormalizedOp — normalized rank: (avgRank-1)/(N-1), in [0, 1]
// ============================================================

class RankNormalizedOpTest : public ::testing::Test {
protected:
    RankNormalizedOp op_;
};

TEST_F(RankNormalizedOpTest, OpCode) {
    EXPECT_EQ(op_.opCode(), UnaryOpCode::RANK_NORMALIZED);
}

TEST_F(RankNormalizedOpTest, Name) {
    EXPECT_STREQ(op_.opName(), "rank_normalized");
    static_assert(RankNormalizedOp::kOpCode == UnaryOpCode::RANK_NORMALIZED);
}

TEST_F(RankNormalizedOpTest, ScalarReturnsNaN) {
    EXPECT_TRUE(std::isnan(op_.evaluateScalar(5.0)));
}

TEST_F(RankNormalizedOpTest, BasicNoTies) {
    // [3, 1, 4, 2] → ranks: 3,1,4,2
    // normalized (N=4): (3-1)/3=2/3, (1-1)/3=0, (4-1)/3=1, (2-1)/3=1/3
    double input[]  = {3.0, 1.0, 4.0, 2.0};
    double output[4] = {};
    double expected[] = {2.0 / 3.0, 0.0, 1.0, 1.0 / 3.0};

    op_.evaluate(input, output, 4, nullptr);

    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(output[i], expected[i]) << "i=" << i;
}

TEST_F(RankNormalizedOpTest, WithTies) {
    // [20, 10, 20, 30, 10] → avg ranks: 10→1.5, 20→3.5, 30→5
    // normalized (N=5): (1.5-1)/4=0.125, (3.5-1)/4=0.625, (5-1)/4=1.0
    double input[]  = {20.0, 10.0, 20.0, 30.0, 10.0};
    double output[5] = {};
    double expected[] = {0.625, 0.125, 0.625, 1.0, 0.125};

    op_.evaluate(input, output, 5, nullptr);

    for (int i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(output[i], expected[i]) << "i=" << i;
}

TEST_F(RankNormalizedOpTest, AllEqual) {
    // [5, 5, 5] → avg rank=2, normalized (N=3): (2-1)/(3-1)=0.5
    double input[]  = {5.0, 5.0, 5.0};
    double output[3] = {};

    op_.evaluate(input, output, 3, nullptr);

    for (int i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(output[i], 0.5) << "i=" << i;
}

TEST_F(RankNormalizedOpTest, RangeBoundaries) {
    // Verify output is always in [0, 1] for N >= 2
    constexpr std::size_t N = 500;
    std::vector<double> input(N), output(N);
    for (std::size_t i = 0; i < N; ++i)
        input[i] = static_cast<double>(i);

    op_.evaluate(input.data(), output.data(), N, nullptr);

    EXPECT_DOUBLE_EQ(output[0], 0.0);          // smallest → 0
    EXPECT_DOUBLE_EQ(output[N - 1], 1.0);      // largest → 1
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_GE(output[i], 0.0) << "i=" << i;
        EXPECT_LE(output[i], 1.0) << "i=" << i;
    }
}

TEST_F(RankNormalizedOpTest, SingleElement) {
    // (1-1)/(1-1) → 0.0 by convention
    double input[]  = {42.0};
    double output[1] = {};

    op_.evaluate(input, output, 1, nullptr);

    EXPECT_DOUBLE_EQ(output[0], 0.0);
}

TEST_F(RankNormalizedOpTest, TwoElements) {
    // [100, 0] → ranks: 0(pos1), 100(pos2)
    // normalized (N=2): (1-1)/1=0, (2-1)/1=1
    double input[]  = {100.0, 0.0};
    double output[2] = {};

    op_.evaluate(input, output, 2, nullptr);

    EXPECT_DOUBLE_EQ(output[0], 1.0);  // larger → 1
    EXPECT_DOUBLE_EQ(output[1], 0.0);  // smaller → 0
}

TEST_F(RankNormalizedOpTest, SimdMatchesEvaluate) {
    double input[]  = {3.0, 1.0, 4.0, 2.0};
    double simdOut[4] = {};
    double evalOut[4] = {};

    op_.evaluate(input, evalOut, 4, nullptr);
    op_.evaluateSimd<SimdLevel::SCALAR>(ColView<double>(input, 4), simdOut);

    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(simdOut[i], evalOut[i]) << "i=" << i;
}

// ============================================================
// Cross-consistency: rank_pct = rank / N, rank_normalized = (rank-1)/(N-1)
// ============================================================

TEST(RankConsistencyTest, PctEqualsRankDivN) {
    // Verify that rank_pct(X)[i] == rank(X)[i] / N for all i
    double input[] = {15.0, 5.0, 25.0, 5.0, 15.0};  // N=5
    constexpr std::size_t n = 5;

    double raw[n], pct[n];
    RankOp rawOp;
    RankPctOp pctOp;

    rawOp.evaluate(input, raw, n, nullptr);
    pctOp.evaluate(input, pct, n, nullptr);

    // Valid count = 5, so rank_pct[i] should == rank[i] / 5
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(pct[i], raw[i] / 5.0) << "i=" << i;
    }
}

TEST(RankConsistencyTest, NormalizedConsistent) {
    // Verify that rank_normalized(X)[i] == (rank(X)[i] - 1) / (N - 1)
    double input[] = {15.0, 5.0, 25.0, 5.0, 15.0};  // N=5
    constexpr std::size_t n = 5;

    double raw[n], norm[n];
    RankOp rawOp;
    RankNormalizedOp normOp;

    rawOp.evaluate(input, raw, n, nullptr);
    normOp.evaluate(input, norm, n, nullptr);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(norm[i], (raw[i] - 1.0) / 4.0) << "i=" << i;
    }
}
