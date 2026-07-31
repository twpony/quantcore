// test_expression.cpp — unit tests for expression AST nodes
// Phase: 一期必实现
//
// Tests ColumnRef, Scalar, UnaryExpr, BinaryExpr, RollingExpr, and
// composite expressions.  Covers correct evaluation, clone semantics,
// null propagation, and deep nesting.
//
// Test data: synthetic MarketData with known values for all 7 fields.
// This avoids I/O dependencies and makes expected values deterministic.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>

#include "quantcore/core/Types.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/expression/ColumnRef.h"
#include "quantcore/expression/Scalar.h"
#include "quantcore/expression/UnaryExpr.h"
#include "quantcore/expression/BinaryExpr.h"
#include "quantcore/expression/RedExpr.h"
#include "quantcore/expression/CsExpr.h"
#include "quantcore/expression/ExprTraits.h"
#include "quantcore/expression/RollingExpr.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;

// ============================================================
// Test fixture — provides MarketData with known values
// ============================================================

class ExpressionTest : public ::testing::Test {
protected:
    static constexpr std::size_t kN = 20;

    void SetUp() override {
        // Build timestamp index (just sequential indices; not real dates)
        std::vector<int64_t> timestamps(kN);
        for (std::size_t i = 0; i < kN; ++i) {
            timestamps[i] = static_cast<int64_t>(i);
        }
        TimestampIndex tsIdx(timestamps.data(), kN);
        md_ = MarketData("TEST", std::move(tsIdx));

        // Fill each field with distinct values:
        //   OPEN  = 10.0 + i
        //   HIGH  = 15.0 + i
        //   LOW   =  8.0 + i
        //   CLOSE = 12.0 + i
        //   VOLUME= 100.0 * (i + 1)
        //   AMOUNT= 1000.0 * (i + 1)
        //   VWAP  = 11.5 + i
        for (std::size_t f = 0; f < static_cast<std::size_t>(Field::kFieldCount); ++f) {
            auto field = static_cast<Field>(f);
            Column<double> col(kN);
            double base = 0.0;
            double scale = 1.0;
            switch (field) {
                case Field::OPEN:   base = 10.0; scale = 1.0;  break;
                case Field::HIGH:   base = 15.0; scale = 1.0;  break;
                case Field::LOW:    base =  8.0; scale = 1.0;  break;
                case Field::CLOSE:  base = 12.0; scale = 1.0;  break;
                case Field::VOLUME: base = 100.0; scale = 100.0; break;
                case Field::AMOUNT: base = 1000.0; scale = 1000.0; break;
                case Field::VWAP:   base = 11.5; scale = 1.0;  break;
                default: break;
            }
            for (std::size_t i = 0; i < kN; ++i) {
                col[i] = base + scale * static_cast<double>(i);
            }
            md_.setColumn(field, std::move(col));
        }
    }

    MarketData md_;
};

// ============================================================
// ColumnRef tests
// ============================================================

TEST_F(ExpressionTest, ColumnRefClose) {
    ColumnRef ref(Field::CLOSE);
    std::vector<double> output(kN);
    ref.evaluate(md_, output.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], 12.0 + static_cast<double>(i));
    }
}

TEST_F(ExpressionTest, ColumnRefAllFields) {
    // Verify that each field produces the correct data
    for (std::size_t f = 0; f < static_cast<std::size_t>(Field::kFieldCount); ++f) {
        auto field = static_cast<Field>(f);
        ColumnRef ref(field);
        std::vector<double> output(kN);
        ref.evaluate(md_, output.data(), kN);

        const auto& col = md_.column<double>(field);
        for (std::size_t i = 0; i < kN; ++i) {
            EXPECT_DOUBLE_EQ(output[i], col[i]);
        }
    }
}

TEST_F(ExpressionTest, ColumnRefClone) {
    auto ref1 = std::make_unique<ColumnRef>(Field::CLOSE);
    auto ref2 = ref1->clone();

    std::vector<double> out1(kN), out2(kN);
    ref1->evaluate(md_, out1.data(), kN);
    ref2->evaluate(md_, out2.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(out1[i], out2[i]);
    }
}

// ============================================================
// Scalar tests
// ============================================================

TEST_F(ExpressionTest, ScalarPositive) {
    Scalar s(3.14);
    std::vector<double> output(kN);
    s.evaluate(md_, output.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], 3.14);
    }
}

TEST_F(ExpressionTest, ScalarZero) {
    Scalar s(0.0);
    std::vector<double> output(kN);
    s.evaluate(md_, output.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], 0.0);
    }
}

TEST_F(ExpressionTest, ScalarNegative) {
    Scalar s(-5.0);
    std::vector<double> output(kN);
    s.evaluate(md_, output.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], -5.0);
    }
}

TEST_F(ExpressionTest, ScalarClone) {
    auto s1 = std::make_unique<Scalar>(42.0);
    auto s2 = s1->clone();

    std::vector<double> out1(kN), out2(kN);
    s1->evaluate(md_, out1.data(), kN);
    s2->evaluate(md_, out2.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(out1[i], out2[i]);
    }
}

TEST_F(ExpressionTest, ScalarNoNullMask) {
    Scalar s(1.0);
    std::vector<double> output(kN);
    const uint64_t* nullMask = s.evaluate(md_, output.data(), kN);
    EXPECT_EQ(nullMask, nullptr);
}

// ============================================================
// UnaryExpr tests
// ============================================================

TEST_F(ExpressionTest, UnaryLog) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    UnaryExpr logExpr(UnaryOpCode::LOG, std::move(close));

    std::vector<double> output(kN);
    logExpr.evaluate(md_, output.data(), kN);

    const auto& col = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], std::log(col[i]));
    }
}

TEST_F(ExpressionTest, UnaryAbs) {
    // Use (LOW - HIGH) via ColumnRef first, then ABS
    // Actually test with a negative scalar for simplicity
    auto neg = std::make_unique<Scalar>(-5.0);
    UnaryExpr absExpr(UnaryOpCode::ABS, std::move(neg));

    std::vector<double> output(kN);
    absExpr.evaluate(md_, output.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], 5.0);
    }
}

TEST_F(ExpressionTest, UnarySqrt) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    UnaryExpr sqrtExpr(UnaryOpCode::SQRT, std::move(close));

    std::vector<double> output(kN);
    sqrtExpr.evaluate(md_, output.data(), kN);

    const auto& col = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], std::sqrt(col[i]));
    }
}

TEST_F(ExpressionTest, UnaryNested) {
    // LOG(ABS(CLOSE)) — double unary nesting
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    auto absExpr = std::make_unique<UnaryExpr>(UnaryOpCode::ABS, std::move(close));
    UnaryExpr logExpr(UnaryOpCode::LOG, std::move(absExpr));

    std::vector<double> output(kN);
    logExpr.evaluate(md_, output.data(), kN);

    const auto& col = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], std::log(std::abs(col[i])));
    }
}

TEST_F(ExpressionTest, UnaryClone) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    auto expr1 = std::make_unique<UnaryExpr>(UnaryOpCode::LOG, std::move(close));
    auto expr2 = expr1->clone();

    std::vector<double> out1(kN), out2(kN);
    expr1->evaluate(md_, out1.data(), kN);
    expr2->evaluate(md_, out2.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(out1[i], out2[i]);
    }
}

// ============================================================
// BinaryExpr tests
// ============================================================

TEST_F(ExpressionTest, BinaryAdd) {
    auto lhs = std::make_unique<ColumnRef>(Field::CLOSE);
    auto rhs = std::make_unique<ColumnRef>(Field::OPEN);
    BinaryExpr add(BinaryOpCode::ADD, std::move(lhs), std::move(rhs));

    std::vector<double> output(kN);
    add.evaluate(md_, output.data(), kN);

    const auto& close = md_.column<double>(Field::CLOSE);
    const auto& open  = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], close[i] + open[i]);
    }
}

TEST_F(ExpressionTest, BinarySub) {
    auto lhs = std::make_unique<ColumnRef>(Field::HIGH);
    auto rhs = std::make_unique<ColumnRef>(Field::LOW);
    BinaryExpr sub(BinaryOpCode::SUB, std::move(lhs), std::move(rhs));

    std::vector<double> output(kN);
    sub.evaluate(md_, output.data(), kN);

    const auto& high = md_.column<double>(Field::HIGH);
    const auto& low  = md_.column<double>(Field::LOW);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], high[i] - low[i]);
    }
}

TEST_F(ExpressionTest, BinaryMulColScalar) {
    auto col = std::make_unique<ColumnRef>(Field::CLOSE);
    auto s   = std::make_unique<Scalar>(2.5);
    BinaryExpr mul(BinaryOpCode::MUL, std::move(col), std::move(s));

    std::vector<double> output(kN);
    mul.evaluate(md_, output.data(), kN);

    const auto& close = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], close[i] * 2.5);
    }
}

TEST_F(ExpressionTest, BinaryDiv) {
    auto lhs = std::make_unique<ColumnRef>(Field::AMOUNT);
    auto rhs = std::make_unique<ColumnRef>(Field::VOLUME);
    BinaryExpr div(BinaryOpCode::DIV, std::move(lhs), std::move(rhs));

    std::vector<double> output(kN);
    div.evaluate(md_, output.data(), kN);

    const auto& amount = md_.column<double>(Field::AMOUNT);
    const auto& volume = md_.column<double>(Field::VOLUME);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], amount[i] / volume[i]);
    }
}

TEST_F(ExpressionTest, BinaryNested) {
    // (CLOSE + HIGH) / 2.0
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    auto high  = std::make_unique<ColumnRef>(Field::HIGH);
    auto add = std::make_unique<BinaryExpr>(BinaryOpCode::ADD,
        std::move(close), std::move(high));
    auto two = std::make_unique<Scalar>(2.0);
    BinaryExpr div(BinaryOpCode::DIV, std::move(add), std::move(two));

    std::vector<double> output(kN);
    div.evaluate(md_, output.data(), kN);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& h = md_.column<double>(Field::HIGH);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], (c[i] + h[i]) / 2.0);
    }
}

TEST_F(ExpressionTest, BinaryClone) {
    auto lhs = std::make_unique<ColumnRef>(Field::CLOSE);
    auto rhs = std::make_unique<ColumnRef>(Field::VWAP);
    auto expr1 = std::make_unique<BinaryExpr>(BinaryOpCode::SUB,
        std::move(lhs), std::move(rhs));
    auto expr2 = expr1->clone();

    std::vector<double> out1(kN), out2(kN);
    expr1->evaluate(md_, out1.data(), kN);
    expr2->evaluate(md_, out2.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(out1[i], out2[i]);
    }
}

// ============================================================
// BinaryExpr null propagation tests
// ============================================================

TEST_F(ExpressionTest, BinaryNullPropagationLhsNull) {
    // Create MarketData where CLOSE has nulls
    MarketData mdNull;
    {
        std::vector<int64_t> ts(kN);
        for (std::size_t i = 0; i < kN; ++i) {
            ts[i] = static_cast<int64_t>(i);
        }
        TimestampIndex tsi(ts.data(), kN);
        mdNull = MarketData("NULL_TEST", std::move(tsi));

        Column<double> close(kN);
        Column<double> open(kN);
        for (std::size_t i = 0; i < kN; ++i) {
            close[i] = 10.0 + static_cast<double>(i);
            open[i]  = 20.0 + static_cast<double>(i);
        }
        // Mark element 3 as null in CLOSE
        close.setNull(3);
        mdNull.setColumn(Field::CLOSE, std::move(close));
        mdNull.setColumn(Field::OPEN, std::move(open));
    }

    auto lhs = std::make_unique<ColumnRef>(Field::CLOSE);
    auto rhs = std::make_unique<ColumnRef>(Field::OPEN);
    BinaryExpr add(BinaryOpCode::ADD, std::move(lhs), std::move(rhs));

    std::vector<double> output(kN);
    const uint64_t* nullMask = add.evaluate(mdNull, output.data(), kN);

    // CLOSE[3] is null → output[3] should be null
    EXPECT_NE(nullMask, nullptr);
    EXPECT_TRUE((nullMask[0] >> 3) & uint64_t{1});
}

// ============================================================
// RollingExpr tests
// ============================================================

TEST_F(ExpressionTest, RollingMeanBasic) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    RollingExpr rolling(RollingOpCode::ROLLING_MEAN, 5, std::move(close));

    std::vector<double> output(kN);
    rolling.evaluate(md_, output.data(), kN);

    const auto& col = md_.column<double>(Field::CLOSE);

    // First 4 positions (window-1) should be NaN
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(output[i]));
    }

    // Position 4: mean of [0..4]
    double expected0 = (col[0] + col[1] + col[2] + col[3] + col[4]) / 5.0;
    EXPECT_DOUBLE_EQ(output[4], expected0);

    // Position 5: mean of [1..5]
    double expected1 = (col[1] + col[2] + col[3] + col[4] + col[5]) / 5.0;
    EXPECT_DOUBLE_EQ(output[5], expected1);
}

TEST_F(ExpressionTest, RollingSum) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    RollingExpr rolling(RollingOpCode::ROLLING_SUM, 3, std::move(close));

    std::vector<double> output(kN);
    rolling.evaluate(md_, output.data(), kN);

    const auto& col = md_.column<double>(Field::CLOSE);

    // First 2 positions NaN
    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));

    // Position 2: sum of [0..2]
    EXPECT_DOUBLE_EQ(output[2], col[0] + col[1] + col[2]);
}

TEST_F(ExpressionTest, RollingSma) {
    // ROLLING_SMA should be identical to ROLLING_MEAN
    auto close1 = std::make_unique<ColumnRef>(Field::CLOSE);
    auto close2 = std::make_unique<ColumnRef>(Field::CLOSE);
    RollingExpr mean(RollingOpCode::ROLLING_MEAN, 4, std::move(close1));
    RollingExpr sma(RollingOpCode::ROLLING_SMA, 4, std::move(close2));

    std::vector<double> outMean(kN), outSma(kN);
    mean.evaluate(md_, outMean.data(), kN);
    sma.evaluate(md_, outSma.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        if (std::isnan(outMean[i])) {
            EXPECT_TRUE(std::isnan(outSma[i]));
        } else {
            EXPECT_DOUBLE_EQ(outMean[i], outSma[i]);
        }
    }
}

TEST_F(ExpressionTest, RollingWithCompositeChild) {
    // ROLLING_MEAN(LOG(CLOSE), 3)
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    auto logExpr = std::make_unique<UnaryExpr>(UnaryOpCode::LOG, std::move(close));
    RollingExpr rolling(RollingOpCode::ROLLING_MEAN, 3, std::move(logExpr));

    std::vector<double> output(kN);
    rolling.evaluate(md_, output.data(), kN);

    const auto& col = md_.column<double>(Field::CLOSE);

    // First 2 NaN
    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));

    // Position 2: mean(log(c[0]), log(c[1]), log(c[2]))
    double expected = (std::log(col[0]) + std::log(col[1]) + std::log(col[2])) / 3.0;
    EXPECT_DOUBLE_EQ(output[2], expected);
}

TEST_F(ExpressionTest, RollingClone) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    auto expr1 = std::make_unique<RollingExpr>(RollingOpCode::ROLLING_STD, 5,
        std::move(close));
    auto expr2 = expr1->clone();

    std::vector<double> out1(kN), out2(kN);
    expr1->evaluate(md_, out1.data(), kN);
    expr2->evaluate(md_, out2.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        if (std::isnan(out1[i])) {
            EXPECT_TRUE(std::isnan(out2[i]));
        } else {
            EXPECT_DOUBLE_EQ(out1[i], out2[i]);
        }
    }
}

// ============================================================
// Composite expression tests
// ============================================================

TEST_F(ExpressionTest, CompositeAbsLogDiffMulVolume) {
    // ABS(LOG(CLOSE) - LOG(VWAP)) * VOLUME
    auto close  = std::make_unique<ColumnRef>(Field::CLOSE);
    auto vwap   = std::make_unique<ColumnRef>(Field::VWAP);
    auto volume = std::make_unique<ColumnRef>(Field::VOLUME);

    auto logClose = std::make_unique<UnaryExpr>(UnaryOpCode::LOG, std::move(close));
    auto logVwap  = std::make_unique<UnaryExpr>(UnaryOpCode::LOG, std::move(vwap));

    auto diff = std::make_unique<BinaryExpr>(BinaryOpCode::SUB,
        std::move(logClose), std::move(logVwap));

    auto absDiff = std::make_unique<UnaryExpr>(UnaryOpCode::ABS, std::move(diff));

    auto result = std::make_unique<BinaryExpr>(BinaryOpCode::MUL,
        std::move(absDiff), std::move(volume));

    std::vector<double> output(kN);
    result->evaluate(md_, output.data(), kN);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& v = md_.column<double>(Field::VWAP);
    const auto& vol = md_.column<double>(Field::VOLUME);

    for (std::size_t i = 0; i < kN; ++i) {
        double expected = std::abs(std::log(c[i]) - std::log(v[i])) * vol[i];
        EXPECT_DOUBLE_EQ(output[i], expected);
    }
}

TEST_F(ExpressionTest, CompositeHighMinusLowOverClose) {
    // (HIGH - LOW) / CLOSE
    auto high  = std::make_unique<ColumnRef>(Field::HIGH);
    auto low   = std::make_unique<ColumnRef>(Field::LOW);
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);

    auto diff = std::make_unique<BinaryExpr>(BinaryOpCode::SUB,
        std::move(high), std::move(low));
    auto result = std::make_unique<BinaryExpr>(BinaryOpCode::DIV,
        std::move(diff), std::move(close));

    std::vector<double> output(kN);
    result->evaluate(md_, output.data(), kN);

    const auto& h = md_.column<double>(Field::HIGH);
    const auto& l = md_.column<double>(Field::LOW);
    const auto& c = md_.column<double>(Field::CLOSE);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], (h[i] - l[i]) / c[i]);
    }
}

TEST_F(ExpressionTest, CompositeDeepNesting) {
    // SQRT(ABS(LOG(CLOSE))) — three levels of unary nesting
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    auto logExpr = std::make_unique<UnaryExpr>(UnaryOpCode::LOG, std::move(close));
    auto absExpr = std::make_unique<UnaryExpr>(UnaryOpCode::ABS, std::move(logExpr));
    UnaryExpr sqrtExpr(UnaryOpCode::SQRT, std::move(absExpr));

    std::vector<double> output(kN);
    sqrtExpr.evaluate(md_, output.data(), kN);

    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], std::sqrt(std::abs(std::log(c[i]))));
    }
}

// ============================================================
// Clone + re-evaluate on different MarketData
// ============================================================

TEST_F(ExpressionTest, CloneAndEvaluateOnDifferentData) {
    // Build tree once: CLOSE + OPEN
    auto close1 = std::make_unique<ColumnRef>(Field::CLOSE);
    auto open1  = std::make_unique<ColumnRef>(Field::OPEN);
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::ADD,
        std::move(close1), std::move(open1));

    // Clone
    auto clone1 = expr->clone();

    // Evaluate original on md_
    std::vector<double> outOrig(kN);
    expr->evaluate(md_, outOrig.data(), kN);

    // Evaluate clone on md_
    std::vector<double> outClone(kN);
    clone1->evaluate(md_, outClone.data(), kN);

    // Results should match
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(outOrig[i], outClone[i]);
    }

    // Create a second MarketData with different values and evaluate clone on it
    MarketData md2;
    {
        std::vector<int64_t> ts(10);
        for (std::size_t i = 0; i < 10; ++i) {
            ts[i] = static_cast<int64_t>(i);
        }
        TimestampIndex tsi(ts.data(), 10);
        md2 = MarketData("OTHER", std::move(tsi));

        Column<double> close(10), open(10);
        for (std::size_t i = 0; i < 10; ++i) {
            close[i] = 100.0 + static_cast<double>(i);
            open[i]  =  95.0 + static_cast<double>(i);
        }
        md2.setColumn(Field::CLOSE, std::move(close));
        md2.setColumn(Field::OPEN, std::move(open));
    }

    auto clone2 = expr->clone();
    std::vector<double> outClone2(10);
    clone2->evaluate(md2, outClone2.data(), 10);

    // Verify: 100+i + 95+i = 195 + 2i
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(outClone2[i], 195.0 + 2.0 * static_cast<double>(i));
    }
}

// ============================================================
// Edge cases
// ============================================================

TEST_F(ExpressionTest, ZeroSizeEvaluation) {
    // Evaluate on a size-0 slice: should not crash
    // Use the fixture's md_ with n=0 — ColumnRef reads 0 bytes via memcpy
    ColumnRef ref(Field::CLOSE);
    std::vector<double> output(0);
    EXPECT_NO_THROW(ref.evaluate(md_, output.data(), 0));
}

TEST_F(ExpressionTest, RollingWithAllSameValues) {
    // Create MarketData where CLOSE is constant
    MarketData constMd;
    {
        std::vector<int64_t> ts(10);
        for (std::size_t i = 0; i < 10; ++i) {
            ts[i] = static_cast<int64_t>(i);
        }
        TimestampIndex tsi(ts.data(), 10);
        constMd = MarketData("CONST", std::move(tsi));

        Column<double> close(10);
        for (std::size_t i = 0; i < 10; ++i) {
            close[i] = 5.0;  // all same
        }
        constMd.setColumn(Field::CLOSE, std::move(close));
    }

    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    RollingExpr rolling(RollingOpCode::ROLLING_STD, 3, std::move(close));

    std::vector<double> output(10);
    rolling.evaluate(constMd, output.data(), 10);

    // First 2 NaN, rest should be 0.0 (no variance)
    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    for (std::size_t i = 2; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(output[i], 0.0);
    }
}

// ============================================================
// RedExpr tests
// ============================================================

TEST_F(ExpressionTest, RedSumBasic) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    RedExpr redSum(RedOpCode::RED_SUM, std::move(close));

    std::vector<double> output(kN);
    redSum.evaluate(md_, output.data(), kN);

    const auto& col = md_.column<double>(Field::CLOSE);
    double expectedSum = 0.0;
    for (std::size_t i = 0; i < kN; ++i) {
        expectedSum += col[i];
    }

    // RED_SUM broadcasts the same sum to every position
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], expectedSum);
    }
}

TEST_F(ExpressionTest, RedMeanBasic) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    RedExpr redMean(RedOpCode::RED_MEAN, std::move(close));

    std::vector<double> output(kN);
    redMean.evaluate(md_, output.data(), kN);

    const auto& col = md_.column<double>(Field::CLOSE);
    double sum = 0.0;
    for (std::size_t i = 0; i < kN; ++i) {
        sum += col[i];
    }
    double expectedMean = sum / static_cast<double>(kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], expectedMean);
    }
}

TEST_F(ExpressionTest, RedStdBasic) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    RedExpr redStd(RedOpCode::RED_STD, std::move(close));

    std::vector<double> output(kN);
    redStd.evaluate(md_, output.data(), kN);

    const auto& col = md_.column<double>(Field::CLOSE);
    double sum = 0.0, sumSq = 0.0;
    for (std::size_t i = 0; i < kN; ++i) {
        sum += col[i];
        sumSq += col[i] * col[i];
    }
    double mean = sum / static_cast<double>(kN);
    double var = sumSq / static_cast<double>(kN) - mean * mean;
    double expectedStd = std::sqrt(std::max(0.0, var));

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], expectedStd);
    }
}

TEST_F(ExpressionTest, RedMin) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    RedExpr redMin(RedOpCode::RED_MIN, std::move(close));

    std::vector<double> output(kN);
    redMin.evaluate(md_, output.data(), kN);

    const auto& col = md_.column<double>(Field::CLOSE);
    double expectedMin = col[0];
    for (std::size_t i = 1; i < kN; ++i) {
        if (col[i] < expectedMin) expectedMin = col[i];
    }

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], expectedMin);
    }
}

TEST_F(ExpressionTest, RedMax) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    RedExpr redMax(RedOpCode::RED_MAX, std::move(close));

    std::vector<double> output(kN);
    redMax.evaluate(md_, output.data(), kN);

    const auto& col = md_.column<double>(Field::CLOSE);
    double expectedMax = col[0];
    for (std::size_t i = 1; i < kN; ++i) {
        if (col[i] > expectedMax) expectedMax = col[i];
    }

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(output[i], expectedMax);
    }
}

TEST_F(ExpressionTest, RedClone) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    auto expr1 = std::make_unique<RedExpr>(RedOpCode::RED_STD, std::move(close));
    auto expr2 = expr1->clone();

    std::vector<double> out1(kN), out2(kN);
    expr1->evaluate(md_, out1.data(), kN);
    expr2->evaluate(md_, out2.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(out1[i], out2[i]);
    }
}

// ============================================================
// CsExpr tests — evaluate() on single MarketData now throws
// because cross-sectional operators require PanelData context.
// Use FactorCalculator::evaluateCSExpression() instead.
// ============================================================

TEST_F(ExpressionTest, CsExprEvaluateThrowsOnSingleAsset) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    CsExpr csRank(CsOpCode::CS_RANK, std::move(close));

    std::vector<double> output(kN);
    EXPECT_THROW(csRank.evaluate(md_, output.data(), kN), std::logic_error);
}

TEST_F(ExpressionTest, CsExprEvaluatePoolOverloadAlsoThrows) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    CsExpr csRank(CsOpCode::CS_RANK, std::move(close));

    std::vector<double> output(kN);
    BufferPool pool;
    EXPECT_THROW(csRank.evaluate(md_, output.data(), kN, &pool), std::logic_error);
}

TEST_F(ExpressionTest, CsClone) {
    // Clone should work (AST structure, not evaluation)
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    auto expr1 = std::make_unique<CsExpr>(CsOpCode::CS_RANK, std::move(close));
    auto expr2 = expr1->clone();

    // Both should throw on single-asset evaluation
    std::vector<double> out1(kN), out2(kN);
    EXPECT_THROW(expr1->evaluate(md_, out1.data(), kN), std::logic_error);
    EXPECT_THROW(expr2->evaluate(md_, out2.data(), kN), std::logic_error);
}

TEST_F(ExpressionTest, CsExprNodeInfo) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    CsExpr csRank(CsOpCode::CS_RANK, std::move(close));

    EXPECT_EQ(csRank.nodeCount(), 2u);  // CsExpr + ColumnRef
    EXPECT_EQ(csRank.maxDepth(), 1u);
    EXPECT_EQ(csRank.opCode(), CsOpCode::CS_RANK);
    EXPECT_NE(csRank.child(), nullptr);
}

TEST_F(ExpressionTest, CsExprExtraParams) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    std::vector<double> params = {0.1, 0.9};
    CsExpr csWinsorize(CsOpCode::CS_WINSORIZE, std::move(close), params);

    EXPECT_EQ(csWinsorize.opCode(), CsOpCode::CS_WINSORIZE);
    EXPECT_EQ(csWinsorize.extraParams().size(), 2u);
    EXPECT_DOUBLE_EQ(csWinsorize.extraParams()[0], 0.1);
    EXPECT_DOUBLE_EQ(csWinsorize.extraParams()[1], 0.9);
}

TEST_F(ExpressionTest, CsExprThrowsDescriptiveMessage) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    CsExpr csRank(CsOpCode::CS_RANK, std::move(close));

    std::vector<double> output(kN);
    try {
        csRank.evaluate(md_, output.data(), kN);
        FAIL() << "Expected std::logic_error";
    } catch (const std::logic_error& e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("cross-sectional"), std::string::npos);
        EXPECT_NE(msg.find("PanelData"), std::string::npos);
        EXPECT_NE(msg.find("evaluateCSExpression"), std::string::npos);
    }
}

TEST_F(ExpressionTest, CsExprThrowMessageIncludesOperatorName) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    CsExpr csZScore(CsOpCode::CS_ZSCORE, std::move(close));

    std::vector<double> output(kN);
    try {
        csZScore.evaluate(md_, output.data(), kN);
        FAIL() << "Expected std::logic_error";
    } catch (const std::logic_error& e) {
        std::string msg(e.what());
        EXPECT_NE(msg.find("cs_zscore"), std::string::npos);
    }
}

// ============================================================
// Parameterized CsExpr tests — verify they also throw
// ============================================================

TEST_F(ExpressionTest, CsWinsorizeThrowsOnSingleAsset) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    CsExpr csWinsorize(CsOpCode::CS_WINSORIZE, std::move(close),
                       {0.1, 0.9});

    std::vector<double> output(kN);
    EXPECT_THROW(csWinsorize.evaluate(md_, output.data(), kN), std::logic_error);
}

TEST_F(ExpressionTest, CsClipThrowsOnSingleAsset) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    CsExpr csClip(CsOpCode::CS_CLIP, std::move(close), {14.0, 20.0});

    std::vector<double> output(kN);
    EXPECT_THROW(csClip.evaluate(md_, output.data(), kN), std::logic_error);
}

// ============================================================
// Mixed expression tests (Red/Cs with Unary/Binary children)
// ============================================================

TEST_F(ExpressionTest, CompositeRedWithUnary) {
    // RED_ZSCORE(LOG(CLOSE))
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    auto logExpr = std::make_unique<UnaryExpr>(UnaryOpCode::LOG, std::move(close));
    RedExpr redZScore(RedOpCode::RED_ZSCORE, std::move(logExpr));

    std::vector<double> output(kN);
    redZScore.evaluate(md_, output.data(), kN);

    // Z-scores of log values should have mean ≈ 0
    double sum = 0.0;
    for (std::size_t i = 0; i < kN; ++i) {
        sum += output[i];
    }
    double mean = sum / static_cast<double>(kN);
    EXPECT_NEAR(mean, 0.0, 1e-12);
}

TEST_F(ExpressionTest, CompositeCsWithBinaryThrowsOnSingleAsset) {
    // CS_NORMALIZE(HIGH - LOW) — must use PanelData path, not single MarketData.
    auto high = std::make_unique<ColumnRef>(Field::HIGH);
    auto low  = std::make_unique<ColumnRef>(Field::LOW);
    auto diff = std::make_unique<BinaryExpr>(BinaryOpCode::SUB,
        std::move(high), std::move(low));
    CsExpr csNorm(CsOpCode::CS_NORMALIZE, std::move(diff));

    std::vector<double> output(kN);
    EXPECT_THROW(csNorm.evaluate(md_, output.data(), kN), std::logic_error);
}

// ============================================================
// Node count and depth tests
// ============================================================

TEST_F(ExpressionTest, NodeCountLeaf) {
    ColumnRef ref(Field::CLOSE);
    EXPECT_EQ(ref.nodeCount(), 1);

    Scalar s(3.14);
    EXPECT_EQ(s.nodeCount(), 1);
}

TEST_F(ExpressionTest, NodeCountComposite) {
    // LOG(CLOSE)
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    UnaryExpr logExpr(UnaryOpCode::LOG, std::move(close));
    EXPECT_EQ(logExpr.nodeCount(), 2);  // UnaryExpr + ColumnRef

    // CLOSE + OPEN
    auto lhs = std::make_unique<ColumnRef>(Field::CLOSE);
    auto rhs = std::make_unique<ColumnRef>(Field::OPEN);
    BinaryExpr add(BinaryOpCode::ADD, std::move(lhs), std::move(rhs));
    EXPECT_EQ(add.nodeCount(), 3);  // BinaryExpr + 2 ColumnRefs
}

TEST_F(ExpressionTest, MaxDepth) {
    // Leaf depth = 0
    ColumnRef ref(Field::CLOSE);
    EXPECT_EQ(ref.maxDepth(), 0);

    // LOG(CLOSE) depth = 1
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    UnaryExpr logExpr(UnaryOpCode::LOG, std::move(close));
    EXPECT_EQ(logExpr.maxDepth(), 1);

    // ABS(LOG(CLOSE)) depth = 2
    auto close2 = std::make_unique<ColumnRef>(Field::CLOSE);
    auto log2 = std::make_unique<UnaryExpr>(UnaryOpCode::LOG, std::move(close2));
    UnaryExpr absExpr(UnaryOpCode::ABS, std::move(log2));
    EXPECT_EQ(absExpr.maxDepth(), 2);
}

// ============================================================
// Dump / string representation tests
// ============================================================

TEST_F(ExpressionTest, DumpColumnRef) {
    ColumnRef ref(Field::CLOSE);
    std::string s = exprToString(&ref);
    EXPECT_TRUE(s.find("COLUMN") != std::string::npos);
    EXPECT_TRUE(s.find("close") != std::string::npos);
}

TEST_F(ExpressionTest, DumpScalar) {
    Scalar s(3.14);
    std::string str = exprToString(&s);
    EXPECT_TRUE(str.find("SCALAR") != std::string::npos);
    EXPECT_TRUE(str.find("3.14") != std::string::npos);
}

TEST_F(ExpressionTest, DumpUnaryExpr) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    UnaryExpr logExpr(UnaryOpCode::LOG, std::move(close));
    std::string str = exprToString(&logExpr);
    EXPECT_TRUE(str.find("log") != std::string::npos);
    EXPECT_TRUE(str.find("COLUMN") != std::string::npos);
}

TEST_F(ExpressionTest, DumpBinaryExpr) {
    auto lhs = std::make_unique<ColumnRef>(Field::CLOSE);
    auto rhs = std::make_unique<ColumnRef>(Field::OPEN);
    BinaryExpr add(BinaryOpCode::ADD, std::move(lhs), std::move(rhs));
    std::string str = exprToString(&add);
    EXPECT_TRUE(str.find("add") != std::string::npos);
}

TEST_F(ExpressionTest, DumpRollingExpr) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    RollingExpr rolling(RollingOpCode::ROLLING_MEAN, 5, std::move(close));
    std::string str = exprToString(&rolling);
    EXPECT_TRUE(str.find("rolling_mean") != std::string::npos);
    EXPECT_TRUE(str.find("5") != std::string::npos);
}

TEST_F(ExpressionTest, DumpRedExpr) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    RedExpr redMean(RedOpCode::RED_MEAN, std::move(close));
    std::string str = exprToString(&redMean);
    EXPECT_TRUE(str.find("red_mean") != std::string::npos);
    EXPECT_TRUE(str.find("COLUMN") != std::string::npos);
}

TEST_F(ExpressionTest, DumpCsExpr) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    CsExpr csRank(CsOpCode::CS_RANK, std::move(close));
    std::string str = exprToString(&csRank);
    EXPECT_TRUE(str.find("cs_rank") != std::string::npos);
    EXPECT_TRUE(str.find("COLUMN") != std::string::npos);
}

TEST_F(ExpressionTest, DumpParameterizedCsExpr) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    CsExpr csClip(CsOpCode::CS_CLIP, std::move(close), {-3.0, 3.0});
    std::string str = exprToString(&csClip);
    EXPECT_TRUE(str.find("cs_clip") != std::string::npos);
    EXPECT_TRUE(str.find("-3") != std::string::npos);
    EXPECT_TRUE(str.find("3") != std::string::npos);
}
