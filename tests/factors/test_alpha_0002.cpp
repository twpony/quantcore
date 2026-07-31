// test_alpha0002.cpp — unit tests for 5-day vs 20-day MA deviation factor
//
// Tests:
//   1. Factor registers and evaluates correctly
//   2. First 20 positions are NaN (20-day window boundary)
//   3. Positions 4-19 are NaN (5-day window boundary)
//   4. Correct values with constant prices (deviation ≈ 0)
//   5. Correct values with trending prices
//   6. Expression string is accessible

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "factors/alpha_0002.h"
#include "quantcore/core/FactorCalculator.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;
using namespace quantcore::factors;

// ============================================================
// Fixture: 50 rows of synthetic market data
// ============================================================

class Alpha0002Test : public ::testing::Test {
protected:
    static constexpr std::size_t kN = 50;  // > 20 for both windows

    void SetUp() override {
        std::vector<int64_t> timestamps(kN);
        for (std::size_t i = 0; i < kN; ++i)
            timestamps[i] = static_cast<int64_t>(i);
        TimestampIndex tsIdx(timestamps.data(), kN);
        md_ = MarketData("TEST", std::move(tsIdx));
    }

    /// Set CLOSE to linearly increasing values: close[i] = start + i * step
    void setLinearClose(double start = 10.0, double step = 1.0) {
        Column<double> close(kN);
        for (std::size_t i = 0; i < kN; ++i)
            close[i] = start + step * static_cast<double>(i);
        md_.setColumn(Field::CLOSE, std::move(close));
        allocateOtherFields();
    }

    /// Set CLOSE to all the same value
    void setConstantClose(double value = 100.0) {
        Column<double> close(kN);
        for (std::size_t i = 0; i < kN; ++i)
            close[i] = value;
        md_.setColumn(Field::CLOSE, std::move(close));
        allocateOtherFields();
    }

private:
    void allocateOtherFields() {
        for (std::size_t f = 0; f < static_cast<std::size_t>(Field::kFieldCount); ++f) {
            auto field = static_cast<Field>(f);
            if (field == Field::CLOSE) continue;
            Column<double> col(kN);
            for (std::size_t i = 0; i < kN; ++i)
                col[i] = static_cast<double>(i);
            md_.setColumn(field, std::move(col));
        }
    }

protected:
    MarketData md_;
    FactorCalculator calc_;
};

// ============================================================
// Tests
// ============================================================

TEST_F(Alpha0002Test, RegistersSuccessfully) {
    EXPECT_NO_THROW(registerAlpha0002(calc_));

    auto formulas = calc_.formulas();
    EXPECT_EQ(formulas.size(), 1);
    EXPECT_EQ(formulas[0], "alpha_0002");
}

TEST_F(Alpha0002Test, ExpressionStringIsCorrect) {
    registerAlpha0002(calc_);

    std::string expr = calc_.formulaExpression(kAlpha0002);
    EXPECT_EQ(expr, kAlpha0002Expr);
    EXPECT_NE(expr.find("rolling_mean"), std::string::npos);
    EXPECT_NE(expr.find("close"), std::string::npos);
}

TEST_F(Alpha0002Test, BoundaryPositionsAreNaN) {
    setLinearClose();
    registerAlpha0002(calc_);

    auto result = calc_.evaluate(kAlpha0002, md_);

    ASSERT_EQ(result.size(), kN);

    // First 19 positions (0..18) should be NaN
    // rolling_mean(close, 5):  NaN at 0..3  (window-1 = 4 positions)
    // rolling_mean(close, 20): NaN at 0..18 (window-1 = 19 positions)
    // The combined result is NaN where EITHER operand is NaN:
    //   positions 0..3:  both NaN
    //   positions 4..18: MA5 valid, MA20 NaN → result NaN
    //   position 19+:    both valid → result valid
    for (std::size_t i = 0; i < 19; ++i) {
        EXPECT_TRUE(std::isnan(result[i]))
            << "Position " << i << " should be NaN (window boundary)";
    }

    // Position 19+ should NOT be NaN
    for (std::size_t i = 19; i < kN; ++i) {
        EXPECT_FALSE(std::isnan(result[i]))
            << "Position " << i << " should be valid";
    }
}

TEST_F(Alpha0002Test, ConstantPriceReturnsNearZero) {
    // When price never changes, MA(5) == MA(20) == constant,
    // so the ratio - 1 ≈ 0
    setConstantClose(100.0);
    registerAlpha0002(calc_);

    auto result = calc_.evaluate(kAlpha0002, md_);

    for (std::size_t i = 19; i < kN; ++i) {
        EXPECT_NEAR(result[i], 0.0, 1e-10)
            << "Position " << i
            << ": constant price → MA5/MA20 = 1 → return ≈ 0";
    }
}

TEST_F(Alpha0002Test, LinearTrendingPrice) {
    // close[i] = 100 + i
    // MA5 at position i:  avg of close[i-4..i]
    // MA20 at position i: avg of close[i-19..i]
    //
    // For i = 25:
    //   MA5  = avg(122,123,124,125,126) = 124.0
    //   MA20 = avg(107..126) = 116.5
    //   result = 124.0 / 116.5 - 1 ≈ 0.064377...
    setLinearClose(100.0, 1.0);
    registerAlpha0002(calc_);

    auto result = calc_.evaluate(kAlpha0002, md_);

    // Position 25: manual verification
    double sum5 = 0.0, sum20 = 0.0;
    for (std::size_t j = 25 - 4; j <= 25; ++j)
        sum5 += (100.0 + static_cast<double>(j));
    for (std::size_t j = 25 - 19; j <= 25; ++j)
        sum20 += (100.0 + static_cast<double>(j));
    double ma5 = sum5 / 5.0;
    double ma20 = sum20 / 20.0;
    double expected25 = ma5 / ma20 - 1.0;
    EXPECT_NEAR(result[25], expected25, 1e-10);

    // Position 35: manual verification
    double sum5_35 = 0.0, sum20_35 = 0.0;
    for (std::size_t j = 35 - 4; j <= 35; ++j)
        sum5_35 += (100.0 + static_cast<double>(j));
    for (std::size_t j = 35 - 19; j <= 35; ++j)
        sum20_35 += (100.0 + static_cast<double>(j));
    double ma5_35 = sum5_35 / 5.0;
    double ma20_35 = sum20_35 / 20.0;
    double expected35 = ma5_35 / ma20_35 - 1.0;
    EXPECT_NEAR(result[35], expected35, 1e-10);
}

TEST_F(Alpha0002Test, ConvenienceEvaluateFunction) {
    setLinearClose(50.0, 2.0);
    registerAlpha0002(calc_);

    auto result = evaluateAlpha0002(calc_, md_);

    ASSERT_EQ(result.size(), kN);
    // Position 19+ should be valid
    EXPECT_FALSE(std::isnan(result[19]));
    // Position 0-18 should be NaN
    for (std::size_t i = 0; i < 19; ++i)
        EXPECT_TRUE(std::isnan(result[i]));
}

TEST_F(Alpha0002Test, FormulaStringConstantAccessible) {
    EXPECT_EQ(kAlpha0002, "alpha_0002");
    EXPECT_EQ(kAlpha0002Expr,
              "rolling_mean(close, 5) / rolling_mean(close, 20) - 1");
}
