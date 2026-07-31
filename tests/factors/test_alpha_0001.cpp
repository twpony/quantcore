// test_alpha0001.cpp — unit tests for 20-day return factor
//
// Tests:
//   1. Factor registers and evaluates correctly
//   2. First 20 positions are NaN (shift boundary)
//   3. Positions 20+ have correct values
//   4. Works with constant growth (should produce constant returns)
//   5. Works with varying prices
//   6. Expression string is accessible

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "factors/alpha_0001.h"
#include "quantcore/core/FactorCalculator.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;
using namespace quantcore::factors;

// ============================================================
// Fixture: 30 rows of synthetic market data
// ============================================================

class Alpha0001Test : public ::testing::Test {
protected:
    static constexpr std::size_t kN = 30;  // need > 20 for window

    void SetUp() override {
        std::vector<int64_t> timestamps(kN);
        for (std::size_t i = 0; i < kN; ++i)
            timestamps[i] = static_cast<int64_t>(i);
        TimestampIndex tsIdx(timestamps.data(), kN);
        md_ = MarketData("TEST", std::move(tsIdx));
    }

    /// Set CLOSE to a constant 1% daily growth: close[i] = base * (1.01)^i
    void setConstantGrowthClose(double base = 100.0) {
        Column<double> close(kN);
        for (std::size_t i = 0; i < kN; ++i)
            close[i] = base * std::pow(1.01, static_cast<double>(i));
        md_.setColumn(Field::CLOSE, std::move(close));

        // Fill other required fields with dummy values
        allocateOtherFields();
    }

    /// Set CLOSE to linearly increasing values: close[i] = start + i * step
    void setLinearClose(double start = 10.0, double step = 1.0) {
        Column<double> close(kN);
        for (std::size_t i = 0; i < kN; ++i)
            close[i] = start + step * static_cast<double>(i);
        md_.setColumn(Field::CLOSE, std::move(close));

        allocateOtherFields();
    }

private:
    void allocateOtherFields() {
        for (std::size_t f = 0; f < static_cast<std::size_t>(Field::kFieldCount); ++f) {
            auto field = static_cast<Field>(f);
            if (field == Field::CLOSE) continue;  // already set
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

TEST_F(Alpha0001Test, RegistersSuccessfully) {
    EXPECT_NO_THROW(registerAlpha0001(calc_));

    auto formulas = calc_.formulas();
    EXPECT_EQ(formulas.size(), 1);
    EXPECT_EQ(formulas[0], "alpha_0001");
}

TEST_F(Alpha0001Test, ExpressionStringIsCorrect) {
    registerAlpha0001(calc_);

    std::string expr = calc_.formulaExpression(kAlpha0001);
    EXPECT_EQ(expr, kAlpha0001Expr);
    EXPECT_NE(expr.find("rolling_shift"), std::string::npos);
    EXPECT_NE(expr.find("close"), std::string::npos);
}

TEST_F(Alpha0001Test, BoundaryPositionsAreNaN) {
    setConstantGrowthClose();
    registerAlpha0001(calc_);

    auto result = calc_.evaluate(kAlpha0001, md_);

    ASSERT_EQ(result.size(), kN);

    // First 20 positions (0..19) should be NaN
    for (std::size_t i = 0; i < 20; ++i) {
        EXPECT_TRUE(std::isnan(result[i]))
            << "Position " << i << " should be NaN (shift boundary)";
    }

    // Position 20+ should NOT be NaN
    for (std::size_t i = 20; i < kN; ++i) {
        EXPECT_FALSE(std::isnan(result[i]))
            << "Position " << i << " should be valid";
    }
}

TEST_F(Alpha0001Test, ConstantGrowthReturns) {
    // 1% daily growth → 20-day return = 1.01^20 - 1 ≈ 0.22019
    setConstantGrowthClose();
    registerAlpha0001(calc_);

    auto result = calc_.evaluate(kAlpha0001, md_);

    double expected = std::pow(1.01, 20.0) - 1.0;  // ≈ 0.22019

    for (std::size_t i = 20; i < kN; ++i) {
        EXPECT_NEAR(result[i], expected, 1e-10)
            << "Position " << i << ": constant growth → constant return";
    }
}

TEST_F(Alpha0001Test, LinearPriceReturns) {
    // close[i] = 10 + i
    // 20-day return at i: close[i]/close[i-20] - 1 = (10+i)/(10+i-20) - 1
    setLinearClose(10.0, 1.0);
    registerAlpha0001(calc_);

    auto result = calc_.evaluate(kAlpha0001, md_);

    // Position 0-19: NaN
    for (std::size_t i = 0; i < 20; ++i)
        EXPECT_TRUE(std::isnan(result[i]));

    // Position 25: (10+25)/(10+5) - 1 = 35/15 - 1 = 1.333...
    double expected = (10.0 + 25.0) / (10.0 + 5.0) - 1.0;
    EXPECT_NEAR(result[25], expected, 1e-10);

    // Position 29: (10+29)/(10+9) - 1 = 39/19 - 1 ≈ 1.0526
    double expected2 = (10.0 + 29.0) / (10.0 + 9.0) - 1.0;
    EXPECT_NEAR(result[29], expected2, 1e-10);
}

TEST_F(Alpha0001Test, ConvenienceEvaluateFunction) {
    setLinearClose(100.0, 2.0);
    registerAlpha0001(calc_);

    auto result = evaluateAlpha0001(calc_, md_);

    ASSERT_EQ(result.size(), kN);
    // Position 25: (100+50)/(100+10) - 1 = 150/110 - 1 ≈ 0.3636
    double expected = (100.0 + 50.0) / (100.0 + 10.0) - 1.0;
    EXPECT_NEAR(result[25], expected, 1e-10);
}

TEST_F(Alpha0001Test, FormulaStringConstantAccessible) {
    // Verify the formula string is accessible without a calculator
    EXPECT_EQ(kAlpha0001, "alpha_0001");
    EXPECT_EQ(kAlpha0001Expr, "close / rolling_shift(close, 20) - 1");
}
