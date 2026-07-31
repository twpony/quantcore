// test_alpha_1001.cpp — unit tests for SIMD 20-day SMA factor
//
// Tests:
//   1. computeSma20() standalone — linear data
//   2. computeSma20() standalone — NaN boundary
//   3. computeSma20() standalone — constant values
//   4. computeSma20() standalone — too-short input
//   5. computeSma20() standalone — exactly window size
//   6. computeSma20() standalone — long sequence accuracy
//   7. registerAlpha1001() + calc.evaluate() (aligned API)
//   8. evaluateAlpha1001() convenience wrapper
//   9. formulas() lists alpha_1001 after registration
//  10. formulaExpression() returns description
//  11. isCustomFactor() returns true
//  12. registerCustomFactor via FactorCalculator directly

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "factors/alpha_1001.h"
#include "quantcore/core/FactorCalculator.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;
using namespace quantcore::factors;

// ============================================================
// Standalone computeSma20() tests
// ============================================================

TEST(Alpha1001StandaloneTest, LinearPriceSequence) {
    constexpr std::size_t kN = 30;
    Column<double> close(kN);
    for (std::size_t i = 0; i < kN; ++i)
        close[i] = 100.0 + static_cast<double>(i);

    auto result = computeSma20(close);

    ASSERT_EQ(result.size(), kN);

    // Positions 0..18 NaN
    for (std::size_t i = 0; i < 19; ++i)
        EXPECT_TRUE(std::isnan(result[i])) << "Position " << i;

    // Position 19: mean(close[0..19]) = mean(100..119) = 109.5
    double sum = 0.0;
    for (std::size_t i = 0; i < 20; ++i) sum += close[i];
    EXPECT_DOUBLE_EQ(result[19], sum / 20.0);

    // Position 25: mean(close[6..25]) = mean(106..125) = 115.5
    sum = 0.0;
    for (std::size_t i = 6; i < 26; ++i) sum += close[i];
    EXPECT_DOUBLE_EQ(result[25], sum / 20.0);

    // Position 29: mean(close[10..29])
    sum = 0.0;
    for (std::size_t i = 10; i < 30; ++i) sum += close[i];
    EXPECT_DOUBLE_EQ(result[29], sum / 20.0);
}

TEST(Alpha1001StandaloneTest, ConstantValues) {
    constexpr std::size_t kN = 50;
    Column<double> close(kN, 42.0);

    auto result = computeSma20(close);

    for (std::size_t i = 0; i < 19; ++i)
        EXPECT_TRUE(std::isnan(result[i]));
    for (std::size_t i = 19; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], 42.0);
}

TEST(Alpha1001StandaloneTest, InputTooShort) {
    Column<double> close(10);
    for (std::size_t i = 0; i < 10; ++i)
        close[i] = static_cast<double>(i);

    auto result = computeSma20(close);
    ASSERT_EQ(result.size(), 10);
    for (std::size_t i = 0; i < 10; ++i)
        EXPECT_TRUE(std::isnan(result[i]));
}

TEST(Alpha1001StandaloneTest, ExactlyWindowSize) {
    constexpr std::size_t kN = 20;
    Column<double> close(kN);
    for (std::size_t i = 0; i < kN; ++i)
        close[i] = static_cast<double>(i + 1);  // 1..20

    auto result = computeSma20(close);

    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < 19; ++i)
        EXPECT_TRUE(std::isnan(result[i]));
    EXPECT_DOUBLE_EQ(result[19], 10.5);  // mean(1..20)
}

TEST(Alpha1001StandaloneTest, RunningSumAccuracy) {
    constexpr std::size_t kN = 1000;
    Column<double> close(kN);
    for (std::size_t i = 0; i < kN; ++i)
        close[i] = 1.0 + static_cast<double>(i) * 0.01;

    auto result = computeSma20(close);

    for (std::size_t i = 19; i < kN; i += 50) {
        double expected = 0.0;
        for (std::size_t j = i - 19; j <= i; ++j)
            expected += close[j];
        expected /= 20.0;
        EXPECT_NEAR(result[i], expected, 1e-12) << "Position " << i;
    }
}

// ============================================================
// Integration tests — aligned with expression factor API
// ============================================================

class Alpha1001IntegrationTest : public ::testing::Test {
protected:
    static constexpr std::size_t kN = 50;

    void SetUp() override {
        std::vector<int64_t> timestamps(kN);
        for (std::size_t i = 0; i < kN; ++i)
            timestamps[i] = static_cast<int64_t>(i);
        TimestampIndex tsIdx(timestamps.data(), kN);
        md_ = MarketData("TEST", std::move(tsIdx));

        Column<double> close(kN);
        for (std::size_t i = 0; i < kN; ++i)
            close[i] = 100.0 + static_cast<double>(i);
        md_.setColumn(Field::CLOSE, std::move(close));

        for (std::size_t f = 0; f < static_cast<std::size_t>(Field::kFieldCount); ++f) {
            auto field = static_cast<Field>(f);
            if (field == Field::CLOSE) continue;
            Column<double> col(kN, 0.0);
            md_.setColumn(field, std::move(col));
        }
    }

    MarketData md_;
    FactorCalculator calc_;
};

TEST_F(Alpha1001IntegrationTest, RegisterAndEvaluateAlignedApi) {
    // Same usage pattern as expression-based factors
    registerAlpha1001(calc_);

    // evaluate() works — dispatches to SIMD computeSma20() via
    // the custom evaluator, NOT the expression engine
    auto result = calc_.evaluate("alpha_1001", md_);

    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < 19; ++i)
        EXPECT_TRUE(std::isnan(result[i]));

    const auto& c = md_.column<double>(Field::CLOSE);
    double sum = 0.0;
    for (std::size_t j = 0; j < 20; ++j) sum += c[j];
    EXPECT_DOUBLE_EQ(result[19], sum / 20.0);
}

TEST_F(Alpha1001IntegrationTest, FormulasListsFactor) {
    registerAlpha1001(calc_);

    auto names = calc_.formulas();
    EXPECT_EQ(names.size(), 1);
    EXPECT_EQ(names[0], "alpha_1001");
}

TEST_F(Alpha1001IntegrationTest, FormulaExpressionReturnsDescription) {
    registerAlpha1001(calc_);

    std::string desc = calc_.formulaExpression("alpha_1001");
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("SMA"), std::string::npos);
}

TEST_F(Alpha1001IntegrationTest, IsCustomFactorReturnsTrue) {
    registerAlpha1001(calc_);
    EXPECT_TRUE(calc_.isCustomFactor("alpha_1001"));
}

TEST_F(Alpha1001IntegrationTest, ConvenienceWrapper) {
    registerAlpha1001(calc_);

    auto result = evaluateAlpha1001(calc_, md_);

    ASSERT_EQ(result.size(), kN);
    EXPECT_FALSE(std::isnan(result[25]));
}

TEST_F(Alpha1001IntegrationTest, EvaluateBatchWorks) {
    registerAlpha1001(calc_);
    // Also register an expression factor to test mixed batch
    calc_.registerFormula("just_close", "close");

    auto results = calc_.evaluateBatch({"alpha_1001", "just_close"}, md_);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].size(), kN);  // alpha_1001
    EXPECT_EQ(results[1].size(), kN);  // just_close
}

TEST_F(Alpha1001IntegrationTest, UnregisterWorks) {
    registerAlpha1001(calc_);
    EXPECT_EQ(calc_.formulas().size(), 1);

    calc_.unregisterFormula("alpha_1001");
    EXPECT_EQ(calc_.formulas().size(), 0);
    EXPECT_FALSE(calc_.isCustomFactor("alpha_1001"));

    EXPECT_THROW(calc_.evaluate("alpha_1001", md_), std::runtime_error);
}

// ============================================================
// FactorCalculator::registerCustomFactor direct tests
// ============================================================

TEST_F(Alpha1001IntegrationTest, RegisterCustomFactorDirectly) {
    calc_.registerCustomFactor(
        "my_sma",
        [](const MarketData& md) -> Column<double> {
            const auto& c = md.column<double>(Field::CLOSE);
            Column<double> out(c.size());
            for (std::size_t i = 0; i < c.size(); ++i)
                out[i] = c[i] * 2.0;
            return out;
        },
        "double close");

    EXPECT_TRUE(calc_.isCustomFactor("my_sma"));

    auto result = calc_.evaluate("my_sma", md_);
    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i] * 2.0);
}

TEST_F(Alpha1001IntegrationTest, NullEvaluatorThrows) {
    EXPECT_THROW(calc_.registerCustomFactor("bad", nullptr, "desc"),
                 std::runtime_error);
}

// ============================================================
// Constants
// ============================================================

TEST(Alpha1001ConstantsTest, NameAndDescription) {
    EXPECT_EQ(kAlpha1001, "alpha_1001");
    EXPECT_FALSE(kAlpha1001Desc.empty());
    EXPECT_NE(kAlpha1001Desc.find("SMA"), std::string::npos);
    EXPECT_NE(kAlpha1001Desc.find("20"), std::string::npos);
}
