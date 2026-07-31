// test_full_pipeline.cpp — end-to-end integration tests
//
// Tests the complete pipeline: string expression → Lexer → Parser → AST
// → ExecutionEngine → Column<double>.  Each formula is parsed from a
// string and evaluated, with results verified against hand-computed values.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "quantcore/core/FactorCalculator.h"
#include "quantcore/core/Types.h"
#include "quantcore/engine/ExecutionEngine.h"
#include "quantcore/expression/Parser.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;

// ============================================================
// Test fixture — provides MarketData with known values
// ============================================================

class FullPipelineTest : public ::testing::Test {
protected:
    static constexpr std::size_t kN = 20;

    void SetUp() override {
        std::vector<int64_t> timestamps(kN);
        for (std::size_t i = 0; i < kN; ++i)
            timestamps[i] = static_cast<int64_t>(i);
        TimestampIndex tsIdx(timestamps.data(), kN);
        md_ = MarketData("TEST", std::move(tsIdx));

        for (std::size_t f = 0; f < static_cast<std::size_t>(Field::kFieldCount); ++f) {
            auto field = static_cast<Field>(f);
            Column<double> col(kN);
            double base = 0.0, scale = 1.0;
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
            for (std::size_t i = 0; i < kN; ++i)
                col[i] = base + scale * static_cast<double>(i);
            md_.setColumn(field, std::move(col));
        }
    }

    /// Parse and evaluate an expression string.
    Column<double> run(const std::string& expr) {
        return calc_.evaluateExpression(expr, md_);
    }

    MarketData md_;
    FactorCalculator calc_;
};

// ============================================================
// Single-field expressions
// ============================================================

TEST_F(FullPipelineTest, SingleField) {
    auto result = run("close");
    ASSERT_EQ(result.size(), kN);
    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i]);
}

TEST_F(FullPipelineTest, ScalarLiteral) {
    auto result = run("42.0");
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], 42.0);
}

// ============================================================
// Unary expressions
// ============================================================

TEST_F(FullPipelineTest, Log) {
    auto result = run("log(close)");
    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::log(c[i]));
}

TEST_F(FullPipelineTest, Abs) {
    auto result = run("abs(close)");
    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::abs(c[i]));
}

TEST_F(FullPipelineTest, Sqrt) {
    auto result = run("sqrt(close)");
    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::sqrt(c[i]));
}

TEST_F(FullPipelineTest, ChainedUnary) {
    auto result = run("sqrt(abs(log(close)))");
    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::sqrt(std::abs(std::log(c[i]))));
}

// ============================================================
// Binary arithmetic expressions
// ============================================================

TEST_F(FullPipelineTest, Add) {
    auto result = run("close + open");
    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i] + o[i]);
}

TEST_F(FullPipelineTest, Sub) {
    auto result = run("high - low");
    const auto& h = md_.column<double>(Field::HIGH);
    const auto& l = md_.column<double>(Field::LOW);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], h[i] - l[i]);
}

TEST_F(FullPipelineTest, Mul) {
    auto result = run("close * volume");
    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& v = md_.column<double>(Field::VOLUME);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i] * v[i]);
}

TEST_F(FullPipelineTest, Div) {
    auto result = run("close / open");
    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i] / o[i]);
}

// ============================================================
// Operator precedence
// ============================================================

TEST_F(FullPipelineTest, PrecedenceMulBeforeAdd) {
    auto result = run("close + open * 2.0");
    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i] + o[i] * 2.0);
}

TEST_F(FullPipelineTest, PrecedenceParentheses) {
    auto result = run("(close + open) * 2.0");
    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], (c[i] + o[i]) * 2.0);
}

// ============================================================
// Unary minus
// ============================================================

TEST_F(FullPipelineTest, UnaryMinus) {
    auto result = run("-close");
    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], -c[i]);
}

TEST_F(FullPipelineTest, UnaryMinusNumber) {
    auto result = run("-3.14");
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], -3.14);
}

// ============================================================
// Binary function form
// ============================================================

TEST_F(FullPipelineTest, MaxFunction) {
    auto result = run("max(high, low)");
    const auto& h = md_.column<double>(Field::HIGH);
    const auto& l = md_.column<double>(Field::LOW);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::max(h[i], l[i]));
}

TEST_F(FullPipelineTest, MinFunction) {
    auto result = run("min(high, low)");
    const auto& h = md_.column<double>(Field::HIGH);
    const auto& l = md_.column<double>(Field::LOW);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::min(h[i], l[i]));
}

// ============================================================
// Composite expressions
// ============================================================

TEST_F(FullPipelineTest, CompositeArithmetic) {
    auto result = run("(high - low) / close");
    const auto& h = md_.column<double>(Field::HIGH);
    const auto& l = md_.column<double>(Field::LOW);
    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], (h[i] - l[i]) / c[i]);
}

TEST_F(FullPipelineTest, CompositeWithUnary) {
    auto result = run("abs(log(close) - log(vwap)) * volume");
    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& v = md_.column<double>(Field::VWAP);
    const auto& vol = md_.column<double>(Field::VOLUME);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i],
            std::abs(std::log(c[i]) - std::log(v[i])) * vol[i]);
}

// ============================================================
// Rolling expressions
// ============================================================

TEST_F(FullPipelineTest, RollingMean) {
    auto result = run("rolling_mean(close, 5)");
    const auto& c = md_.column<double>(Field::CLOSE);

    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_TRUE(std::isnan(result[i]));
    double expected = (c[0] + c[1] + c[2] + c[3] + c[4]) / 5.0;
    EXPECT_DOUBLE_EQ(result[4], expected);
}

TEST_F(FullPipelineTest, RollingStd) {
    auto result = run("rolling_std(close, 5)");
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_TRUE(std::isnan(result[i]));
}

TEST_F(FullPipelineTest, RollingSum) {
    auto result = run("rolling_sum(close, 3)");
    const auto& c = md_.column<double>(Field::CLOSE);
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < 2; ++i)
        EXPECT_TRUE(std::isnan(result[i]));
    EXPECT_DOUBLE_EQ(result[2], c[0] + c[1] + c[2]);
}

// ============================================================
// Red expressions
// ============================================================

TEST_F(FullPipelineTest, RedMean) {
    auto result = run("red_mean(close)");
    ASSERT_EQ(result.size(), kN);
    const auto& c = md_.column<double>(Field::CLOSE);
    double sum = 0.0;
    for (std::size_t i = 0; i < kN; ++i) sum += c[i];
    double expected = sum / static_cast<double>(kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], expected);
}

TEST_F(FullPipelineTest, RedSum) {
    auto result = run("red_sum(close)");
    ASSERT_EQ(result.size(), kN);
    const auto& c = md_.column<double>(Field::CLOSE);
    double sum = 0.0;
    for (std::size_t i = 0; i < kN; ++i) sum += c[i];
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], sum);
}

// ============================================================
// Complex real-world factor formulas
// ============================================================

TEST_F(FullPipelineTest, MomentumFactor) {
    auto result = run("close / rolling_mean(close, 5) - 1");
    const auto& c = md_.column<double>(Field::CLOSE);

    ASSERT_EQ(result.size(), kN);
    // First 4 positions: rolling_mean is NaN → result should be NaN
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_TRUE(std::isnan(result[i]));
    // Position 4: rolling_mean = (c0+c1+c2+c3+c4)/5
    double mean4 = (c[0] + c[1] + c[2] + c[3] + c[4]) / 5.0;
    EXPECT_DOUBLE_EQ(result[4], c[4] / mean4 - 1.0);
}

TEST_F(FullPipelineTest, TurnoverFactor) {
    auto result = run("abs(log(close) - log(open)) * volume");
    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    const auto& v = md_.column<double>(Field::VOLUME);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i],
            std::abs(std::log(c[i]) - std::log(o[i])) * v[i]);
}

TEST_F(FullPipelineTest, NormalizedSpread) {
    auto result = run("(high - low) / close");
    const auto& h = md_.column<double>(Field::HIGH);
    const auto& l = md_.column<double>(Field::LOW);
    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], (h[i] - l[i]) / c[i]);
}

TEST_F(FullPipelineTest, LogReturn) {
    auto result = run("log(close) - log(open)");
    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::log(c[i]) - std::log(o[i]));
}

// ============================================================
// Error cases
// ============================================================

TEST_F(FullPipelineTest, UnknownFieldThrows) {
    EXPECT_THROW(run("unknown_field"), std::runtime_error);
}

TEST_F(FullPipelineTest, UnknownFunctionThrows) {
    EXPECT_THROW(run("unknown_func(close)"), std::runtime_error);
}

TEST_F(FullPipelineTest, UnterminatedParenthesis) {
    EXPECT_THROW(run("(close + open"), std::runtime_error);
}

// ============================================================
// Formula references via FactorCalculator
// ============================================================

TEST_F(FullPipelineTest, FormulaRefEndToEnd) {
    calc_.registerFormula("log_c", "log(close)");
    calc_.registerFormula("log_o", "log(open)");
    calc_.registerFormula("log_return", "$log_c - $log_o");

    auto result = calc_.evaluate("log_return", md_);
    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::log(c[i]) - std::log(o[i]));
}

// ============================================================
// Fusion stress test: deep unary chain
// ============================================================

TEST_F(FullPipelineTest, DeepUnaryChain) {
    // log(abs(sqrt(exp(neg(sign(square(close)))))))
    // The fused loop should handle this without intermediate allocations
    auto result = run("log(abs(sqrt(close)))");
    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i) {
        double expected = std::log(std::abs(std::sqrt(c[i])));
        if (c[i] <= 0.0)
            EXPECT_TRUE(std::isnan(result[i]));
        else
            EXPECT_DOUBLE_EQ(result[i], expected);
    }
}

TEST_F(FullPipelineTest, BinaryWithUnaryChains) {
    // LOG(CLOSE) + LOG(VOLUME) — fused as binary with unary chains on both sides
    auto result = run("log(close) + log(volume)");
    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& v = md_.column<double>(Field::VOLUME);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::log(c[i]) + std::log(v[i]));
}
