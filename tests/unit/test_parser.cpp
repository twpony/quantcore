// test_parser.cpp — unit tests for Lexer, Parser, and parseExpression()
// Phase: 三期必实现
//
// Covers: tokenization, all grammar productions, name resolution for all
// 5 operator families, composite expressions, error messages, and
// end-to-end evaluation via ExecutionEngine.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/engine/ExecutionEngine.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/expression/ExprTraits.h"
#include "quantcore/expression/Token.h"
#include "quantcore/expression/Lexer.h"
#include "quantcore/expression/FusedLoopGenerator.h"
#include "quantcore/expression/Parser.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;

// ============================================================
// Lexer tests
// ============================================================

TEST(LexerTest, Numbers) {
    Lexer lexer;

    {
        auto tokens = lexer.tokenize("42");
        ASSERT_EQ(tokens.size(), 2);  // NUMBER + END
        EXPECT_EQ(tokens[0].type, TokenType::NUMBER);
        EXPECT_DOUBLE_EQ(tokens[0].numberValue, 42.0);
    }
    {
        auto tokens = lexer.tokenize("3.14");
        ASSERT_EQ(tokens.size(), 2);
        EXPECT_EQ(tokens[0].type, TokenType::NUMBER);
        EXPECT_DOUBLE_EQ(tokens[0].numberValue, 3.14);
    }
    {
        auto tokens = lexer.tokenize("1e-3");
        ASSERT_EQ(tokens.size(), 2);
        EXPECT_EQ(tokens[0].type, TokenType::NUMBER);
        EXPECT_DOUBLE_EQ(tokens[0].numberValue, 0.001);
    }
    {
        auto tokens = lexer.tokenize("0.5");
        ASSERT_EQ(tokens.size(), 2);
        EXPECT_DOUBLE_EQ(tokens[0].numberValue, 0.5);
    }
}

TEST(LexerTest, Identifiers) {
    Lexer lexer;

    {
        auto tokens = lexer.tokenize("CLOSE");
        ASSERT_EQ(tokens.size(), 2);
        EXPECT_EQ(tokens[0].type, TokenType::IDENTIFIER);
        EXPECT_EQ(tokens[0].text, "close");  // lowercased
    }
    {
        auto tokens = lexer.tokenize("rolling_mean");
        ASSERT_EQ(tokens.size(), 2);
        EXPECT_EQ(tokens[0].type, TokenType::IDENTIFIER);
        EXPECT_EQ(tokens[0].text, "rolling_mean");
    }
    {
        auto tokens = lexer.tokenize("Abs");  // case insensitive
        ASSERT_EQ(tokens.size(), 2);
        EXPECT_EQ(tokens[0].text, "abs");
    }
}

TEST(LexerTest, Operators) {
    Lexer lexer;
    auto tokens = lexer.tokenize("( ) + - * / ,");

    ASSERT_EQ(tokens.size(), 8);  // 7 ops + END
    EXPECT_EQ(tokens[0].type, TokenType::LPAREN);
    EXPECT_EQ(tokens[1].type, TokenType::RPAREN);
    EXPECT_EQ(tokens[2].type, TokenType::PLUS);
    EXPECT_EQ(tokens[3].type, TokenType::MINUS);
    EXPECT_EQ(tokens[4].type, TokenType::STAR);
    EXPECT_EQ(tokens[5].type, TokenType::SLASH);
    EXPECT_EQ(tokens[6].type, TokenType::COMMA);
    EXPECT_EQ(tokens[7].type, TokenType::END);
}

TEST(LexerTest, WhitespaceSkipping) {
    Lexer lexer;
    auto tokens = lexer.tokenize("  CLOSE \t +\n 3.14  ");

    ASSERT_EQ(tokens.size(), 4);  // IDENTIFIER + PLUS + NUMBER + END
    EXPECT_EQ(tokens[0].text, "close");
    EXPECT_EQ(tokens[1].type, TokenType::PLUS);
    EXPECT_DOUBLE_EQ(tokens[2].numberValue, 3.14);
}

TEST(LexerTest, PositionTracking) {
    Lexer lexer;
    auto tokens = lexer.tokenize("  abs(100)");
    // Positions: 'a' at 2, '(' at 5, '100' at 6, ')' at 9
    EXPECT_EQ(tokens[0].pos, 2);  // "abs"
    EXPECT_EQ(tokens[1].pos, 5);  // "("
    EXPECT_EQ(tokens[2].pos, 6);  // "100"
    EXPECT_EQ(tokens[3].pos, 9);  // ")"
}

TEST(LexerTest, InvalidCharacter) {
    Lexer lexer;
    EXPECT_THROW(lexer.tokenize("CLOSE @ OPEN"), std::runtime_error);
}

// ============================================================
// Parser — column references
// ============================================================

TEST(ParserTest, ColumnRefs) {
    EXPECT_NO_THROW(parseExpression("close"));
    EXPECT_NO_THROW(parseExpression("open"));
    EXPECT_NO_THROW(parseExpression("high"));
    EXPECT_NO_THROW(parseExpression("low"));
    EXPECT_NO_THROW(parseExpression("volume"));
    EXPECT_NO_THROW(parseExpression("amount"));
    EXPECT_NO_THROW(parseExpression("vwap"));
}

TEST(ParserTest, ColumnRefCaseInsensitive) {
    EXPECT_NO_THROW(parseExpression("CLOSE"));
    EXPECT_NO_THROW(parseExpression("Close"));
    EXPECT_NO_THROW(parseExpression("clOsE"));
}

TEST(ParserTest, UnknownIdentifier) {
    EXPECT_THROW(parseExpression("UNKNOWN"), std::runtime_error);
}

// ============================================================
// Parser — scalars
// ============================================================

TEST(ParserTest, ScalarPositive) {
    auto expr = parseExpression("3.14");
    auto* s = dynamic_cast<Scalar*>(expr.get());
    ASSERT_NE(s, nullptr);
    EXPECT_DOUBLE_EQ(s->value(), 3.14);
}

TEST(ParserTest, ScalarNegative) {
    auto expr = parseExpression("-5.0");
    auto* s = dynamic_cast<Scalar*>(expr.get());
    ASSERT_NE(s, nullptr);
    EXPECT_DOUBLE_EQ(s->value(), -5.0);
}

TEST(ParserTest, ScalarInteger) {
    auto expr = parseExpression("100");
    auto* s = dynamic_cast<Scalar*>(expr.get());
    ASSERT_NE(s, nullptr);
    EXPECT_DOUBLE_EQ(s->value(), 100.0);
}

// ============================================================
// Parser — unary functions
// ============================================================

TEST(ParserTest, UnaryFunction) {
    auto expr = parseExpression("abs(close)");
    // Should be UnaryExpr(ABS) wrapping ColumnRef(CLOSE)
    auto* u = dynamic_cast<UnaryExpr*>(expr.get());
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->opCode(), UnaryOpCode::ABS);
    EXPECT_NE(u->child(), nullptr);
}

TEST(ParserTest, UnaryLog) {
    auto expr = parseExpression("log(close)");
    auto* u = dynamic_cast<UnaryExpr*>(expr.get());
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->opCode(), UnaryOpCode::LOG);
}

TEST(ParserTest, UnarySqrt) {
    auto expr = parseExpression("sqrt(close)");
    auto* u = dynamic_cast<UnaryExpr*>(expr.get());
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->opCode(), UnaryOpCode::SQRT);
}

TEST(ParserTest, UnaryRank) {
    auto expr = parseExpression("rank(close)");
    auto* u = dynamic_cast<UnaryExpr*>(expr.get());
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->opCode(), UnaryOpCode::RANK);
}

TEST(ParserTest, UnaryNested) {
    auto expr = parseExpression("log(abs(close))");
    auto* outer = dynamic_cast<UnaryExpr*>(expr.get());
    ASSERT_NE(outer, nullptr);
    EXPECT_EQ(outer->opCode(), UnaryOpCode::LOG);

    auto* inner = dynamic_cast<const UnaryExpr*>(outer->child());
    ASSERT_NE(inner, nullptr);
    EXPECT_EQ(inner->opCode(), UnaryOpCode::ABS);
}

// ============================================================
// Parser — binary infix
// ============================================================

TEST(ParserTest, BinaryAdd) {
    auto expr = parseExpression("close + open");
    auto* b = dynamic_cast<BinaryExpr*>(expr.get());
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->opCode(), BinaryOpCode::ADD);
}

TEST(ParserTest, BinarySub) {
    auto expr = parseExpression("high - low");
    auto* b = dynamic_cast<BinaryExpr*>(expr.get());
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->opCode(), BinaryOpCode::SUB);
}

TEST(ParserTest, BinaryMul) {
    auto expr = parseExpression("close * volume");
    auto* b = dynamic_cast<BinaryExpr*>(expr.get());
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->opCode(), BinaryOpCode::MUL);
}

TEST(ParserTest, BinaryDiv) {
    auto expr = parseExpression("amount / volume");
    auto* b = dynamic_cast<BinaryExpr*>(expr.get());
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->opCode(), BinaryOpCode::DIV);
}

TEST(ParserTest, PrecedenceMulBeforeAdd) {
    // a + b * c  should parse as  a + (b * c)
    auto expr = parseExpression("close + open * 2");
    auto* b = dynamic_cast<BinaryExpr*>(expr.get());
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->opCode(), BinaryOpCode::ADD);
    // Right child should be MUL
    auto* rhs = dynamic_cast<const BinaryExpr*>(b->rhs());
    ASSERT_NE(rhs, nullptr);
    EXPECT_EQ(rhs->opCode(), BinaryOpCode::MUL);
}

TEST(ParserTest, PrecedenceMulBeforeSub) {
    // a - b / c  should parse as  a - (b / c)
    auto expr = parseExpression("close - open / volume");
    auto* b = dynamic_cast<BinaryExpr*>(expr.get());
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->opCode(), BinaryOpCode::SUB);
    auto* rhs = dynamic_cast<const BinaryExpr*>(b->rhs());
    ASSERT_NE(rhs, nullptr);
    EXPECT_EQ(rhs->opCode(), BinaryOpCode::DIV);
}

// ============================================================
// Parser — binary function call form
// ============================================================

TEST(ParserTest, BinaryFunctionMax) {
    auto expr = parseExpression("max(close, open)");
    auto* b = dynamic_cast<BinaryExpr*>(expr.get());
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->opCode(), BinaryOpCode::MAX);
}

TEST(ParserTest, BinaryFunctionEq) {
    auto expr = parseExpression("eq(close, open)");
    auto* b = dynamic_cast<BinaryExpr*>(expr.get());
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->opCode(), BinaryOpCode::EQ);
}

// ============================================================
// Parser — grouping
// ============================================================

TEST(ParserTest, Grouping) {
    auto expr = parseExpression("(close)");
    auto* ref = dynamic_cast<ColumnRef*>(expr.get());
    ASSERT_NE(ref, nullptr);
    EXPECT_EQ(ref->field(), Field::CLOSE);
}

TEST(ParserTest, GroupingOverridesPrecedence) {
    // (a + b) * c  → MUL(ADD(a,b), c)
    auto expr = parseExpression("(close + open) * volume");
    auto* b = dynamic_cast<BinaryExpr*>(expr.get());
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->opCode(), BinaryOpCode::MUL);
}

// ============================================================
// Parser — rolling functions
// ============================================================

TEST(ParserTest, RollingMean) {
    auto expr = parseExpression("rolling_mean(close, 20)");
    auto* r = dynamic_cast<RollingExpr*>(expr.get());
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->opCode(), RollingOpCode::ROLLING_MEAN);
    EXPECT_EQ(r->window(), 20);
}

TEST(ParserTest, RollingStd) {
    auto expr = parseExpression("rolling_std(close, 60)");
    auto* r = dynamic_cast<RollingExpr*>(expr.get());
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->opCode(), RollingOpCode::ROLLING_STD);
    EXPECT_EQ(r->window(), 60);
}

TEST(ParserTest, RollingShift) {
    auto expr = parseExpression("rolling_shift(close, 1)");
    auto* r = dynamic_cast<RollingExpr*>(expr.get());
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->opCode(), RollingOpCode::ROLLING_SHIFT);
    EXPECT_EQ(r->window(), 1);
}

TEST(ParserTest, RollingWithComplexChild) {
    auto expr = parseExpression("rolling_mean(high - low, 10)");
    auto* r = dynamic_cast<RollingExpr*>(expr.get());
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->opCode(), RollingOpCode::ROLLING_MEAN);
    EXPECT_EQ(r->window(), 10);
    // Child should be SUB(HIGH, LOW)
    auto* child = dynamic_cast<const BinaryExpr*>(r->child());
    ASSERT_NE(child, nullptr);
    EXPECT_EQ(child->opCode(), BinaryOpCode::SUB);
}

// ============================================================
// Parser — Red functions
// ============================================================

TEST(ParserTest, RedMean) {
    auto expr = parseExpression("red_mean(close)");
    auto* r = dynamic_cast<RedExpr*>(expr.get());
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->opCode(), RedOpCode::RED_MEAN);
}

TEST(ParserTest, RedZScore) {
    auto expr = parseExpression("red_zscore(close)");
    auto* r = dynamic_cast<RedExpr*>(expr.get());
    ASSERT_NE(r, nullptr);
    EXPECT_EQ(r->opCode(), RedOpCode::RED_ZSCORE);
}

// ============================================================
// Parser — CS functions
// ============================================================

TEST(ParserTest, CsRank) {
    auto expr = parseExpression("cs_rank(close)");
    auto* c = dynamic_cast<CsExpr*>(expr.get());
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->opCode(), CsOpCode::CS_RANK);
}

TEST(ParserTest, CsZScore) {
    auto expr = parseExpression("cs_zscore(close)");
    auto* c = dynamic_cast<CsExpr*>(expr.get());
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->opCode(), CsOpCode::CS_ZSCORE);
}

TEST(ParserTest, CsNormalize) {
    auto expr = parseExpression("cs_normalize(close)");
    auto* c = dynamic_cast<CsExpr*>(expr.get());
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->opCode(), CsOpCode::CS_NORMALIZE);
}

TEST(ParserTest, CsWinsorizeWithParams) {
    auto expr = parseExpression("cs_winsorize(close, 0.02, 0.98)");
    auto* c = dynamic_cast<CsExpr*>(expr.get());
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->opCode(), CsOpCode::CS_WINSORIZE);
    ASSERT_EQ(c->extraParams().size(), 2);
    EXPECT_DOUBLE_EQ(c->extraParams()[0], 0.02);
    EXPECT_DOUBLE_EQ(c->extraParams()[1], 0.98);
}

TEST(ParserTest, CsClipWithParams) {
    auto expr = parseExpression("cs_clip(close, -3.0, 3.0)");
    auto* c = dynamic_cast<CsExpr*>(expr.get());
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->opCode(), CsOpCode::CS_CLIP);
    ASSERT_EQ(c->extraParams().size(), 2);
    EXPECT_DOUBLE_EQ(c->extraParams()[0], -3.0);
    EXPECT_DOUBLE_EQ(c->extraParams()[1], 3.0);
}

// ============================================================
// Parser — composite expressions
// ============================================================

TEST(ParserTest, CompositeAbsLogDiffMulVolume) {
    // ABS(LOG(CLOSE) - LOG(VWAP)) * VOLUME
    auto expr = parseExpression("abs(log(close) - log(vwap)) * volume");

    auto* top = dynamic_cast<BinaryExpr*>(expr.get());
    ASSERT_NE(top, nullptr);
    EXPECT_EQ(top->opCode(), BinaryOpCode::MUL);

    // Left child: ABS(...)
    auto* absNode = dynamic_cast<const UnaryExpr*>(top->lhs());
    ASSERT_NE(absNode, nullptr);
    EXPECT_EQ(absNode->opCode(), UnaryOpCode::ABS);

    // Right child: ColumnRef(VOLUME)
    auto* volRef = dynamic_cast<const ColumnRef*>(top->rhs());
    ASSERT_NE(volRef, nullptr);
    EXPECT_EQ(volRef->field(), Field::VOLUME);
}

TEST(ParserTest, CompositeSqrtAbsLog) {
    auto expr = parseExpression("sqrt(abs(log(close)))");

    auto* sqrtNode = dynamic_cast<UnaryExpr*>(expr.get());
    ASSERT_NE(sqrtNode, nullptr);
    EXPECT_EQ(sqrtNode->opCode(), UnaryOpCode::SQRT);

    auto* absNode = dynamic_cast<const UnaryExpr*>(sqrtNode->child());
    ASSERT_NE(absNode, nullptr);
    EXPECT_EQ(absNode->opCode(), UnaryOpCode::ABS);
}

TEST(ParserTest, UnaryNeg) {
    // -close should parse as NEG(CLOSE)
    auto expr = parseExpression("-close");
    auto* u = dynamic_cast<UnaryExpr*>(expr.get());
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->opCode(), UnaryOpCode::NEG);
}

TEST(ParserTest, UnaryNegWithComposite) {
    // -(close + open) → NEG(ADD(CLOSE, OPEN))
    auto expr = parseExpression("-(close + open)");
    auto* u = dynamic_cast<UnaryExpr*>(expr.get());
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->opCode(), UnaryOpCode::NEG);
}

// ============================================================
// Parser — error cases
// ============================================================

TEST(ParserTest, ErrorMissingCloseParen) {
    EXPECT_THROW(parseExpression("abs(close"), std::runtime_error);
}

TEST(ParserTest, ErrorMissingOpenParen) {
    EXPECT_THROW(parseExpression("abs close)"), std::runtime_error);
}

TEST(ParserTest, ErrorUnknownFunction) {
    EXPECT_THROW(parseExpression("unknown_func(close)"), std::runtime_error);
}

TEST(ParserTest, ErrorWrongArgCount) {
    // abs takes 1 arg, give it 2
    EXPECT_THROW(parseExpression("abs(close, open)"), std::runtime_error);
}

TEST(ParserTest, ErrorEmptyArgs) {
    EXPECT_THROW(parseExpression("abs()"), std::runtime_error);
}

TEST(ParserTest, ErrorTrailingGarbage) {
    EXPECT_THROW(parseExpression("close +"), std::runtime_error);
}

// ============================================================
// End-to-end: parseExpression() + ExecutionEngine
// ============================================================

class ParserEndToEndTest : public ::testing::Test {
protected:
    static constexpr std::size_t kN = 20;

    void SetUp() override {
        std::vector<int64_t> timestamps(kN);
        for (std::size_t i = 0; i < kN; ++i) {
            timestamps[i] = static_cast<int64_t>(i);
        }
        TimestampIndex tsIdx(timestamps.data(), kN);
        md_ = MarketData("TEST", std::move(tsIdx));

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
    ExecutionEngine engine_;
};

TEST_F(ParserEndToEndTest, SimpleColumnRef) {
    auto ast = parseExpression("close");
    auto result = engine_.evaluate(*ast, md_);

    ASSERT_EQ(result.size(), kN);
    const auto& col = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(result[i], col[i]);
    }
}

TEST_F(ParserEndToEndTest, UnaryLog) {
    auto ast = parseExpression("log(close)");
    auto result = engine_.evaluate(*ast, md_);

    const auto& col = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(result[i], std::log(col[i]));
    }
}

TEST_F(ParserEndToEndTest, BinaryAddViaInfix) {
    auto ast = parseExpression("close + open");
    auto result = engine_.evaluate(*ast, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(result[i], c[i] + o[i]);
    }
}

TEST_F(ParserEndToEndTest, CompositeExpression) {
    auto ast = parseExpression("abs(log(close) - log(vwap)) * volume");
    auto result = engine_.evaluate(*ast, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& v = md_.column<double>(Field::VWAP);
    const auto& vol = md_.column<double>(Field::VOLUME);
    for (std::size_t i = 0; i < kN; ++i) {
        double expected = std::abs(std::log(c[i]) - std::log(v[i])) * vol[i];
        EXPECT_DOUBLE_EQ(result[i], expected);
    }
}

TEST_F(ParserEndToEndTest, RollingMeanExpression) {
    auto ast = parseExpression("rolling_mean(close, 5)");
    auto result = engine_.evaluate(*ast, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    // First 4 NaN
    for (std::size_t i = 0; i < 4; ++i) EXPECT_TRUE(std::isnan(result[i]));
    // Position 4
    double expected = (c[0] + c[1] + c[2] + c[3] + c[4]) / 5.0;
    EXPECT_DOUBLE_EQ(result[4], expected);
}

TEST_F(ParserEndToEndTest, CsRankExpression) {
    // CsExpr cannot be evaluated on a single MarketData — it requires
    // PanelData cross-sectional evaluation via FactorCalculator.
    // The engine should propagate the error from CsExpr::evaluate().
    auto ast = parseExpression("cs_rank(close)");
    EXPECT_THROW(engine_.evaluate(*ast, md_), std::logic_error);
}

TEST_F(ParserEndToEndTest, BinaryFunctionMax) {
    auto ast = parseExpression("max(close, open)");
    auto result = engine_.evaluate(*ast, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(result[i], std::max(c[i], o[i]));
    }
}

TEST_F(ParserEndToEndTest, CaseInsensitive) {
    auto ast1 = parseExpression("ABS(CLOSE)");
    auto ast2 = parseExpression("abs(close)");
    auto r1 = engine_.evaluate(*ast1, md_);
    auto r2 = engine_.evaluate(*ast2, md_);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(r1[i], r2[i]);
    }
}

// ============================================================
// Phase 4: Fusion correctness — fused vs standard evaluation
// ============================================================

// Helper: build AST via parseExpression, evaluate via engine (which uses fusion).
// Compare with manually-built AST evaluated directly (no fusion).
// If both give the same result, fusion is correct.

TEST_F(ParserEndToEndTest, FusionUnaryChain) {
    // SQRT(ABS(LOG(CLOSE))) — this should fuse into a single loop
    auto ast = parseExpression("sqrt(abs(log(close)))");
    auto fusedResult = engine_.evaluate(*ast, md_);

    // Manual evaluation (separate evaluate() call bypasses fusion)
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    auto logExpr = std::make_unique<UnaryExpr>(UnaryOpCode::LOG, std::move(close));
    auto absExpr = std::make_unique<UnaryExpr>(UnaryOpCode::ABS, std::move(logExpr));
    UnaryExpr sqrtExpr(UnaryOpCode::SQRT, std::move(absExpr));
    std::vector<double> manual(kN);
    sqrtExpr.evaluate(md_, manual.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(fusedResult[i], manual[i]);
    }
}

TEST_F(ParserEndToEndTest, FusionBinaryWithChains) {
    // LOG(CLOSE) + LOG(VWAP)
    auto ast = parseExpression("log(close) + log(vwap)");
    auto fusedResult = engine_.evaluate(*ast, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& v = md_.column<double>(Field::VWAP);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(fusedResult[i], std::log(c[i]) + std::log(v[i]));
    }
}

TEST_F(ParserEndToEndTest, FusionResultChainOnBinary) {
    // ABS(LOG(CLOSE) - LOG(VWAP)) — unary result chain on binary
    auto ast = parseExpression("abs(log(close) - log(vwap))");
    auto fusedResult = engine_.evaluate(*ast, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& v = md_.column<double>(Field::VWAP);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(fusedResult[i],
                         std::abs(std::log(c[i]) - std::log(v[i])));
    }
}

TEST_F(ParserEndToEndTest, FusionDeepUnaryChain) {
    // EXP(SQRT(ABS(LOG(CLOSE)))) — 4-level unary chain
    auto ast = parseExpression("exp(sqrt(abs(log(close))))");
    auto fusedResult = engine_.evaluate(*ast, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(fusedResult[i],
                         std::exp(std::sqrt(std::abs(std::log(c[i])))));
    }
}

TEST_F(ParserEndToEndTest, FusionScalarLeaf) {
    // ABS(-3.0) + CLOSE → fused binary with scalar lhs
    auto ast = parseExpression("abs(-3.0) + close");
    auto fusedResult = engine_.evaluate(*ast, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(fusedResult[i], 3.0 + c[i]);
    }
}

// ============================================================
// Phase 4: RollingQuantileOp
// ============================================================

TEST_F(ParserEndToEndTest, RollingQuantile) {
    // ROLLING_QUANTILE(CLOSE, 5, 0.5) — median over window=5
    auto ast = parseExpression("rolling_quantile(close, 5, 0.5)");
    auto result = engine_.evaluate(*ast, md_);

    ASSERT_EQ(result.size(), kN);

    // First 4 positions NaN
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(result[i]));
    }

    // Position 4: median of c[0..4] = {12,13,14,15,16} = 14
    EXPECT_DOUBLE_EQ(result[4], 14.0);
}

// ============================================================
// Phase 4: Non-fusible expression fallback
// ============================================================

TEST_F(ParserEndToEndTest, NonFusibleRollingFallsBack) {
    // ROLLING_MEAN has a fusion boundary — must fall back to standard eval
    auto ast = parseExpression("rolling_mean(close, 5)");
    auto result = engine_.evaluate(*ast, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < 4; ++i) EXPECT_TRUE(std::isnan(result[i]));
    double expected = (c[0] + c[1] + c[2] + c[3] + c[4]) / 5.0;
    EXPECT_DOUBLE_EQ(result[4], expected);
}

TEST_F(ParserEndToEndTest, NonFusibleRedFallsBack) {
    auto ast = parseExpression("red_zscore(close)");
    auto result = engine_.evaluate(*ast, md_);

    // Z-scores should have mean ≈ 0
    double sum = 0.0;
    for (std::size_t i = 0; i < kN; ++i) sum += result[i];
    EXPECT_NEAR(sum / static_cast<double>(kN), 0.0, 1e-12);
}

// ============================================================
// Phase 4: FusedKernel direct compilation tests
// ============================================================

TEST(ParserTest, FusedKernelCompilesUnaryChain) {
    auto ast = parseExpression("abs(log(close))");
    // Manually build the same to verify fusion compiles
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    auto logExpr = std::make_unique<UnaryExpr>(UnaryOpCode::LOG, std::move(close));
    UnaryExpr absExpr(UnaryOpCode::ABS, std::move(logExpr));

    FusedLoopGenerator gen;
    auto kernel = gen.tryCompile(&absExpr);
    EXPECT_NE(kernel, nullptr);
    if (kernel) {
        EXPECT_EQ(kernel->kind, FusedKernel::kUnaryChain);
        EXPECT_EQ(kernel->lhsChain.size(), 2);  // LOG + ABS
        EXPECT_TRUE(kernel->lhsIsColumn);
    }
}

TEST(ParserTest, FusedKernelCompilesBinary) {
    auto ast = parseExpression("log(close) + log(vwap)");

    FusedLoopGenerator gen;
    auto kernel = gen.tryCompile(ast.get());
    EXPECT_NE(kernel, nullptr);
    if (kernel) {
        EXPECT_EQ(kernel->kind, FusedKernel::kBinary);
        EXPECT_EQ(kernel->lhsChain.size(), 1);  // LOG
        EXPECT_EQ(kernel->rhsChain.size(), 1);  // LOG
        EXPECT_EQ(kernel->binaryOp, BinaryOpCode::ADD);
    }
}

TEST(ParserTest, FusedKernelRejectsRolling) {
    auto ast = parseExpression("rolling_mean(close, 5)");

    FusedLoopGenerator gen;
    auto kernel = gen.tryCompile(ast.get());
    EXPECT_EQ(kernel, nullptr);  // RollingExpr is a fusion boundary
}

// ============================================================
// Node count / depth verification
// ============================================================

TEST(ParserTest, NodeCountSimple) {
    auto expr = parseExpression("close");
    EXPECT_EQ(expr->nodeCount(), 1);
}

TEST(ParserTest, NodeCountComposite) {
    // abs(log(close)) → 3 nodes (2 Unary + 1 ColumnRef)
    auto expr = parseExpression("abs(log(close))");
    EXPECT_EQ(expr->nodeCount(), 3);

    // close + open → 3 nodes (1 Binary + 2 ColumnRef)
    auto expr2 = parseExpression("close + open");
    EXPECT_EQ(expr2->nodeCount(), 3);
}
