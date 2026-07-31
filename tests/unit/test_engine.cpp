// test_engine.cpp — unit tests for ExecutionEngine
// Phase: 二期必实现
//
// Tests engine.evaluate() with various expression trees, verifies
// correct results, metrics collection, and pool integration.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/engine/ExecutionEngine.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/expression/ColumnRef.h"
#include "quantcore/expression/Scalar.h"
#include "quantcore/expression/UnaryExpr.h"
#include "quantcore/expression/BinaryExpr.h"
#include "quantcore/expression/RollingExpr.h"
#include "quantcore/expression/RedExpr.h"
#include "quantcore/expression/CsExpr.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;

// ============================================================
// Test fixture
// ============================================================

class EngineTest : public ::testing::Test {
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

// ============================================================
// Basic evaluation tests
// ============================================================

TEST_F(EngineTest, EvaluateColumnRef) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    UnaryExpr expr(UnaryOpCode::LOG, std::move(close));  // LOG(CLOSE)

    Column<double> result = engine_.evaluate(expr, md_);

    ASSERT_EQ(result.size(), kN);
    const auto& col = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(result[i], std::log(col[i]));
    }
}

TEST_F(EngineTest, EvaluateCompositeExpression) {
    // ABS(LOG(CLOSE) - LOG(VWAP)) * VOLUME
    auto close  = std::make_unique<ColumnRef>(Field::CLOSE);
    auto vwap   = std::make_unique<ColumnRef>(Field::VWAP);
    auto volume = std::make_unique<ColumnRef>(Field::VOLUME);

    auto logClose = std::make_unique<UnaryExpr>(UnaryOpCode::LOG, std::move(close));
    auto logVwap  = std::make_unique<UnaryExpr>(UnaryOpCode::LOG, std::move(vwap));
    auto diff = std::make_unique<BinaryExpr>(BinaryOpCode::SUB,
        std::move(logClose), std::move(logVwap));
    auto absDiff = std::make_unique<UnaryExpr>(UnaryOpCode::ABS, std::move(diff));
    auto resultExpr = std::make_unique<BinaryExpr>(BinaryOpCode::MUL,
        std::move(absDiff), std::move(volume));

    Column<double> result = engine_.evaluate(*resultExpr, md_);

    ASSERT_EQ(result.size(), kN);
    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& v = md_.column<double>(Field::VWAP);
    const auto& vol = md_.column<double>(Field::VOLUME);
    for (std::size_t i = 0; i < kN; ++i) {
        double expected = std::abs(std::log(c[i]) - std::log(v[i])) * vol[i];
        EXPECT_DOUBLE_EQ(result[i], expected);
    }
}

TEST_F(EngineTest, EvaluateRollingExpression) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    RollingExpr rolling(RollingOpCode::ROLLING_MEAN, 5, std::move(close));

    Column<double> result = engine_.evaluate(rolling, md_);

    ASSERT_EQ(result.size(), kN);
    const auto& col = md_.column<double>(Field::CLOSE);

    // First 4 positions should be NaN
    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_TRUE(std::isnan(result[i]));
    }

    double expected4 = (col[0] + col[1] + col[2] + col[3] + col[4]) / 5.0;
    EXPECT_DOUBLE_EQ(result[4], expected4);
}

TEST_F(EngineTest, EvaluateWithPoolIntegration) {
    // Verify that the pool path produces same results as direct evaluation
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    BinaryExpr add(BinaryOpCode::ADD,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<ColumnRef>(Field::OPEN));

    Column<double> engineResult = engine_.evaluate(add, md_);

    // Compare with manual evaluation
    std::vector<double> manualOutput(kN);
    add.evaluate(md_, manualOutput.data(), kN);

    for (std::size_t i = 0; i < kN; ++i) {
        EXPECT_DOUBLE_EQ(engineResult[i], manualOutput[i]);
    }
}

TEST_F(EngineTest, EvaluateEmptyMarketData) {
    MarketData emptyMd;
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);

    Column<double> result = engine_.evaluate(*close, emptyMd);

    EXPECT_EQ(result.size(), 0);
}

// ============================================================
// Metrics tests
// ============================================================

TEST_F(EngineTest, MetricsAfterEvaluation) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);

    // Before evaluation
    EXPECT_EQ(engine_.metrics().evaluationCount(), 0);

    engine_.evaluate(*close, md_);

    // After evaluation
    EXPECT_EQ(engine_.metrics().evaluationCount(), 1);
    EXPECT_GT(engine_.metrics().totalUsec(), 0);
    EXPECT_EQ(engine_.metrics().totalRows(), kN);
    EXPECT_GT(engine_.metrics().totalNodeCount(), 0);
}

TEST_F(EngineTest, MetricsReset) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    engine_.evaluate(*close, md_);

    EXPECT_EQ(engine_.metrics().evaluationCount(), 1);

    engine_.resetMetrics();
    EXPECT_EQ(engine_.metrics().evaluationCount(), 0);
    EXPECT_EQ(engine_.metrics().totalUsec(), 0);
}

TEST_F(EngineTest, MultipleEvaluationsAccumulateMetrics) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);

    engine_.evaluate(*close, md_);
    engine_.evaluate(*close, md_);
    engine_.evaluate(*close, md_);

    EXPECT_EQ(engine_.metrics().evaluationCount(), 3);
    EXPECT_EQ(engine_.metrics().totalRows(), kN * 3);
}

// ============================================================
// Pool access tests
// ============================================================

TEST_F(EngineTest, PoolIsAccessible) {
    EXPECT_NO_THROW(engine_.pool());
}

TEST_F(EngineTest, PoolHasAllocationsAfterEvaluation) {
    auto close = std::make_unique<ColumnRef>(Field::CLOSE);
    UnaryExpr logExpr(UnaryOpCode::LOG, std::move(close));

    engine_.evaluate(logExpr, md_);

    // Pool should have allocated at least the result buffer
    EXPECT_GT(engine_.pool().totalAllocated(), 0);
}

// ============================================================
// Clone + different data via engine
// ============================================================

TEST_F(EngineTest, CloneEvaluateOnDifferentData) {
    auto close1 = std::make_unique<ColumnRef>(Field::CLOSE);
    auto open1  = std::make_unique<ColumnRef>(Field::OPEN);
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::ADD,
        std::move(close1), std::move(open1));

    Column<double> result1 = engine_.evaluate(*expr, md_);

    // Create different MarketData
    MarketData md2;
    {
        std::vector<int64_t> ts(10);
        for (std::size_t i = 0; i < 10; ++i) ts[i] = static_cast<int64_t>(i);
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

    ExecutionEngine engine2;
    auto cloned = expr->clone();
    Column<double> result2 = engine2.evaluate(*cloned, md2);

    ASSERT_EQ(result2.size(), 10);
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(result2[i], 195.0 + 2.0 * static_cast<double>(i));
    }
}
