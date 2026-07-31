// test_binary_ops.cpp — comprehensive BinaryExpr tests
//
// Tests all binary operators with column+column, column+scalar,
// and scalar+column patterns through the ExecutionEngine.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/engine/ExecutionEngine.h"
#include "quantcore/expression/BinaryExpr.h"
#include "quantcore/expression/ColumnRef.h"
#include "quantcore/expression/Scalar.h"
#include "quantcore/expression/UnaryExpr.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;

// ============================================================
// Test fixture
// ============================================================

class BinaryOpsTest : public ::testing::Test {
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

    MarketData md_;
    ExecutionEngine engine_;
};

// ============================================================
// Column + Column
// ============================================================

TEST_F(BinaryOpsTest, AddColumns) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::ADD,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<ColumnRef>(Field::OPEN));
    auto result = engine_.evaluate(*expr, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i] + o[i]);
}

TEST_F(BinaryOpsTest, SubColumns) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::SUB,
        std::make_unique<ColumnRef>(Field::HIGH),
        std::make_unique<ColumnRef>(Field::LOW));
    auto result = engine_.evaluate(*expr, md_);

    const auto& h = md_.column<double>(Field::HIGH);
    const auto& l = md_.column<double>(Field::LOW);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], h[i] - l[i]);
}

TEST_F(BinaryOpsTest, MulColumns) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::MUL,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<ColumnRef>(Field::VOLUME));
    auto result = engine_.evaluate(*expr, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& v = md_.column<double>(Field::VOLUME);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i] * v[i]);
}

TEST_F(BinaryOpsTest, DivColumns) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::DIV,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<ColumnRef>(Field::OPEN));
    auto result = engine_.evaluate(*expr, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i] / o[i]);
}

TEST_F(BinaryOpsTest, MaxColumns) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::MAX,
        std::make_unique<ColumnRef>(Field::HIGH),
        std::make_unique<ColumnRef>(Field::LOW));
    auto result = engine_.evaluate(*expr, md_);

    const auto& h = md_.column<double>(Field::HIGH);
    const auto& l = md_.column<double>(Field::LOW);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::max(h[i], l[i]));
}

TEST_F(BinaryOpsTest, MinColumns) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::MIN,
        std::make_unique<ColumnRef>(Field::HIGH),
        std::make_unique<ColumnRef>(Field::LOW));
    auto result = engine_.evaluate(*expr, md_);

    const auto& h = md_.column<double>(Field::HIGH);
    const auto& l = md_.column<double>(Field::LOW);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::min(h[i], l[i]));
}

// ============================================================
// Comparison operators
// ============================================================

TEST_F(BinaryOpsTest, GtColumns) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::GT,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<ColumnRef>(Field::OPEN));
    auto result = engine_.evaluate(*expr, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i] > o[i] ? 1.0 : 0.0);
}

TEST_F(BinaryOpsTest, LtColumns) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::LT,
        std::make_unique<ColumnRef>(Field::LOW),
        std::make_unique<ColumnRef>(Field::HIGH));
    auto result = engine_.evaluate(*expr, md_);

    const auto& l = md_.column<double>(Field::LOW);
    const auto& h = md_.column<double>(Field::HIGH);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], l[i] < h[i] ? 1.0 : 0.0);
}

TEST_F(BinaryOpsTest, EqColumns) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::EQ,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<ColumnRef>(Field::CLOSE));
    auto result = engine_.evaluate(*expr, md_);

    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], 1.0);  // CLOSE == CLOSE always true
}

TEST_F(BinaryOpsTest, NeqColumns) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::NEQ,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<ColumnRef>(Field::OPEN));
    auto result = engine_.evaluate(*expr, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i] != o[i] ? 1.0 : 0.0);
}

// ============================================================
// Column + Scalar
// ============================================================

TEST_F(BinaryOpsTest, AddColumnScalar) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::ADD,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<Scalar>(10.0));
    auto result = engine_.evaluate(*expr, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i] + 10.0);
}

TEST_F(BinaryOpsTest, MulColumnScalar) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::MUL,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<Scalar>(0.5));
    auto result = engine_.evaluate(*expr, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], c[i] * 0.5);
}

// ============================================================
// Scalar + Column
// ============================================================

TEST_F(BinaryOpsTest, AddScalarColumn) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::ADD,
        std::make_unique<Scalar>(100.0),
        std::make_unique<ColumnRef>(Field::CLOSE));
    auto result = engine_.evaluate(*expr, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], 100.0 + c[i]);
}

// ============================================================
// Scalar + Scalar
// ============================================================

TEST_F(BinaryOpsTest, AddScalars) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::ADD,
        std::make_unique<Scalar>(3.0),
        std::make_unique<Scalar>(4.0));
    auto result = engine_.evaluate(*expr, md_);

    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], 7.0);
}

// ============================================================
// Nested binary expressions
// ============================================================

TEST_F(BinaryOpsTest, NestedAddMul) {
    // (CLOSE + OPEN) * VOLUME
    auto add = std::make_unique<BinaryExpr>(BinaryOpCode::ADD,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<ColumnRef>(Field::OPEN));
    auto mul = std::make_unique<BinaryExpr>(BinaryOpCode::MUL,
        std::move(add),
        std::make_unique<ColumnRef>(Field::VOLUME));
    auto result = engine_.evaluate(*mul, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& o = md_.column<double>(Field::OPEN);
    const auto& v = md_.column<double>(Field::VOLUME);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], (c[i] + o[i]) * v[i]);
}

TEST_F(BinaryOpsTest, NestedWithUnary) {
    // LOG(CLOSE) + LOG(VOLUME)
    auto logClose = std::make_unique<UnaryExpr>(UnaryOpCode::LOG,
        std::make_unique<ColumnRef>(Field::CLOSE));
    auto logVolume = std::make_unique<UnaryExpr>(UnaryOpCode::LOG,
        std::make_unique<ColumnRef>(Field::VOLUME));
    auto add = std::make_unique<BinaryExpr>(BinaryOpCode::ADD,
        std::move(logClose), std::move(logVolume));
    auto result = engine_.evaluate(*add, md_);

    const auto& c = md_.column<double>(Field::CLOSE);
    const auto& v = md_.column<double>(Field::VOLUME);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::log(c[i]) + std::log(v[i]));
}

// ============================================================
// Edge cases
// ============================================================

TEST_F(BinaryOpsTest, DivByZero) {
    auto expr = std::make_unique<BinaryExpr>(BinaryOpCode::DIV,
        std::make_unique<ColumnRef>(Field::CLOSE),
        std::make_unique<Scalar>(0.0));
    auto result = engine_.evaluate(*expr, md_);

    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_TRUE(std::isinf(result[i]));
}
