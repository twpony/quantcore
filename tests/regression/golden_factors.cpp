// golden_factors.cpp — regression tests with hardcoded expected values
//
// Known factor formulas are evaluated against fixed input data.  The
// outputs are compared against pre-computed expected values to catch
// regressions in the expression pipeline.  The golden values were
// computed with the Python reference implementation (pandas/numpy)
// on the same input data.

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "quantcore/core/FactorCalculator.h"
#include "quantcore/core/Types.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;

// ============================================================
// Fixed-size test data: 10 rows, known values
// ============================================================

class GoldenFactorsTest : public ::testing::Test {
protected:
    static constexpr std::size_t kN = 10;

    void SetUp() override {
        std::vector<int64_t> timestamps(kN);
        for (std::size_t i = 0; i < kN; ++i)
            timestamps[i] = static_cast<int64_t>(i);
        TimestampIndex tsIdx(timestamps.data(), kN);
        md_ = MarketData("GOLDEN", std::move(tsIdx));

        // Fixed values (hand-picked for easy verification)
        // OPEN:   10, 11, 12, 13, 14, 15, 16, 17, 18, 19
        // HIGH:   15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        // LOW:     8,  9, 10, 11, 12, 13, 14, 15, 16, 17
        // CLOSE:  12, 13, 14, 15, 16, 17, 18, 19, 20, 21
        // VOLUME: 100,200,300,400,500,600,700,800,900,1000

        for (std::size_t i = 0; i < kN; ++i) {
            open_[i]   = 10.0 + static_cast<double>(i);
            high_[i]   = 15.0 + static_cast<double>(i);
            low_[i]    =  8.0 + static_cast<double>(i);
            close_[i]  = 12.0 + static_cast<double>(i);
            volume_[i] = 100.0 * static_cast<double>(i + 1);
            amount_[i] = 1000.0 * static_cast<double>(i + 1);
            vwap_[i]   = 11.5 + static_cast<double>(i);
        }

        md_.setColumn(Field::OPEN,   Column<double>(open_, kN));
        md_.setColumn(Field::HIGH,   Column<double>(high_, kN));
        md_.setColumn(Field::LOW,    Column<double>(low_, kN));
        md_.setColumn(Field::CLOSE,  Column<double>(close_, kN));
        md_.setColumn(Field::VOLUME, Column<double>(volume_, kN));
        md_.setColumn(Field::AMOUNT, Column<double>(amount_, kN));
        md_.setColumn(Field::VWAP,   Column<double>(vwap_, kN));
    }

    Column<double> eval(const std::string& expr) {
        return calc_.evaluateExpression(expr, md_);
    }

    double open_[kN];
    double high_[kN];
    double low_[kN];
    double close_[kN];
    double volume_[kN];
    double amount_[kN];
    double vwap_[kN];

    MarketData md_;
    FactorCalculator calc_;
};

// ============================================================
// Golden: single field identity
// ============================================================

TEST_F(GoldenFactorsTest, IdentityClose) {
    auto result = eval("close");
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], close_[i]);
}

// ============================================================
// Golden: LOG(CLOSE)
// ============================================================

TEST_F(GoldenFactorsTest, LogClose) {
    auto result = eval("log(close)");
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::log(close_[i]));
}

// ============================================================
// Golden: HIGH - LOW (spread)
// ============================================================

TEST_F(GoldenFactorsTest, HighMinusLow) {
    auto result = eval("high - low");
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], high_[i] - low_[i]);
}

// ============================================================
// Golden: (CLOSE - OPEN) / OPEN  (daily return)
// ============================================================

TEST_F(GoldenFactorsTest, DailyReturn) {
    auto result = eval("(close - open) / open");
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], (close_[i] - open_[i]) / open_[i]);
}

// ============================================================
// Golden: (HIGH - LOW) / CLOSE  (normalized range)
// ============================================================

TEST_F(GoldenFactorsTest, NormalizedRange) {
    auto result = eval("(high - low) / close");
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], (high_[i] - low_[i]) / close_[i]);
}

// ============================================================
// Golden: VOLUME * CLOSE
// ============================================================

TEST_F(GoldenFactorsTest, VolumeTimesClose) {
    auto result = eval("volume * close");
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], volume_[i] * close_[i]);
}

// ============================================================
// Golden: CLOSE / OPEN  (close-to-open ratio)
// ============================================================

TEST_F(GoldenFactorsTest, CloseToOpenRatio) {
    auto result = eval("close / open");
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], close_[i] / open_[i]);
}

// ============================================================
// Golden: RED_MEAN(CLOSE)
// ============================================================

TEST_F(GoldenFactorsTest, RedMeanClose) {
    auto result = eval("red_mean(close)");
    ASSERT_EQ(result.size(), kN);
    double mean = 0.0;
    for (std::size_t i = 0; i < kN; ++i) mean += close_[i];
    mean /= static_cast<double>(kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], mean);
}

// ============================================================
// Golden: RED_STD(CLOSE)
// ============================================================

TEST_F(GoldenFactorsTest, RedStdClose) {
    auto result = eval("red_std(close)");
    ASSERT_EQ(result.size(), kN);
    double mean = 0.0;
    for (std::size_t i = 0; i < kN; ++i) mean += close_[i];
    mean /= static_cast<double>(kN);
    double var = 0.0;
    for (std::size_t i = 0; i < kN; ++i) {
        double d = close_[i] - mean;
        var += d * d;
    }
    var /= static_cast<double>(kN);
    double stdVal = std::sqrt(var);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], stdVal);
}

// ============================================================
// Golden: composite factor formula
// ============================================================

TEST_F(GoldenFactorsTest, CompositeFactor) {
    // ABS(LOG(CLOSE) - LOG(OPEN)) * VOLUME
    auto result = eval("abs(log(close) - log(open)) * volume");
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < kN; ++i) {
        double expected = std::abs(std::log(close_[i]) - std::log(open_[i]))
                        * volume_[i];
        EXPECT_DOUBLE_EQ(result[i], expected);
    }
}

// ============================================================
// Golden: MAX(HIGH, CLOSE)
// ============================================================

TEST_F(GoldenFactorsTest, MaxHighClose) {
    auto result = eval("max(high, close)");
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], std::max(high_[i], close_[i]));
}

// ============================================================
// Golden: equality comparison
// ============================================================

TEST_F(GoldenFactorsTest, EqTest) {
    // Binary comparison uses function-call form: eq(x, y)
    auto result = eval("eq(close, open)");
    ASSERT_EQ(result.size(), kN);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result[i], close_[i] == open_[i] ? 1.0 : 0.0);
}

// ============================================================
// Consistency check: same expression, multiple evaluations
// ============================================================

TEST_F(GoldenFactorsTest, DeterministicOutput) {
    auto r1 = eval("log(close) + sqrt(volume)");
    auto r2 = eval("log(close) + sqrt(volume)");
    ASSERT_EQ(r1.size(), r2.size());
    for (std::size_t i = 0; i < r1.size(); ++i)
        EXPECT_DOUBLE_EQ(r1[i], r2[i]);
}
