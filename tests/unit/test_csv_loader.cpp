// test_csv_loader.cpp — tests for CsvLoader factor CSV loading
//
// Tests:
//   1. Parse valid 3-column CSV
//   2. Parse CSV with missing optional description
//   3. Skip comment lines and blank lines
//   4. loadFactorsFromCsv batch registers correctly
//   5. Evaluate factors loaded from CSV
//   6. Error on missing file

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "factors/CsvLoader.h"
#include "quantcore/core/FactorCalculator.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;
using namespace quantcore::factors;

// ============================================================
// Fixture: 30 rows of synthetic market data
// ============================================================

class CsvLoaderTest : public ::testing::Test {
protected:
    static constexpr std::size_t kN = 30;

    void SetUp() override {
        std::vector<int64_t> timestamps(kN);
        for (std::size_t i = 0; i < kN; ++i)
            timestamps[i] = static_cast<int64_t>(i);
        TimestampIndex tsIdx(timestamps.data(), kN);
        md_ = MarketData("TEST", std::move(tsIdx));

        // Fill all fields with linear data
        for (std::size_t f = 0; f < static_cast<std::size_t>(Field::kFieldCount); ++f) {
            auto field = static_cast<Field>(f);
            Column<double> col(kN);
            for (std::size_t i = 0; i < kN; ++i)
                col[i] = 100.0 + static_cast<double>(i);
            md_.setColumn(field, std::move(col));
        }
    }

    /// Write a temporary CSV file and return its path.
    std::string writeTempCsv(const std::string& content) {
        std::string path = "/tmp/test_factors_" +
                           std::to_string(++counter_) + ".csv";
        std::ofstream f(path);
        f << content;
        f.close();
        tempFiles_.push_back(path);
        return path;
    }

    void TearDown() override {
        for (auto& path : tempFiles_) {
            std::remove(path.c_str());
        }
    }

    MarketData md_;
    FactorCalculator calc_;
    std::vector<std::string> tempFiles_;
    static int counter_;
};

int CsvLoaderTest::counter_ = 0;

// ============================================================
// parseFactorCsv tests
// ============================================================

TEST_F(CsvLoaderTest, ParseValidThreeColumnCsv) {
    std::string path = writeTempCsv(
        "name,expression,description\n"
        "f1,close,just the close price\n"
        "f2,log(close),log of close\n"
    );

    auto records = parseFactorCsv(path);
    ASSERT_EQ(records.size(), 2);

    EXPECT_EQ(records[0].name, "f1");
    EXPECT_EQ(records[0].expression, "close");
    EXPECT_EQ(records[0].description, "just the close price");

    EXPECT_EQ(records[1].name, "f2");
    EXPECT_EQ(records[1].expression, "log(close)");
    EXPECT_EQ(records[1].description, "log of close");
}

TEST_F(CsvLoaderTest, ParseCsvMissingOptionalDescription) {
    std::string path = writeTempCsv(
        "name,expression,description\n"
        "f1,close\n"         // no description
        "f2,log(close),has description\n"
    );

    auto records = parseFactorCsv(path);
    ASSERT_EQ(records.size(), 2);

    EXPECT_EQ(records[0].name, "f1");
    EXPECT_EQ(records[0].expression, "close");
    EXPECT_TRUE(records[0].description.empty());

    EXPECT_EQ(records[1].name, "f2");
    EXPECT_EQ(records[1].expression, "log(close)");
    EXPECT_EQ(records[1].description, "has description");
}

TEST_F(CsvLoaderTest, SkipCommentsAndBlankLines) {
    std::string path = writeTempCsv(
        "name,expression,description\n"
        "# This is a comment — should be skipped\n"
        "f1,close,price\n"
        "\n"
        "   # indented comment\n"
        "f2,log(close),log price\n"
        "\n"
    );

    auto records = parseFactorCsv(path);
    ASSERT_EQ(records.size(), 2);
    EXPECT_EQ(records[0].name, "f1");
    EXPECT_EQ(records[1].name, "f2");
}

TEST_F(CsvLoaderTest, ParseCsvWithExtraWhitespace) {
    std::string path = writeTempCsv(
        "name,expression,description\n"
        "  f1  ,  close + open  ,  sum of close and open  \n"
    );

    auto records = parseFactorCsv(path);
    ASSERT_EQ(records.size(), 1);
    EXPECT_EQ(records[0].name, "f1");
    EXPECT_EQ(records[0].expression, "close + open");
    EXPECT_EQ(records[0].description, "sum of close and open");
}

TEST_F(CsvLoaderTest, ParseEmptyCsvReturnsEmpty) {
    std::string path = writeTempCsv("name,expression,description\n");
    auto records = parseFactorCsv(path);
    EXPECT_TRUE(records.empty());
}

TEST_F(CsvLoaderTest, ParseQuotedExpressionWithCommas) {
    // Expressions like rolling_mean(close, 5) contain commas and
    // must be quoted so the CSV parser doesn't split on them.
    std::string path = writeTempCsv(
        "name,expression,description\n"
        "ma5,\"rolling_mean(close, 5)\",5-day SMA\n"
        "ma20,\"rolling_mean(close, 20)\",20-day SMA\n"
    );

    auto records = parseFactorCsv(path);
    ASSERT_EQ(records.size(), 2);

    EXPECT_EQ(records[0].name, "ma5");
    EXPECT_EQ(records[0].expression, "rolling_mean(close, 5)");
    EXPECT_EQ(records[0].description, "5-day SMA");

    EXPECT_EQ(records[1].name, "ma20");
    EXPECT_EQ(records[1].expression, "rolling_mean(close, 20)");
    EXPECT_EQ(records[1].description, "20-day SMA");
}

// ============================================================
// loadFactorsFromCsv tests
// ============================================================

TEST_F(CsvLoaderTest, LoadFactorsRegistersCorrectly) {
    std::string path = writeTempCsv(
        "name,expression,description\n"
        "f1,close,price\n"
        "f2,log(close),log price\n"
        "f3,close + open,composite\n"
    );

    std::size_t count = loadFactorsFromCsv(calc_, path);
    EXPECT_EQ(count, 3);

    auto names = calc_.formulas();
    EXPECT_EQ(names.size(), 3);
}

TEST_F(CsvLoaderTest, LoadedFactorsEvaluateCorrectly) {
    std::string path = writeTempCsv(
        "name,expression,description\n"
        "f_close,close,the close price\n"
        "f_log,log(close),log close\n"
    );

    loadFactorsFromCsv(calc_, path);

    // Evaluate "f_close"
    auto result1 = calc_.evaluate("f_close", md_);
    const auto& c = md_.column<double>(Field::CLOSE);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result1[i], c[i]);

    // Evaluate "f_log"
    auto result2 = calc_.evaluate("f_log", md_);
    for (std::size_t i = 0; i < kN; ++i)
        EXPECT_DOUBLE_EQ(result2[i], std::log(c[i]));
}

TEST_F(CsvLoaderTest, LoadFactorsWithRollingExpression) {
    std::string path = writeTempCsv(
        "name,expression,description\n"
        "ma5,\"rolling_mean(close, 5)\",5-day moving average\n"
    );

    loadFactorsFromCsv(calc_, path);
    auto result = calc_.evaluate("ma5", md_);

    ASSERT_EQ(result.size(), kN);
    // First 4 positions are NaN (window boundary)
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_TRUE(std::isnan(result[i]));
    // Position 4+ should be valid
    for (std::size_t i = 4; i < kN; ++i)
        EXPECT_FALSE(std::isnan(result[i]));
}

TEST_F(CsvLoaderTest, LoadFactorsReturnsCorrectExpression) {
    std::string path = writeTempCsv(
        "name,expression,description\n"
        "my_factor,log(close) * volume,log price times volume\n"
    );

    loadFactorsFromCsv(calc_, path);
    EXPECT_EQ(calc_.formulaExpression("my_factor"),
              "log(close) * volume");
}

TEST_F(CsvLoaderTest, InvalidExpressionThrows) {
    std::string path = writeTempCsv(
        "name,expression,description\n"
        "bad,unknown_func(close),bad factor\n"
    );

    EXPECT_THROW(loadFactorsFromCsv(calc_, path), std::runtime_error);
}

TEST_F(CsvLoaderTest, MissingFileThrows) {
    EXPECT_THROW(parseFactorCsv("/tmp/nonexistent_file_12345.csv"),
                 std::runtime_error);
}

// ============================================================
// Example CSV file test (real file in factors/)
// ============================================================

TEST_F(CsvLoaderTest, LoadExampleFactorsCsv) {
    // Use compile-time source directory (set via CMake) to locate the
    // example CSV regardless of ctest working directory.
    std::string examplePath =
        std::string(TEST_SOURCE_DIR) + "/factors/example_factors.csv";

    std::size_t count = loadFactorsFromCsv(calc_, examplePath);
    EXPECT_GE(count, 3);  // at least 3 example factors

    // Verify alpha0001 was loaded
    auto names = calc_.formulas();
    bool found = false;
    for (auto& n : names) {
        if (n == "alpha_0001") { found = true; break; }
    }
    EXPECT_TRUE(found) << "alpha_0001 should be in example_factors.csv";
}
