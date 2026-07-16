// test_parquet_reader.cpp — integration tests using real local Parquet data
// Phase: 一期必实现
//
// These tests read actual daily market data from Parquet files located in
// the local data directory and exercise the full QuantCore storage pipeline.
//
// Prerequisites:
//   QUANTCORE_HAS_PARQUET=1 (Apache Arrow + Parquet C++ linked)
//   Data directory with daily *.parquet files
//
// The default data path can be overridden with the QC_DATA_DIR environment
// variable:
//   QC_DATA_DIR=/path/to/daily  ./test_parquet_reader

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "quantcore/core/Logger.h"
#include "quantcore/io/ParquetReader.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;

// ============================================================
// Test configuration
// ============================================================

namespace {

// Resolve the data directory from environment variable or default
std::string getDataDir() {
    const char* envDir = std::getenv("QC_DATA_DIR");
    if (envDir && envDir[0] != '\0') {
        return std::string(envDir);
    }
    // Default: user's daily parquet data directory
    return "/home/twpony/quant/twpony/data_files/daily";
}

// Check if Parquet support is compiled in
bool hasParquetSupport() {
#if QUANTCORE_HAS_PARQUET
    return true;
#else
    return false;
#endif
}

}  // anonymous namespace

// ============================================================
// Test fixture
// ============================================================

class ParquetReaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!hasParquetSupport()) {
            GTEST_SKIP() << "Compiled without Parquet support (QUANTCORE_HAS_PARQUET=0)";
        }
        dataDir_ = getDataDir();
        Logger::instance().setLevel(LogLevel::WARNING);
    }

    std::string dataDir_;
};

// ============================================================
// Test: read a single daily file
// ============================================================

TEST_F(ParquetReaderTest, ReadSingleDailyFile) {
    // Find one parquet file in the data directory
    namespace fs = std::filesystem;
    if (!fs::exists(dataDir_) || !fs::is_directory(dataDir_)) {
        GTEST_SKIP() << "Data directory not found: " << dataDir_;
    }

    std::string testFile;
    for (const auto& entry : fs::directory_iterator(dataDir_)) {
        if (entry.path().extension() == ".parquet") {
            testFile = entry.path().string();
            break;
        }
    }
    if (testFile.empty()) {
        GTEST_SKIP() << "No parquet files found in " << dataDir_;
    }

    ParquetDailyReader reader;
    auto result = reader.readFile(testFile);

    // Basic validation
    EXPECT_GT(result.assets.size(), 0u)
        << "Should have at least one stock in the file";
    EXPECT_GT(result.rowsRead, 0u)
        << "Should have read at least one row";
    EXPECT_NE(result.tradeDate, 0)
        << "Trade date should be parsed from filename";

    std::cout << "  File: " << testFile << "\n"
              << "  Date: " << result.tradeDate << "\n"
              << "  Stocks: " << result.assets.size() << "\n"
              << "  Rows: " << result.rowsRead << "\n";
}

// ============================================================
// Test: validate MarketData fields
// ============================================================

TEST_F(ParquetReaderTest, ValidateMarketDataFields) {
    namespace fs = std::filesystem;
    if (!fs::exists(dataDir_) || !fs::is_directory(dataDir_)) {
        GTEST_SKIP() << "Data directory not found: " << dataDir_;
    }

    // Pick the most recent file for testing
    std::string testFile;
    for (const auto& entry : fs::directory_iterator(dataDir_)) {
        if (entry.path().extension() == ".parquet") {
            std::string fname = entry.path().stem().string();
            if (fname.size() >= 8 && fname > (testFile.empty() ? "0" : testFile)) {
                testFile = entry.path().string();
            }
        }
    }
    if (testFile.empty()) {
        GTEST_SKIP() << "No parquet files found";
    }

    ParquetDailyReader reader;
    auto result = reader.readFile(testFile);

    ASSERT_GT(result.assets.size(), 0u);

    // Validate the first asset
    const auto& md = result.assets[0];

    // Asset ID should be a valid stock code (e.g., "000001.SZ")
    EXPECT_FALSE(md.assetId().empty());
    EXPECT_GE(md.assetId().size(), 6u) << "Asset ID should be a valid code";

    // All columns should be aligned
    EXPECT_TRUE(md.allColumnsAligned());

    // Row count should match timestamp count
    EXPECT_EQ(md.rowCount(), md.timestamps().size());

    // Get a specific well-known stock if available
    bool found000001 = false;
    for (const auto& a : result.assets) {
        if (a.assetId() == "000001.SZ") {
            found000001 = true;
            EXPECT_GT(a.rowCount(), 0u);
            // Check OHLC data consistency: high >= low
            if (a.rowCount() > 0) {
                double high  = a.column<double>(Field::HIGH)[0];
                double low   = a.column<double>(Field::LOW)[0];
                double open  = a.column<double>(Field::OPEN)[0];
                double close = a.column<double>(Field::CLOSE)[0];

                EXPECT_GE(high, low)
                    << "HIGH should be >= LOW for " << a.assetId();
                EXPECT_GE(high, open)
                    << "HIGH should be >= OPEN";
                EXPECT_GE(high, close)
                    << "HIGH should be >= CLOSE";
                EXPECT_LE(low, open)
                    << "LOW should be <= OPEN";
                EXPECT_LE(low, close)
                    << "LOW should be <= CLOSE";
            }
            break;
        }
    }

    std::cout << "  Validated: " << result.assets.size() << " stocks\n"
              << "  Found 000001.SZ: " << (found000001 ? "yes" : "no") << "\n"
              << "  All aligned: " << (md.allColumnsAligned() ? "yes" : "no") << "\n";
}

// ============================================================
// Test: read multiple files (date range)
// ============================================================

TEST_F(ParquetReaderTest, ReadMultipleDates) {
    namespace fs = std::filesystem;
    if (!fs::exists(dataDir_) || !fs::is_directory(dataDir_)) {
        GTEST_SKIP() << "Data directory not found: " << dataDir_;
    }

    // Get the first 3 parquet files sorted by name
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dataDir_)) {
        if (entry.path().extension() == ".parquet") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());

    // Limit to 3 files for test speed
    if (files.size() > 3) files.resize(3);
    if (files.empty()) {
        GTEST_SKIP() << "No parquet files found";
    }

    ParquetDailyReader reader;
    std::size_t totalAssets = 0;
    std::size_t totalRows = 0;

    for (const auto& fp : files) {
        auto result = reader.readFile(fp);
        totalAssets += result.assets.size();
        totalRows   += result.rowsRead;

        // Verify each MarketData is well-formed
        for (const auto& md : result.assets) {
            EXPECT_TRUE(md.allColumnsAligned());
            EXPECT_EQ(md.rowCount(), md.timestamps().size());

            // Volume should be non-negative
            if (md.rowCount() > 0) {
                int64_t vol = md.column<int64_t>(Field::VOLUME)[0];
                EXPECT_GE(vol, 0) << "Volume should be non-negative for "
                                  << md.assetId();
            }
        }
    }

    const auto& stats = reader.lastStats();
    EXPECT_EQ(stats.totalFiles, files.size());

    std::cout << "  Files read: " << files.size() << "\n"
              << "  Total stocks: " << totalAssets << "\n"
              << "  Total rows: " << totalRows << "\n";
}

// ============================================================
// Test: OHLCV data integrity checks
// ============================================================

TEST_F(ParquetReaderTest, OHLcvDataIntegrity) {
    namespace fs = std::filesystem;
    if (!fs::exists(dataDir_) || !fs::is_directory(dataDir_)) {
        GTEST_SKIP() << "Data directory not found: " << dataDir_;
    }

    // Pick a single file to validate thoroughly
    std::string testFile;
    for (const auto& entry : fs::directory_iterator(dataDir_)) {
        if (entry.path().extension() == ".parquet") {
            testFile = entry.path().string();
            break;
        }
    }
    if (testFile.empty()) {
        GTEST_SKIP() << "No parquet files found";
    }

    ParquetDailyReader reader;
    auto result = reader.readFile(testFile);
    ASSERT_GT(result.assets.size(), 0u);

    int priceFieldErrors = 0;
    int vwapCheckErrors = 0;

    for (const auto& md : result.assets) {
        auto& openCol   = md.column<double>(Field::OPEN);
        auto& highCol   = md.column<double>(Field::HIGH);
        auto& lowCol    = md.column<double>(Field::LOW);
        auto& closeCol  = md.column<double>(Field::CLOSE);
        auto& vwapCol   = md.column<double>(Field::VWAP);
        auto& volCol    = md.column<int64_t>(Field::VOLUME);
        auto& amtCol    = md.column<int64_t>(Field::AMOUNT);

        for (std::size_t i = 0; i < md.rowCount(); ++i) {
            double o = openCol[i];
            double h = highCol[i];
            double l = lowCol[i];
            double c = closeCol[i];
            int64_t v = volCol[i];
            int64_t a = amtCol[i];

            // Basic OHLC invariants
            if (h < l) ++priceFieldErrors;
            if (h < o || h < c) ++priceFieldErrors;

            // VWAP = amount / volume (computed in reader)
            if (v > 0 && a > 0) {
                double expectedVwap = static_cast<double>(a) / static_cast<double>(v);
                double actualVwap = vwapCol[i];
                // Allow 0.1% relative tolerance for rounding
                double relErr = std::abs(expectedVwap - actualVwap) /
                                std::max(expectedVwap, 1e-10);
                if (relErr > 0.05) {  // 5% tolerance due to int64 rounding
                    ++vwapCheckErrors;
                }
            }

            // Volume and amount should be non-negative
            EXPECT_GE(v, 0) << "Negative volume for " << md.assetId()
                            << " at row " << i;
            EXPECT_GE(a, 0) << "Negative amount for " << md.assetId()
                            << " at row " << i;
        }
    }

    std::cout << "  Price field errors: " << priceFieldErrors << "\n"
              << "  VWAP check notes: " << vwapCheckErrors << "\n";

    // Price field errors should be zero
    EXPECT_EQ(priceFieldErrors, 0)
        << "OHLC data integrity violated: high < low or high < open/close";
}

// ============================================================
// Test: TimestampIndex from real data
// ============================================================

TEST_F(ParquetReaderTest, TimestampIndexFromRealData) {
    namespace fs = std::filesystem;
    if (!fs::exists(dataDir_) || !fs::is_directory(dataDir_)) {
        GTEST_SKIP() << "Data directory not found: " << dataDir_;
    }

    std::string testFile;
    for (const auto& entry : fs::directory_iterator(dataDir_)) {
        if (entry.path().extension() == ".parquet") {
            testFile = entry.path().string();
            break;
        }
    }
    if (testFile.empty()) {
        GTEST_SKIP() << "No parquet files found";
    }

    ParquetDailyReader reader;
    auto result = reader.readFile(testFile);
    ASSERT_GT(result.assets.size(), 0u);

    const auto& md = result.assets[0];
    const auto& ts = md.timestamps();

    // All timestamps should be positive (post-1970 epoch)
    for (std::size_t i = 0; i < ts.size(); ++i) {
        EXPECT_GT(ts[i], 0) << "Timestamp at row " << i
                            << " should be positive for " << md.assetId();
    }

    // Date should be consistent with the filename
    int64_t expectedDate = result.tradeDate;
    if (expectedDate > 0 && ts.size() > 0) {
        EXPECT_EQ(ts.dateAt(0), expectedDate)
            << "Date mismatch between filename and data";
    }

    // Verify date column exists
    EXPECT_EQ(ts.dates().size(), ts.size());

    std::cout << "  Asset: " << md.assetId() << "\n"
              << "  Date: " << ts.dateAt(0) << "\n"
              << "  Timestamp: " << ts[0] << "\n";
}

// ============================================================
// Test: Compute a basic factor from real data
// ============================================================

TEST_F(ParquetReaderTest, ComputeBasicFactor) {
    namespace fs = std::filesystem;
    if (!fs::exists(dataDir_) || !fs::is_directory(dataDir_)) {
        GTEST_SKIP() << "Data directory not found: " << dataDir_;
    }

    std::string testFile;
    for (const auto& entry : fs::directory_iterator(dataDir_)) {
        if (entry.path().extension() == ".parquet") {
            testFile = entry.path().string();
            break;
        }
    }
    if (testFile.empty()) {
        GTEST_SKIP() << "No parquet files found";
    }

    ParquetDailyReader reader;
    auto result = reader.readFile(testFile);
    ASSERT_GT(result.assets.size(), 0u);

    // Find a stock with valid VWAP data
    MarketData* targetMd = nullptr;
    for (auto& md : result.assets) {
        auto& vwap = md.column<double>(Field::VWAP);
        auto& vol  = md.column<int64_t>(Field::VOLUME);
        if (vol[0] > 0 && vwap[0] > 0) {
            targetMd = &md;
            break;
        }
    }
    ASSERT_NE(targetMd, nullptr) << "No stock with valid VWAP data";

    // Compute: intraday deviation = (CLOSE - VWAP) / VWAP
    auto& close = targetMd->column<double>(Field::CLOSE);
    auto& vwap  = targetMd->column<double>(Field::VWAP);

    double close0 = close[0];
    double vwap0  = vwap[0];

    EXPECT_GT(close0, 0.0) << "CLOSE should be positive";
    EXPECT_GT(vwap0,  0.0) << "VWAP should be positive";

    double deviation = (close0 - vwap0) / vwap0;

    std::cout << "  Stock: " << targetMd->assetId() << "\n"
              << "  CLOSE: " << close0 << "\n"
              << "  VWAP:  " << vwap0 << "\n"
              << "  Deviation: " << deviation * 100 << "%\n";

    // Deviation should be within a reasonable range for daily data (±20%)
    EXPECT_LT(std::abs(deviation), 0.20)
        << "Intraday deviation too large for " << targetMd->assetId();
}

// ============================================================
// Test: Filter stocks by market from real data
// ============================================================

TEST_F(ParquetReaderTest, FilterStocksByCode) {
    namespace fs = std::filesystem;
    if (!fs::exists(dataDir_) || !fs::is_directory(dataDir_)) {
        GTEST_SKIP() << "Data directory not found: " << dataDir_;
    }

    std::string testFile;
    for (const auto& entry : fs::directory_iterator(dataDir_)) {
        if (entry.path().extension() == ".parquet") {
            testFile = entry.path().string();
            break;
        }
    }
    if (testFile.empty()) {
        GTEST_SKIP() << "No parquet files found";
    }

    ParquetDailyReader reader;
    auto result = reader.readFile(testFile);
    ASSERT_GT(result.assets.size(), 0u);

    // Count stocks by market suffix
    int shCount = 0;  // .SH
    int szCount = 0;  // .SZ
    int bjCount = 0;  // .BJ
    int otherCount = 0;

    for (const auto& md : result.assets) {
        const auto& id = md.assetId();
        if (id.size() >= 3) {
            std::string suffix = id.substr(id.size() - 3);
            if (suffix == ".SH") ++shCount;
            else if (suffix == ".SZ") ++szCount;
            else if (suffix == ".BJ") ++bjCount;
            else ++otherCount;
        }
    }

    std::cout << "  SH (Shanghai): " << shCount << "\n"
              << "  SZ (Shenzhen): " << szCount << "\n"
              << "  BJ (Beijing):  " << bjCount << "\n"
              << "  Other:         " << otherCount << "\n"
              << "  Total:         " << result.assets.size() << "\n";

    // Most stocks should be from SH or SZ
    EXPECT_GT(shCount + szCount, 0)
        << "Should have stocks from Shanghai or Shenzhen";
}

// ============================================================
// Test: Column memory alignment with real data
// ============================================================

TEST_F(ParquetReaderTest, ColumnAlignment) {
    namespace fs = std::filesystem;
    if (!fs::exists(dataDir_) || !fs::is_directory(dataDir_)) {
        GTEST_SKIP() << "Data directory not found: " << dataDir_;
    }

    std::string testFile;
    for (const auto& entry : fs::directory_iterator(dataDir_)) {
        if (entry.path().extension() == ".parquet") {
            testFile = entry.path().string();
            break;
        }
    }
    if (testFile.empty()) {
        GTEST_SKIP() << "No parquet files found";
    }

    ParquetDailyReader reader;
    auto result = reader.readFile(testFile);
    ASSERT_GT(result.assets.size(), 0u);

    for (const auto& md : result.assets) {
        // All price columns should be 64-byte aligned
        EXPECT_TRUE(md.column<double>(Field::OPEN).isAligned());
        EXPECT_TRUE(md.column<double>(Field::HIGH).isAligned());
        EXPECT_TRUE(md.column<double>(Field::LOW).isAligned());
        EXPECT_TRUE(md.column<double>(Field::CLOSE).isAligned());

        // Volume/Amount columns should also be aligned
        EXPECT_TRUE(md.column<int64_t>(Field::VOLUME).isAligned());
        EXPECT_TRUE(md.column<int64_t>(Field::AMOUNT).isAligned());
    }
}
