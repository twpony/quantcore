// ParquetReader.h — read daily market data from Parquet files via Apache Arrow
// Phase: 一期必实现
//
// Reads parquet files with schema:
//   ts_code: string    — stock identifier (e.g. "000001.SZ")
//   trade_date: string — trade date (e.g. "20260626")
//   open, high, low, close: double
//   vol: double        — volume (shares)
//   amount: double     — turnover (CNY)
//
// Each file contains one trading day's data for ~5000 stocks.
// The reader converts each stock into a MarketData object.  For multi-day
// usage, callers should accumulate across files or use the batch helper.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

// Forward-declare Arrow types we need (avoids heavy header inclusion for users)
namespace arrow {
    class Table;
    class ChunkedArray;
}
namespace parquet::arrow {
    class FileReader;
}

namespace quantcore {

// ============================================================
// ParquetDailyReader
// ============================================================

/// Configuration for daily parquet file reading.
struct ParquetReaderConfig {
    /// Column name for stock code (default: "ts_code").
    std::string tsCodeColumn     = "ts_code";

    /// Column name for trade date (default: "trade_date").
    /// Expected format: "YYYYMMDD" string or int64.
    std::string tradeDateColumn  = "trade_date";

    /// Column names for OHLC price fields.
    std::string openColumn       = "open";
    std::string highColumn       = "high";
    std::string lowColumn        = "low";
    std::string closeColumn      = "close";

    /// Column names for volume and amount.
    std::string volumeColumn     = "vol";
    std::string amountColumn     = "amount";

    /// Volume/amount values are stored as double (no rounding needed).
    /// Kept for backward API compatibility — no longer affects read behavior.
    bool roundVolumeAndAmount = false;

    /// Maximum rows to read per file (-1 = all).
    int64_t maxRows = -1;

    /// If true, skip rows where any OHLCV field is null.
    bool skipNullRows = true;
};

/// Result of reading a daily parquet file.
struct ParquetReadResult {
    /// MarketData objects keyed by ts_code.
    std::vector<MarketData> assets;

    /// The trade date (YYYYMMDD) parsed from the file.
    int64_t tradeDate = 0;

    /// Number of rows successfully read.
    std::size_t rowsRead = 0;

    /// Number of rows skipped (nulls, parse errors).
    std::size_t rowsSkipped = 0;
};

// ============================================================
// ParquetDailyReader
// ============================================================

class ParquetDailyReader {
public:
    explicit ParquetDailyReader(const ParquetReaderConfig& config = {});
    ~ParquetDailyReader();

    // Non-copyable, movable
    ParquetDailyReader(const ParquetDailyReader&) = delete;
    ParquetDailyReader& operator=(const ParquetDailyReader&) = delete;
    ParquetDailyReader(ParquetDailyReader&&) noexcept;
    ParquetDailyReader& operator=(ParquetDailyReader&&) noexcept;

    // ============================================================
    // Single-file read
    // ============================================================

    /// Read a single daily parquet file.
    /// Returns one MarketData per unique ts_code in the file.
    ParquetReadResult readFile(const std::string& filepath);

    // ============================================================
    // Batch / directory read
    // ============================================================

    /// Read all parquet files matching a pattern (e.g. "/data/daily/*.parquet").
    /// Returns a map: ts_code → vector of <date, MarketData slice> pairs
    /// sorted by date ascending.
    ///
    /// Note: this reads data across dates and organizes by stock.
    /// For very large datasets, prefer streaming per-file with readFile().
    std::map<std::string, std::vector<std::pair<int64_t, MarketData>>>
    readDirectory(const std::string& dirPath);

    /// Read a date range from a directory of daily parquet files.
    /// Files are named "YYYYMMDD.parquet" within dirPath.
    std::string readDateRange(const std::string& dirPath,
                              int64_t startDate,
                              int64_t endDate);

    // ============================================================
    // Statistics
    // ============================================================

    struct Stats {
        std::size_t totalFiles = 0;
        std::size_t totalRows  = 0;
        std::size_t totalAssets = 0;
    };

    const Stats& lastStats() const noexcept { return stats_; }

    // ============================================================
    // Configuration
    // ============================================================

    const ParquetReaderConfig& config() const noexcept { return config_; }
    void setConfig(const ParquetReaderConfig& config) { config_ = config; }

private:
    // Internal: parse a single Arrow Table into MarketData objects.
    ParquetReadResult tableToMarketData(const arrow::Table& table,
                                         int64_t tradeDate);

    // Parse YYYYMMDD from a string or int64.
    static int64_t parseDate(const std::string& s);
    static int64_t parseDateInt(int64_t dateInt);

    // Extract a column by name from an Arrow table.
    // Returns nullptr if column not found.

    ParquetReaderConfig config_;
    Stats stats_;
};

}  // namespace quantcore
