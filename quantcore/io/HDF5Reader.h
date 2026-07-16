// HDF5Reader.h — read daily market data from HDF5 files
// Phase: 一期必实现
//
// Reads HDF5 files containing daily market data with the following
// expected dataset layout:
//
//   Layout A (single table):
//     /data  — compound dataset with fields:
//              ts_code (string), trade_date (int64 YYYYMMDD),
//              open, high, low, close, amount, volume (double)
//
//   Layout B (column-based):
//     /ts_code, /trade_date, /open, /high, /low, /close, /amount, /volume
//     — 1D datasets of equal length
//
// The reader detects the layout automatically and converts to MarketData
// objects keyed by ts_code.
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

namespace quantcore {

// ============================================================
// HDF5ReaderConfig
// ============================================================

struct HDF5ReaderConfig {
    /// Dataset path within the HDF5 file (default: "/data").
    std::string datasetPath = "/data";

    /// Column names (for Layout B — column-based datasets).
    /// If empty, the reader uses the default names below.
    std::string tsCodeColumn    = "ts_code";
    std::string tradeDateColumn = "trade_date";
    std::string openColumn      = "open";
    std::string highColumn      = "high";
    std::string lowColumn       = "low";
    std::string closeColumn     = "close";
    std::string volumeColumn    = "volume";
    std::string amountColumn    = "amount";

    /// Volume/amount values are stored as double (no rounding needed).
    /// Kept for backward API compatibility — no longer affects read behavior.
    bool roundVolumeAndAmount = false;

    /// Maximum rows to read (-1 = all).
    int64_t maxRows = -1;

    /// Skip rows with null values in OHLC fields.
    bool skipNullRows = true;
};

// ============================================================
// HDF5ReadResult
// ============================================================

struct HDF5ReadResult {
    /// MarketData objects, one per unique ts_code.
    std::vector<MarketData> assets;

    /// Trade date (YYYYMMDD) if a single date was detected.
    int64_t tradeDate = 0;

    /// Number of rows successfully read.
    std::size_t rowsRead = 0;

    /// Number of rows skipped.
    std::size_t rowsSkipped = 0;
};

// ============================================================
// HDF5Reader
// ============================================================

class HDF5Reader {
public:
    explicit HDF5Reader(const HDF5ReaderConfig& config = {});
    ~HDF5Reader();

    // Non-copyable, movable
    HDF5Reader(const HDF5Reader&) = delete;
    HDF5Reader& operator=(const HDF5Reader&) = delete;
    HDF5Reader(HDF5Reader&&) noexcept;
    HDF5Reader& operator=(HDF5Reader&&) noexcept;

    // ============================================================
    // Read operations
    // ============================================================

    /// Read a single HDF5 file.
    HDF5ReadResult readFile(const std::string& filepath);

    /// Read all HDF5 files in a directory.
    std::map<std::string, std::vector<std::pair<int64_t, MarketData>>>
    readDirectory(const std::string& dirPath);

    // ============================================================
    // Statistics
    // ============================================================

    struct Stats {
        std::size_t totalFiles  = 0;
        std::size_t totalRows   = 0;
        std::size_t totalAssets = 0;
    };

    const Stats& stats() const noexcept { return stats_; }

    // ============================================================
    // Configuration
    // ============================================================

    const HDF5ReaderConfig& config() const noexcept { return config_; }
    void setConfig(const HDF5ReaderConfig& config) { config_ = config; }

private:
    // Internal implementation (in .cpp, conditional on QUANTCORE_HAS_HDF5).
    struct Impl;
    std::unique_ptr<Impl> impl_;
    HDF5ReaderConfig config_;
    Stats stats_;
};

}  // namespace quantcore
