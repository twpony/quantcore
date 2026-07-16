// ParquetReader.cpp — read daily market data from Parquet files
// Phase: 一期必实现
#include "ParquetReader.h"

#if QUANTCORE_HAS_PARQUET
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "quantcore/core/ErrorHandling.h"
#include "quantcore/core/Logger.h"

namespace fs = std::filesystem;

namespace quantcore {

// ============================================================
// Construction
// ============================================================

ParquetDailyReader::ParquetDailyReader(const ParquetReaderConfig& config)
    : config_(config)
{}

ParquetDailyReader::~ParquetDailyReader() = default;

ParquetDailyReader::ParquetDailyReader(ParquetDailyReader&&) noexcept = default;
ParquetDailyReader& ParquetDailyReader::operator=(ParquetDailyReader&&) noexcept = default;

// ============================================================
// Date parsing helpers
// ============================================================

int64_t ParquetDailyReader::parseDate(const std::string& s) {
    // Expect "YYYYMMDD"
    if (s.size() >= 8) {
        try {
            return std::stoll(s.substr(0, 8));
        } catch (...) {
            return 0;
        }
    }
    return 0;
}

int64_t ParquetDailyReader::parseDateInt(int64_t dateInt) {
    // Already YYYYMMDD
    return dateInt;
}

// ============================================================
// Unix timestamp from YYYYMMDD (UTC noon)
// ============================================================

static int64_t dateToUnixTimestamp(int64_t yyyymmdd) {
    int year  = static_cast<int>(yyyymmdd / 10000);
    int month = static_cast<int>((yyyymmdd / 100) % 100);
    int day   = static_cast<int>(yyyymmdd % 100);

    if (year < 1970 || month < 1 || month > 12 || day < 1 || day > 31) {
        return 0;
    }

    std::tm tm_buf{};
    tm_buf.tm_year = year - 1900;
    tm_buf.tm_mon  = month - 1;
    tm_buf.tm_mday = day;
    tm_buf.tm_hour = 12;  // noon UTC to avoid DST edge cases
    tm_buf.tm_min  = 0;
    tm_buf.tm_sec  = 0;
    tm_buf.tm_isdst = -1;

    return static_cast<int64_t>(std::mktime(&tm_buf));
}

// ============================================================
// Table → MarketData conversion
// ============================================================

ParquetReadResult ParquetDailyReader::tableToMarketData(
    const arrow::Table& table, int64_t tradeDate) {

    ParquetReadResult result;
    result.tradeDate = tradeDate;

    int numRows = static_cast<int>(table.num_rows());
    if (numRows == 0) return result;

    // Clamp row count
    if (config_.maxRows > 0 && numRows > config_.maxRows) {
        numRows = static_cast<int>(config_.maxRows);
    }

    // Resolve column indices
    auto getCol = [&](const std::string& name) -> std::shared_ptr<arrow::ChunkedArray> {
        auto col = table.GetColumnByName(name);
        if (!col) {
            QC_WARN("ParquetDailyReader: column '" + name + "' not found in file");
        }
        return col;
    };

    auto tsCodeArr   = getCol(config_.tsCodeColumn);
    auto dateArr     = getCol(config_.tradeDateColumn);
    auto openArr     = getCol(config_.openColumn);
    auto highArr     = getCol(config_.highColumn);
    auto lowArr      = getCol(config_.lowColumn);
    auto closeArr    = getCol(config_.closeColumn);
    auto volumeArr   = getCol(config_.volumeColumn);
    auto amountArr   = getCol(config_.amountColumn);

    if (!openArr || !highArr || !lowArr || !closeArr) {
        QC_ERROR("ParquetDailyReader: required OHLC columns missing");
        return result;
    }

    // Gather data per ts_code into temporary buffers
    struct StockRow {
        int64_t timestamp;
        int64_t tsCodeIdx;
        double open, high, low, close;
        double volume, amount;
        bool valid;
    };

    // First pass: identify unique ts_codes and collect rows
    std::vector<std::string> tsCodeStrings;
    std::unordered_map<std::string, int> tsCodeToIdx;
    std::vector<StockRow> rows;
    rows.reserve(static_cast<std::size_t>(numRows));

    for (int i = 0; i < numRows; ++i) {
        StockRow row;
        row.valid = true;

        // ts_code (string)
        if (tsCodeArr) {
            auto chunked = std::static_pointer_cast<arrow::StringArray>(
                tsCodeArr->chunk(0));
            if (chunked && i < chunked->length() && !chunked->IsNull(i)) {
                std::string code = chunked->GetString(i);
                auto it = tsCodeToIdx.find(code);
                if (it == tsCodeToIdx.end()) {
                    row.tsCodeIdx = static_cast<int64_t>(tsCodeStrings.size());
                    tsCodeToIdx[code] = row.tsCodeIdx;
                    tsCodeStrings.push_back(code);
                } else {
                    row.tsCodeIdx = it->second;
                }
            } else {
                row.valid = false;
            }
        }

        // trade_date → timestamp
        if (dateArr) {
            auto chunked = std::static_pointer_cast<arrow::StringArray>(
                dateArr->chunk(0));
            if (chunked && i < chunked->length() && !chunked->IsNull(i)) {
                std::string dateStr = chunked->GetString(i);
                int64_t d = parseDate(dateStr);
                row.timestamp = dateToUnixTimestamp(d);
            } else {
                row.timestamp = 0;
            }
        } else {
            row.timestamp = dateToUnixTimestamp(tradeDate);
        }

        // OHLC (double)
        auto readDouble = [&](std::shared_ptr<arrow::ChunkedArray>& arr,
                              double& out) -> bool {
            if (!arr) return false;
            auto chunked = std::static_pointer_cast<arrow::DoubleArray>(
                arr->chunk(0));
            if (!chunked || i >= chunked->length() || chunked->IsNull(i)) {
                return false;
            }
            out = chunked->Value(i);
            return true;
        };

        if (!readDouble(openArr,   row.open))   row.valid = false;
        if (!readDouble(highArr,   row.high))   row.valid = false;
        if (!readDouble(lowArr,    row.low))    row.valid = false;
        if (!readDouble(closeArr,  row.close))  row.valid = false;

        // volume / amount (double, convert to int64_t for storage)
        readDouble(volumeArr, row.volume);
        readDouble(amountArr, row.amount);

        // Skip invalid rows if configured
        if (!row.valid && config_.skipNullRows) {
            ++result.rowsSkipped;
            continue;
        }

        rows.push_back(row);
    }

    // Second pass: organize by stock
    int numStocks = static_cast<int>(tsCodeStrings.size());
    std::vector<std::vector<StockRow>> stockRows(numStocks);
    for (auto& sr : rows) {
        if (sr.tsCodeIdx >= 0 && sr.tsCodeIdx < numStocks) {
            stockRows[static_cast<std::size_t>(sr.tsCodeIdx)].push_back(sr);
        }
    }

    // Build MarketData for each stock
    for (int s = 0; s < numStocks; ++s) {
        const auto& srows = stockRows[static_cast<std::size_t>(s)];
        if (srows.empty()) continue;

        std::size_t n = srows.size();

        // Build timestamp column
        std::vector<int64_t> timestamps(n);
        for (std::size_t i = 0; i < n; ++i) {
            timestamps[i] = srows[i].timestamp;
        }
        TimestampIndex tsIdx(timestamps.data(), n);

        MarketData md(tsCodeStrings[static_cast<std::size_t>(s)], std::move(tsIdx));
        md.allocateAllFields();

        // Fill columns
        for (std::size_t i = 0; i < n; ++i) {
            const auto& sr = srows[i];
            md.column<double>(Field::OPEN)[i]   = sr.open;
            md.column<double>(Field::HIGH)[i]   = sr.high;
            md.column<double>(Field::LOW)[i]    = sr.low;
            md.column<double>(Field::CLOSE)[i]  = sr.close;

            // VWAP = amount / volume (if both are positive)
            double vwap = 0.0;
            if (sr.volume > 0 && sr.amount > 0) {
                vwap = sr.amount / sr.volume;
            }
            md.column<double>(Field::VWAP)[i] = vwap;

            // Volume and Amount: round to int64_t per config
            if (config_.roundVolumeAndAmount) {
                md.column<int64_t>(Field::VOLUME)[i] =
                    static_cast<int64_t>(std::llround(sr.volume));
                md.column<int64_t>(Field::AMOUNT)[i] =
                    static_cast<int64_t>(std::llround(sr.amount));
            } else {
                md.column<int64_t>(Field::VOLUME)[i] =
                    static_cast<int64_t>(sr.volume);
                md.column<int64_t>(Field::AMOUNT)[i] =
                    static_cast<int64_t>(sr.amount);
            }
        }

        result.assets.push_back(std::move(md));
    }

    result.rowsRead = rows.size();
    return result;
}

// ============================================================
// Single-file read
// ============================================================

ParquetReadResult ParquetDailyReader::readFile(const std::string& filepath) {
    ParquetReadResult result;

    // Open the file
    auto maybeInput = arrow::io::ReadableFile::Open(filepath);
    if (!maybeInput.ok()) {
        QC_ERROR("ParquetDailyReader: cannot open file: " + filepath +
                 " (" + maybeInput.status().ToString() + ")");
        return result;
    }
    auto input = maybeInput.ValueOrDie();

    // Create Parquet reader (modern Arrow API)
    auto readerResult = parquet::arrow::FileReader::Make(
        arrow::default_memory_pool(),
        parquet::ParquetFileReader::Open(input));
    if (!readerResult.ok()) {
        QC_ERROR("ParquetDailyReader: cannot create reader for: " + filepath +
                 " (" + readerResult.status().ToString() + ")");
        return result;
    }
    auto reader = std::move(*readerResult);

    // Parse trade date from filename: "YYYYMMDD.parquet"
    int64_t tradeDate = 0;
    fs::path fp(filepath);
    std::string stem = fp.stem().string();  // "20260626"
    if (stem.size() >= 8) {
        tradeDate = parseDate(stem);
    }

    // Read the full table (modern Arrow API)
    auto tableResult = reader->ReadTable();
    if (!tableResult.ok()) {
        QC_ERROR("ParquetDailyReader: failed to read table: " +
                 tableResult.status().ToString());
        return result;
    }
    auto table = std::move(*tableResult);

    result = tableToMarketData(*table, tradeDate);
    stats_.totalFiles++;
    stats_.totalRows   += result.rowsRead;
    stats_.totalAssets += result.assets.size();

    QC_INFO("ParquetDailyReader: read " + filepath +
            " — " + std::to_string(result.assets.size()) + " stocks, " +
            std::to_string(result.rowsRead) + " rows, date=" +
            std::to_string(tradeDate));

    return result;
}

// ============================================================
// Batch directory read
// ============================================================

std::map<std::string, std::vector<std::pair<int64_t, MarketData>>>
ParquetDailyReader::readDirectory(const std::string& dirPath) {
    std::map<std::string, std::vector<std::pair<int64_t, MarketData>>> stockMap;

    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        QC_ERROR("ParquetDailyReader: directory not found: " + dirPath);
        return stockMap;
    }

    // Collect and sort parquet files
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.path().extension() == ".parquet") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    QC_INFO("ParquetDailyReader: reading " + std::to_string(files.size()) +
            " parquet files from " + dirPath);

    for (const auto& fp : files) {
        auto result = readFile(fp.string());
        for (auto& md : result.assets) {
            stockMap[md.assetId()].push_back(
                {result.tradeDate, std::move(md)});
        }
    }

    return stockMap;
}

}  // namespace quantcore

#else  // !QUANTCORE_HAS_PARQUET

// Stub implementation when Arrow/Parquet not available
#include "quantcore/core/Logger.h"

namespace quantcore {

ParquetDailyReader::ParquetDailyReader(const ParquetReaderConfig&) {}
ParquetDailyReader::~ParquetDailyReader() = default;
ParquetDailyReader::ParquetDailyReader(ParquetDailyReader&&) noexcept = default;
ParquetDailyReader& ParquetDailyReader::operator=(ParquetDailyReader&&) noexcept = default;

ParquetReadResult ParquetDailyReader::readFile(const std::string&) {
    QC_ERROR("ParquetDailyReader: compiled without Parquet support (QUANTCORE_HAS_PARQUET=0)");
    return {};
}

std::map<std::string, std::vector<std::pair<int64_t, MarketData>>>
ParquetDailyReader::readDirectory(const std::string&) {
    QC_ERROR("ParquetDailyReader: compiled without Parquet support");
    return {};
}

}  // namespace quantcore

#endif  // QUANTCORE_HAS_PARQUET
