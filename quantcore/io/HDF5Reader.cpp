// HDF5Reader.cpp — read daily market data from HDF5 files
// Phase: 一期必实现
#include "HDF5Reader.h"

#if QUANTCORE_HAS_HDF5
#include <H5Cpp.h>

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
// Helper: YYYYMMDD → Unix timestamp
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
    tm_buf.tm_hour = 12;
    tm_buf.tm_min  = 0;
    tm_buf.tm_sec  = 0;
    tm_buf.tm_isdst = -1;

    return static_cast<int64_t>(std::mktime(&tm_buf));
}

static int64_t parseDateFromString(const std::string& s) {
    if (s.size() >= 8) {
        try { return std::stoll(s.substr(0, 8)); } catch (...) { return 0; }
    }
    return 0;
}

// ============================================================
// PIMPL implementation
// ============================================================

struct HDF5Reader::Impl {
    // Internal per-stock row buffer during HDF5 reading
    struct StockRow {
        int64_t timestamp;
        int64_t tsCodeIdx;
        double open, high, low, close;
        double volume, amount;
        bool valid;
    };
};

// ============================================================
// Construction
// ============================================================

HDF5Reader::HDF5Reader(const HDF5ReaderConfig& config)
    : impl_(std::make_unique<Impl>())
    , config_(config)
{}

HDF5Reader::~HDF5Reader() = default;
HDF5Reader::HDF5Reader(HDF5Reader&&) noexcept = default;
HDF5Reader& HDF5Reader::operator=(HDF5Reader&&) noexcept = default;

// ============================================================
// Helper: read a 1D dataset from an HDF5 group/file
// ============================================================

namespace {

// Read a 1D double dataset
std::vector<double> readDoubleDataset(const H5::H5File& file,
                                       const std::string& name,
                                       std::size_t expectedSize) {
    std::vector<double> result;
    try {
        H5::DataSet ds = file.openDataSet(name);
        H5::DataSpace space = ds.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims, nullptr);
        if (dims[0] == 0) return result;

        result.resize(static_cast<std::size_t>(dims[0]));
        ds.read(result.data(), H5::PredType::NATIVE_DOUBLE);
    } catch (const H5::Exception& e) {
        QC_WARN(std::string("HDF5Reader: cannot read dataset '") + name +
                "': " + e.getDetailMsg());
    }
    return result;
}

// Read a 1D int64 dataset
std::vector<int64_t> readInt64Dataset(const H5::H5File& file,
                                       const std::string& name) {
    std::vector<int64_t> result;
    try {
        H5::DataSet ds = file.openDataSet(name);
        H5::DataSpace space = ds.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims, nullptr);
        if (dims[0] == 0) return result;

        result.resize(static_cast<std::size_t>(dims[0]));
        ds.read(result.data(), H5::PredType::NATIVE_INT64);
    } catch (const H5::Exception& e) {
        QC_WARN(std::string("HDF5Reader: cannot read dataset '") + name +
                "': " + e.getDetailMsg());
    }
    return result;
}

// Read a 1D string dataset (returns vector of std::string)
std::vector<std::string> readStringDataset(const H5::H5File& file,
                                            const std::string& name) {
    std::vector<std::string> result;
    try {
        H5::DataSet ds = file.openDataSet(name);
        H5::DataSpace space = ds.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims, nullptr);
        if (dims[0] == 0) return result;

        H5::DataType dtype = ds.getDataType();
        std::size_t strSize = dtype.getSize();
        std::vector<char> buffer(static_cast<std::size_t>(dims[0]) * strSize);
        ds.read(buffer.data(), dtype);

        for (hsize_t i = 0; i < dims[0]; ++i) {
            const char* ptr = buffer.data() + i * strSize;
            result.push_back(std::string(ptr, strnlen(ptr, strSize)));
        }
    } catch (const H5::Exception& e) {
        QC_WARN(std::string("HDF5Reader: cannot read dataset '") + name +
                "': " + e.getDetailMsg());
    }
    return result;
}

}  // anonymous namespace

// ============================================================
// Read a single file
// ============================================================

HDF5ReadResult HDF5Reader::readFile(const std::string& filepath) {
    HDF5ReadResult result;

    // Turn off HDF5's default error printing (we handle via exceptions)
    H5::Exception::dontPrint();

    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        // Attempt Layout B (column-based): look for individual datasets
        // under the datasetPath group or at root level.
        std::string prefix = config_.datasetPath;
        if (prefix == "/") prefix = "";
        if (!prefix.empty() && prefix.back() != '/') prefix += "/";

        // Try reading as column-based layout
        auto tsCodes   = readStringDataset(file, prefix + config_.tsCodeColumn);
        auto tradeDt   = readInt64Dataset(file, prefix + config_.tradeDateColumn);
        auto opens     = readDoubleDataset(file, prefix + config_.openColumn, 0);
        auto highs     = readDoubleDataset(file, prefix + config_.highColumn, 0);
        auto lows      = readDoubleDataset(file, prefix + config_.lowColumn, 0);
        auto closes    = readDoubleDataset(file, prefix + config_.closeColumn, 0);
        auto volumes   = readDoubleDataset(file, prefix + config_.volumeColumn, 0);
        auto amounts   = readDoubleDataset(file, prefix + config_.amountColumn, 0);

        std::size_t numRows = opens.size();
        if (numRows == 0) {
            // No data found at this dataset path; try alternative layout
            QC_WARN("HDF5Reader: no data found at dataset path '" +
                    config_.datasetPath + "' in " + filepath);
            return result;
        }

        // Clamp row count
        if (config_.maxRows > 0 && static_cast<int64_t>(numRows) > config_.maxRows) {
            numRows = static_cast<std::size_t>(config_.maxRows);
        }

        // Gather unique ts_codes
        std::vector<std::string> tsCodeStrings;
        std::unordered_map<std::string, int> tsCodeToIdx;

        using StockRow = Impl::StockRow;
        std::vector<StockRow> rows;
        rows.reserve(numRows);

        for (std::size_t i = 0; i < numRows; ++i) {
            StockRow row;
            row.valid = true;

            // ts_code
            if (i < tsCodes.size()) {
                const auto& code = tsCodes[i];
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

            // trade_date → timestamp
            if (i < tradeDt.size() && tradeDt[i] > 0) {
                row.timestamp = dateToUnixTimestamp(tradeDt[i]);
            } else {
                row.timestamp = 0;
            }

            // OHLC
            if (i < opens.size() && i < highs.size() &&
                i < lows.size() && i < closes.size()) {
                row.open  = opens[i];
                row.high  = highs[i];
                row.low   = lows[i];
                row.close = closes[i];
            } else {
                row.valid = false;
            }

            // volume / amount
            if (i < volumes.size()) row.volume = volumes[i];
            else row.volume = 0.0;
            if (i < amounts.size()) row.amount = amounts[i];
            else row.amount = 0.0;

            if (!row.valid && config_.skipNullRows) {
                ++result.rowsSkipped;
                continue;
            }
            rows.push_back(row);
        }

        // Organize by stock
        int numStocks = static_cast<int>(tsCodeStrings.size());
        std::vector<std::vector<StockRow>> stockRows(numStocks);
        for (auto& sr : rows) {
            if (sr.tsCodeIdx >= 0 && sr.tsCodeIdx < numStocks) {
                stockRows[static_cast<std::size_t>(sr.tsCodeIdx)].push_back(sr);
            }
        }

        // Build MarketData per stock
        for (int s = 0; s < numStocks; ++s) {
            const auto& srows = stockRows[static_cast<std::size_t>(s)];
            if (srows.empty()) continue;

            std::size_t n = srows.size();
            std::vector<int64_t> timestamps(n);
            for (std::size_t i = 0; i < n; ++i) {
                timestamps[i] = srows[i].timestamp;
            }
            TimestampIndex tsIdx(timestamps.data(), n);

            MarketData md(tsCodeStrings[static_cast<std::size_t>(s)], std::move(tsIdx));
            md.allocateAllFields();

            for (std::size_t i = 0; i < n; ++i) {
                const auto& sr = srows[i];
                md.column<double>(Field::OPEN)[i]   = sr.open;
                md.column<double>(Field::HIGH)[i]   = sr.high;
                md.column<double>(Field::LOW)[i]    = sr.low;
                md.column<double>(Field::CLOSE)[i]  = sr.close;

                double vwap = 0.0;
                if (sr.volume > 0 && sr.amount > 0) {
                    vwap = sr.amount / sr.volume;
                }
                md.column<double>(Field::VWAP)[i] = vwap;

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

            // Detect trade date from the data
            if (!srows.empty() && !tradeDt.empty()) {
                result.tradeDate = tradeDt[0];
            }

            result.assets.push_back(std::move(md));
        }

        result.rowsRead = rows.size();

        stats_.totalFiles++;
        stats_.totalRows   += result.rowsRead;
        stats_.totalAssets += result.assets.size();

        QC_INFO("HDF5Reader: read " + filepath +
                " — " + std::to_string(result.assets.size()) + " stocks, " +
                std::to_string(result.rowsRead) + " rows");

    } catch (const H5::FileIException& e) {
        QC_ERROR("HDF5Reader: file error: " + filepath + " (" +
                 e.getDetailMsg() + ")");
    } catch (const H5::Exception& e) {
        QC_ERROR("HDF5Reader: HDF5 error: " + std::string(e.getDetailMsg()));
    }

    return result;
}

// ============================================================
// Batch directory read
// ============================================================

std::map<std::string, std::vector<std::pair<int64_t, MarketData>>>
HDF5Reader::readDirectory(const std::string& dirPath) {
    std::map<std::string, std::vector<std::pair<int64_t, MarketData>>> stockMap;

    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        QC_ERROR("HDF5Reader: directory not found: " + dirPath);
        return stockMap;
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        auto ext = entry.path().extension();
        if (ext == ".h5" || ext == ".hdf5" || ext == ".hd5") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

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

#else  // !QUANTCORE_HAS_HDF5

// Stub implementation when HDF5 not available
#include "quantcore/core/Logger.h"

namespace quantcore {

HDF5Reader::HDF5Reader(const HDF5ReaderConfig&)
    : impl_(nullptr) {}
HDF5Reader::~HDF5Reader() = default;
HDF5Reader::HDF5Reader(HDF5Reader&&) noexcept = default;
HDF5Reader& HDF5Reader::operator=(HDF5Reader&&) noexcept = default;

HDF5ReadResult HDF5Reader::readFile(const std::string&) {
    QC_ERROR("HDF5Reader: compiled without HDF5 support (QUANTCORE_HAS_HDF5=0)");
    return {};
}

std::map<std::string, std::vector<std::pair<int64_t, MarketData>>>
HDF5Reader::readDirectory(const std::string&) {
    QC_ERROR("HDF5Reader: compiled without HDF5 support");
    return {};
}

}  // namespace quantcore

#endif  // QUANTCORE_HAS_HDF5
