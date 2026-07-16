// MarketData.h — single-asset multi-field market data container
// Phase: 一期必实现
//
// MarketData bundles the 7 standard OHLCV+VWAP fields for one trading
// instrument together with a TimestampIndex.  Fields are stored in an
// array indexed by the `Field` enum — O(1) access with no hashing.
//
// Supported field types:
//   OPEN / HIGH / LOW / CLOSE / VWAP  → Column<double>
//   VOLUME / AMOUNT                    → Column<int64_t>
//
// Variant storage is used internally via ColumnDataVariant so that each
// field slot can hold its native type.  Accessors template the type at
// the call site and validate at runtime (Debug) or trust the caller
// (Release, for speed).
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <variant>

#include "quantcore/core/Types.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/storage/TimestampIndex.h"

namespace quantcore {

// ============================================================
// Variant types for heterogeneous field storage
// ============================================================

/// Holds any of the supported column types.
using ColumnDataVariant = std::variant<
    Column<double>,
    Column<int64_t>
>;

/// Read-only view variant (for MarketDataView).
using ColViewVariant = std::variant<
    ColView<double>,
    ColView<int64_t>
>;

// ============================================================
// Field type mapping helpers (compile-time)
// ============================================================

/// The native storage type for a given Field.
constexpr bool fieldIsDouble(Field f) noexcept {
    switch (f) {
        case Field::OPEN:   case Field::HIGH:
        case Field::LOW:    case Field::CLOSE:
        case Field::VWAP:
            return true;
        case Field::VOLUME: case Field::AMOUNT:
            return false;
        default:
            return true;  // Default to double for unknown
    }
}

// ============================================================
// MarketData
// ============================================================

class MarketDataView;  // forward

class MarketData {
public:
    // ============================================================
    // Construction
    // ============================================================

    /// Empty dataset (no asset, no data).
    MarketData() = default;

    /// Construct with asset identifier and time axis.
    MarketData(std::string assetId, TimestampIndex timestamps);

    // Movable, not copyable (large data).
    MarketData(const MarketData&) = delete;
    MarketData& operator=(const MarketData&) = delete;
    MarketData(MarketData&&) noexcept = default;
    MarketData& operator=(MarketData&&) noexcept = default;

    // ============================================================
    // Field access (typed)
    // ============================================================

    /// Mutable access to a field's column.  The template parameter must
    /// match the field's native type; a std::bad_variant_access is thrown
    /// on mismatch in Debug builds.
    template <typename T>
    Column<T>& column(Field field) {
        auto& var = columns_[static_cast<std::size_t>(field)];
        return std::get<Column<T>>(var);
    }

    /// Const access to a field's column.
    template <typename T>
    const Column<T>& column(Field field) const {
        const auto& var = columns_[static_cast<std::size_t>(field)];
        return std::get<Column<T>>(var);
    }

    /// Set a field's column data directly.
    template <typename T>
    void setColumn(Field field, Column<T> data) {
        columns_[static_cast<std::size_t>(field)] = std::move(data);
        updateNullFlag();
    }

    // ============================================================
    // Asset metadata
    // ============================================================

    const std::string& assetId() const noexcept { return assetId_; }
    void setAssetId(const std::string& id) { assetId_ = id; }

    const TimestampIndex& timestamps() const noexcept { return timestamps_; }

    // ============================================================
    // Capacity & validation
    // ============================================================

    /// Number of rows (derived from timestamp count).  This is the
    /// authoritative row count; all columns must match this length.
    std::size_t rowCount() const noexcept { return timestamps_.size(); }

    /// Verify that every non-empty column has the same number of rows
    /// as the timestamp index.
    bool allColumnsAligned() const;

    /// Raw variant access (for generic iteration over fields).
    const ColumnDataVariant& columnVariant(Field field) const {
        return columns_[static_cast<std::size_t>(field)];
    }

    /// True if any field has a null mask with at least one null.
    bool hasNullAnywhere() const noexcept { return hasNullAnywhere_; }

    // ============================================================
    // Slicing (zero-copy views)
    // ============================================================

    /// Create a zero-copy view over rows [start, end).
    MarketDataView slice(std::size_t start, std::size_t end) const;

    /// Create a zero-copy view over rows in [startDate, endDate].
    MarketDataView sliceByDate(int64_t startDate, int64_t endDate) const;

    // ============================================================
    // Field initialization helpers
    // ============================================================

    /// Allocate all 7 fields with `rowCount` elements (default-initialized).
    void allocateAllFields();

    /// Allocate a specific field.
    template <typename T>
    void allocateField(Field field, std::size_t size) {
        columns_[static_cast<std::size_t>(field)] = Column<T>(size);
    }

private:
    void updateNullFlag();

    std::string assetId_;
    std::array<ColumnDataVariant, static_cast<std::size_t>(Field::kFieldCount)> columns_;
    TimestampIndex timestamps_;
    bool hasNullAnywhere_ = false;
};

// ============================================================
// MarketDataView — zero-copy slice
// ============================================================
//
// MarketDataView is a lightweight, non-owning view into a sub-range of
// a MarketData.  The underlying MarketData must outlive all views derived
// from it — this is analogous to std::string_view semantics.

class TimestampIndexView {
public:
    TimestampIndexView() = default;

    TimestampIndexView(const int64_t* timestamps,
                       const int64_t* dates,
                       std::size_t size,
                       std::size_t start = 0)
        : timestamps_(timestamps + start)
        , dates_(dates + start)
        , size_(size)
    {}

    std::size_t size() const noexcept { return size_; }

    int64_t timestampAt(std::size_t i) const { return timestamps_[i]; }
    int64_t dateAt(std::size_t i)     const { return dates_[i]; }

    const int64_t* timestampData() const noexcept { return timestamps_; }
    const int64_t* dateData()     const noexcept { return dates_; }

private:
    const int64_t* timestamps_ = nullptr;
    const int64_t* dates_      = nullptr;
    std::size_t    size_       = 0;
};

class MarketDataView {
public:
    MarketDataView() = default;

    /// Number of rows in this view.
    std::size_t rowCount() const noexcept { return rowCount_; }

    /// Access a field as a zero-copy ColView.
    template <typename T>
    ColView<T> column(Field field) const {
        const auto& var = colViews_[static_cast<std::size_t>(field)];
        return std::get<ColView<T>>(var);
    }

    /// Copy of the asset identifier (small string).
    const std::string& assetId() const noexcept { return assetId_; }

    /// Timestamp index view for this slice.
    const TimestampIndexView& timestamps() const noexcept { return timestampView_; }

    // Internal: construct from a MarketData + row range
    static MarketDataView fromMarketData(const MarketData& md,
                                          std::size_t start,
                                          std::size_t end);

private:
    std::string assetId_;
    std::array<ColViewVariant, static_cast<std::size_t>(Field::kFieldCount)> colViews_;
    TimestampIndexView timestampView_;
    std::size_t rowCount_ = 0;
};

}  // namespace quantcore
