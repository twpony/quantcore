// PanelData.h — multi-asset aligned panel data container
// Phase: 五期实现
//
// PanelData holds a collection of MarketData objects with aligned timestamps,
// enabling both time-series (per-asset) and cross-sectional (per-date)
// access patterns.  Replaces the MarketDataBundle stub with a working
// implementation.
//
// Time-series access:   panel.asset(i) → const MarketData&  (zero-copy)
// Cross-sectional:      panel.crossSection(date, field, pool) → ColView<double>
//
// CrossSection materializes values from all assets at a single (date, field)
// into a contiguous buffer allocated from the given BufferPool.  The returned
// BufferHandle keeps the buffer alive; the ColView is a lightweight reference.
//
// Usage:
//   PanelData panel(std::move(assets));
//   for (size_t a = 0; a < panel.assetCount(); ++a)
//       engine.evaluate(ast, panel.asset(a));  // time-series
//   for (size_t d = 0; d < panel.dateCount(); ++d) {
//       auto cs = panel.crossSection(d, Field::CLOSE, pool);
//       reg.invokeCs(CsOpCode::CS_RANK, cs.view, output);  // cross-section
//   }
//
// All assets MUST share the same TimestampIndex.  Construction validates this
// and throws on misaligned timestamps.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/engine/BufferHandle.h"
#include "quantcore/engine/BufferPool.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

namespace quantcore {

class PanelData {
public:
    // ============================================================
    // Construction
    // ============================================================

    PanelData() = default;

    /// Construct from a vector of MarketData objects.
    /// All assets must have identical TimestampIndex (validated).
    /// @throws std::invalid_argument if timestamps are misaligned
    ///         or if assets is empty.
    explicit PanelData(std::vector<MarketData> assets);

    /// Add a single asset.  Timestamps must match existing assets.
    /// If the panel is empty, this asset sets the time axis.
    /// @throws std::invalid_argument if timestamps don't match.
    void addAsset(MarketData asset);

    /// Build from a vector.  Clears any existing data.
    void buildFrom(std::vector<MarketData> assets);

    /// Move-constructible, not copyable (large data).
    PanelData(const PanelData&) = delete;
    PanelData& operator=(const PanelData&) = delete;
    PanelData(PanelData&&) noexcept = default;
    PanelData& operator=(PanelData&&) noexcept = default;

    // ============================================================
    // Dimensions
    // ============================================================

    /// Number of assets in the panel.
    std::size_t assetCount() const noexcept { return assets_.size(); }

    /// Number of time points (same for all assets).
    std::size_t dateCount() const noexcept { return timestamps_.size(); }

    /// Number of fields (always 7 for OHLCV).
    static constexpr std::size_t fieldCount() noexcept {
        return static_cast<std::size_t>(Field::kFieldCount);
    }

    /// True if the panel is empty.
    bool empty() const noexcept { return assets_.empty(); }

    // ============================================================
    // Asset metadata
    // ============================================================

    /// Asset ID at index i.
    const std::string& assetId(std::size_t i) const;

    /// Find the index of an asset by ID.
    /// @throws std::out_of_range if not found.
    std::size_t assetIndex(const std::string& id) const;

    /// The shared timestamp axis.
    const TimestampIndex& timestamps() const noexcept { return timestamps_; }

    /// Timestamp at date index.
    int64_t timestampAt(std::size_t dateIdx) const;

    /// Date (YYYYMMDD) at date index.
    int64_t dateAt(std::size_t dateIdx) const;

    // ============================================================
    // Time-series access (时序方向)
    //
    // Returns a const reference to the existing MarketData for one
    // asset.  This is zero-copy — the data is already stored
    // contiguously in Column<T> within each MarketData.
    //
    // Use this for time-series factor computation via the existing
    // ExecutionEngine pipeline.
    // ============================================================

    const MarketData& asset(std::size_t i) const;

    /// Find asset by ID.  @throws std::out_of_range if not found.
    const MarketData& assetById(const std::string& id) const;

    // ============================================================
    // Cross-sectional access (截面方向)
    //
    // Materializes all assets' values for a single (dateIdx, field)
    // into a contiguous buffer from the given BufferPool.
    //
    // The returned struct bundles a BufferHandle (owns the buffer)
    // with a ColView (lightweight reference into the buffer).  Both
    // must stay in scope while the data is being read.
    //
    // Typical usage:
    //   auto cs = panel.crossSection(d, Field::CLOSE, pool);
    //   reg.invokeCs(CsOpCode::CS_RANK, cs.view, output, {});
    //   // cs.handle auto-releases buffer on scope exit
    // ============================================================

    struct CrossSection {
        BufferHandle<double> handle;  // Owns the materialized buffer
        ColView<double>      view;    // Lightweight reference into handle
    };

    CrossSection crossSection(std::size_t dateIdx,
                              Field field,
                              BufferPool& pool) const;

    // ============================================================
    // Validation & diagnostics
    // ============================================================

    /// Verify all assets have aligned timestamps and non-empty fields.
    bool isValid() const;

    /// Number of null values at a specific (dateIdx, field) across
    /// all assets.  Returns 0 if the field has no null mask.
    std::size_t nullCount(std::size_t dateIdx, Field field) const;

private:
    /// Validate that `asset` has the same timestamp axis as the panel.
    /// @throws std::invalid_argument on mismatch.
    void validateAsset(const MarketData& asset) const;

    /// Compare two timestamp indices for equality.
    static bool timestampsEqual(const TimestampIndex& a,
                                const TimestampIndex& b);

    std::vector<MarketData> assets_;
    std::unordered_map<std::string, std::size_t> assetIndex_;
    TimestampIndex timestamps_;
};

}  // namespace quantcore
