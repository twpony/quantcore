// MarketDataBundle.h — multi-asset aligned panel data (多资产对齐面板)
// Phase: 远期预留接口（仅声明类型，不实现具体逻辑）
//
// MarketDataBundle holds a collection of MarketData objects with aligned
// timestamps, enabling cross-sectional operations across assets.
//
// This is a required dependency for CrossSectionOperator.
//
// Future implementation notes:
//   - All assets must share the same timestamp axis (or have a mapping)
//   - Asset count N is typically large (hundreds to thousands)
//   - Zero-copy views preferred — MarketDataView for each asset
//   - Efficient column-major access pattern for SIMD cross-section ops
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace quantcore {

class MarketData;      // forward — defined in storage/MarketData.h
class MarketDataView;   // forward — defined in storage/MarketData.h

class MarketDataBundle {
public:
    // ============================================================
    // Construction
    // ============================================================

    MarketDataBundle() = default;

    // Future:
    // explicit MarketDataBundle(std::vector<MarketData> assets);
    //
    // Construction validates:
    //   - All assets have the same timestamp length
    //   - Timestamps are aligned (identical dates at each row index)
    //   - At least 1 asset

    // ============================================================
    // Capacity
    // ============================================================

    /// Number of assets in the bundle (远期).
    std::size_t assetCount() const noexcept { return 0; }

    /// Number of time points (远期).
    std::size_t timePointCount() const noexcept { return 0; }

    // ============================================================
    // Asset access (远期)
    // ============================================================

    // Future:
    // const MarketDataView& asset(std::size_t index) const;
    // const std::string&    assetId(std::size_t index) const;
    //
    // template <typename T>
    // ColView<T> column(Field field, std::size_t assetIndex) const;
    //
    // /// Extract all assets' values for a single field at a single
    // /// time point — the fundamental access pattern for
    // /// CrossSectionOperator.
    // std::vector<double> crossSection(
    //     Field field, std::size_t timeIndex) const;

private:
    // Future:
    // std::vector<MarketData> assets_;
    // bool timestampAligned_ = true;
};

}  // namespace quantcore
