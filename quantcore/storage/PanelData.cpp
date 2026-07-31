// PanelData.cpp — multi-asset aligned panel implementation
// Phase: 五期实现

#include "quantcore/storage/PanelData.h"

#include <cstring>
#include <sstream>

namespace quantcore {

// ============================================================
// Construction
// ============================================================

PanelData::PanelData(std::vector<MarketData> assets) {
    buildFrom(std::move(assets));
}

void PanelData::addAsset(MarketData asset) {
    if (assets_.empty()) {
        // First asset sets the time axis.
        timestamps_ = asset.timestamps();
    } else {
        validateAsset(asset);
    }

    std::size_t idx = assets_.size();
    assetIndex_[asset.assetId()] = idx;
    assets_.push_back(std::move(asset));
}

void PanelData::buildFrom(std::vector<MarketData> assets) {
    if (assets.empty()) {
        throw std::invalid_argument("PanelData: assets vector is empty");
    }

    assets_.clear();
    assetIndex_.clear();

    // Use the first asset to establish the time axis.
    timestamps_ = assets[0].timestamps();

    for (auto& a : assets) {
        addAsset(std::move(a));
    }
}

// ============================================================
// Asset metadata
// ============================================================

const std::string& PanelData::assetId(std::size_t i) const {
    return assets_[i].assetId();
}

std::size_t PanelData::assetIndex(const std::string& id) const {
    auto it = assetIndex_.find(id);
    if (it == assetIndex_.end()) {
        throw std::out_of_range("PanelData: asset '" + id + "' not found");
    }
    return it->second;
}

int64_t PanelData::timestampAt(std::size_t dateIdx) const {
    return timestamps_[dateIdx];
}

int64_t PanelData::dateAt(std::size_t dateIdx) const {
    return timestamps_.dateAt(dateIdx);
}

// ============================================================
// Time-series access
// ============================================================

const MarketData& PanelData::asset(std::size_t i) const {
    return assets_[i];
}

const MarketData& PanelData::assetById(const std::string& id) const {
    return assets_[assetIndex(id)];
}

// ============================================================
// Cross-sectional access
// ============================================================

PanelData::CrossSection PanelData::crossSection(
        std::size_t dateIdx,
        Field field,
        BufferPool& pool) const {

    std::size_t nAssets = assets_.size();

    // Allocate contiguous buffer from the pool.
    auto handle = pool.allocate<double>(nAssets);

    // Gather: read the value at dateIdx from each asset's field column.
    for (std::size_t a = 0; a < nAssets; ++a) {
        const auto& col = assets_[a].column<double>(field);
        handle[a] = col[dateIdx];
    }

    // Build a ColView over the materialized buffer.
    // null-mask propagation is deferred to a future iteration;
    // individual null values can be checked via the source MarketData
    // if needed.
    ColView<double> view(handle.data(), nAssets, nullptr);

    return {std::move(handle), view};
}

// ============================================================
// Validation
// ============================================================

bool PanelData::isValid() const {
    if (assets_.empty()) return false;

    for (std::size_t a = 0; a < assets_.size(); ++a) {
        const auto& md = assets_[a];

        // Check timestamp alignment.
        if (!timestampsEqual(md.timestamps(), timestamps_)) {
            return false;
        }

        // Check that all 7 fields have the expected row count.
        if (md.rowCount() != dateCount()) {
            return false;
        }
    }
    return true;
}

std::size_t PanelData::nullCount(std::size_t dateIdx, Field field) const {
    std::size_t count = 0;
    for (const auto& md : assets_) {
        const auto& col = md.column<double>(field);
        if (col.isNull(dateIdx)) {
            ++count;
        }
    }
    return count;
}

// ============================================================
// Internal helpers
// ============================================================

void PanelData::validateAsset(const MarketData& asset) const {
    if (!timestampsEqual(asset.timestamps(), timestamps_)) {
        std::ostringstream oss;
        oss << "PanelData: timestamp mismatch for asset '"
            << asset.assetId() << "': expected "
            << timestamps_.size() << " rows";
        throw std::invalid_argument(oss.str());
    }
}

bool PanelData::timestampsEqual(const TimestampIndex& a,
                                const TimestampIndex& b) {
    if (a.size() != b.size()) return false;

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

}  // namespace quantcore
