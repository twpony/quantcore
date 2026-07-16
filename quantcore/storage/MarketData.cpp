// MarketData.cpp — MarketData and MarketDataView implementation
// Phase: 一期必实现
#include "MarketData.h"

#include <stdexcept>

#include "quantcore/core/ErrorHandling.h"
#include "quantcore/core/Logger.h"

namespace quantcore {

// ============================================================
// MarketData
// ============================================================

MarketData::MarketData(std::string assetId, TimestampIndex timestamps)
    : assetId_(std::move(assetId))
    , timestamps_(std::move(timestamps))
{
    // Initialize all field slots with empty double columns by default.
    // All fields are double; columns start empty until explicitly set.
}

bool MarketData::allColumnsAligned() const {
    std::size_t expected = rowCount();
    for (std::size_t i = 0; i < static_cast<std::size_t>(Field::kFieldCount); ++i) {
        const auto& var = columns_[i];
        bool ok = std::visit([expected](const auto& col) -> bool {
            using ColType = std::decay_t<decltype(col)>;
            if constexpr (std::is_same_v<ColType, Column<double>> ||
                          std::is_same_v<ColType, Column<int64_t>>) {
                return col.empty() || col.size() == expected;
            }
            return true;
        }, var);
        if (!ok) return false;
    }
    return true;
}

void MarketData::updateNullFlag() {
    hasNullAnywhere_ = false;
    for (std::size_t i = 0; i < static_cast<std::size_t>(Field::kFieldCount); ++i) {
        bool fieldHasNull = std::visit([](const auto& col) -> bool {
            return col.hasNullMask();
        }, columns_[i]);
        if (fieldHasNull) {
            hasNullAnywhere_ = true;
            return;
        }
    }
}

MarketDataView MarketData::slice(std::size_t start, std::size_t end) const {
    if (start > end) {
        QC_WARN("MarketData::slice: start > end, swapping");
        std::swap(start, end);
    }
    if (end > rowCount()) {
        QUANTCORE_THROW(ConfigError,
            "MarketData::slice: end (" + std::to_string(end) +
            ") exceeds rowCount (" + std::to_string(rowCount()) + ")");
    }
    return MarketDataView::fromMarketData(*this, start, end);
}

MarketDataView MarketData::sliceByDate(int64_t startDate,
                                         int64_t endDate) const {
    auto [start, end] = timestamps_.dateRange(startDate, endDate);
    if (start >= end) {
        QC_WARN("MarketData::sliceByDate: no rows in [" +
                std::to_string(startDate) + ", " +
                std::to_string(endDate) + "]");
        return MarketDataView{};  // empty view
    }
    return MarketDataView::fromMarketData(*this, start, end);
}

void MarketData::allocateAllFields() {
    std::size_t rows = rowCount();

    // Price fields: double
    columns_[static_cast<std::size_t>(Field::OPEN)]   = Column<double>(rows);
    columns_[static_cast<std::size_t>(Field::HIGH)]   = Column<double>(rows);
    columns_[static_cast<std::size_t>(Field::LOW)]    = Column<double>(rows);
    columns_[static_cast<std::size_t>(Field::CLOSE)]  = Column<double>(rows);
    columns_[static_cast<std::size_t>(Field::VWAP)]   = Column<double>(rows);

    // Quantity fields: double
    columns_[static_cast<std::size_t>(Field::VOLUME)] = Column<double>(rows);
    columns_[static_cast<std::size_t>(Field::AMOUNT)] = Column<double>(rows);
}

// ============================================================
// MarketDataView
// ============================================================

MarketDataView MarketDataView::fromMarketData(const MarketData& md,
                                               std::size_t start,
                                               std::size_t end) {
    MarketDataView view;
    view.assetId_  = md.assetId();
    view.rowCount_ = (end > start) ? (end - start) : 0;

    // Build column views for each field
    for (std::size_t i = 0; i < static_cast<std::size_t>(Field::kFieldCount); ++i) {
        Field field = static_cast<Field>(i);
        const auto& srcVar = md.columnVariant(field);

        ColViewVariant viewVar = std::visit(
            [start, end](const auto& srcCol) -> ColViewVariant {
                using ColType = std::decay_t<decltype(srcCol)>;
                if (srcCol.empty()) {
                    // Empty column → empty view
                    if constexpr (std::is_same_v<ColType, Column<double>>) {
                        return ColView<double>{};
                    } else {
                        return ColView<int64_t>{};
                    }
                }
                if constexpr (std::is_same_v<ColType, Column<double>>) {
                    return ColView<double>(srcCol, start, end);
                } else if constexpr (std::is_same_v<ColType, Column<int64_t>>) {
                    return ColView<int64_t>(srcCol, start, end);
                }
                // unreachable
                return ColView<double>{};
            },
            srcVar);

        view.colViews_[i] = std::move(viewVar);
    }

    // Build timestamp index view
    const auto& tsCol  = md.timestamps().timestamps();
    const auto& dCol   = md.timestamps().dates();
    view.timestampView_ = TimestampIndexView(
        tsCol.data(), dCol.data(), view.rowCount_, start);

    return view;
}

}  // namespace quantcore
