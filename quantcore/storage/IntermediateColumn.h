// IntermediateColumn.h — materialized per-asset columnar storage for intermediate
//                              expression results in cross-sectional computation
// Phase: 五期实现
//
// IntermediateColumn stores the result of evaluating a per-asset (time-series)
// expression for ALL assets in a panel, providing dual-direction access:
//
//   Time-series (→):  per-asset contiguous double* pointers
//                      → optimal for expression evaluation (SIMD, fusion)
//   Cross-sectional (↓): gather all assets' values at a single date into a
//                         contiguous output buffer
//                      → required input for CS operators (rank, zscore, ...)
//
// The storage layout is per-asset (structure-of-arrays):
//
//   asset₀: [v₀, v₁, v₂, ..., v_{T-1}]  ← contiguous in time
//   asset₁: [v₀, v₁, v₂, ..., v_{T-1}]
//   ...
//   asset_{M-1}: [v₀, v₁, v₂, ..., v_{T-1}]
//
// Each asset's buffer is allocated from the BufferPool (64-byte aligned).
// Cross-sectional gathering at date d reads data_[a][d] for a = 0..M-1,
// which is a strided access pattern — acceptable for typical panel sizes
// but a known trade-off vs tile-based storage.
//
// Usage (with FactorCalculator):
//   IntermediateColumn intermediate(M, T, pool);
//   for (std::size_t a = 0; a < M; ++a) {
//       double* out = intermediate.timeSeriesData(a);
//       const uint64_t* nullMask =
//           innerExpr.evaluate(panel.asset(a), out, T, &pool);
//       intermediate.setTimeSeriesNullMask(a, nullMask);
//   }
//   // Then per-date:
//   for (std::size_t d = 0; d < T; ++d) {
//       intermediate.gatherCrossSection(d, workspaceBuf, M);
//       // ... apply CS operator, scatter results ...
//   }
//
// All memory is owned via BufferHandle — move-only, pool-backed, auto-released
// on destruction.  Null values are propagated as NaN during gatherCrossSection,
// matching the CsOperator NaN-as-null convention.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/engine/BufferHandle.h"
#include "quantcore/engine/BufferPool.h"
#include "quantcore/storage/ColView.h"

namespace quantcore {

class IntermediateColumn {
public:
    // ============================================================
    // Construction
    // ============================================================

    /// Empty column — no assets, no data.
    IntermediateColumn() = default;

    /// Construct with given dimensions.  Allocates all per-asset buffers
    /// from the pool immediately (eager allocation), so subsequent
    /// time-series writes and cross-sectional reads have zero allocation
    /// overhead.
    ///
    /// @param nAssets      Number of assets in the panel.
    /// @param nTimePoints  Number of time points (dates) per asset.
    /// @param pool         BufferPool for 64-byte aligned allocations.
    ///                     Must outlive this IntermediateColumn.
    IntermediateColumn(std::size_t nAssets,
                       std::size_t nTimePoints,
                       BufferPool& pool);

    /// Not copyable — owns large pool-allocated buffers.
    IntermediateColumn(const IntermediateColumn&) = delete;
    IntermediateColumn& operator=(const IntermediateColumn&) = delete;

    /// Movable — transfers buffer ownership.
    /// Explicit to zero primitive members in the source.
    IntermediateColumn(IntermediateColumn&& other) noexcept
        : data_(std::move(other.data_))
        , nullMasks_(std::move(other.nullMasks_))
        , assetCount_(other.assetCount_)
        , timePointCount_(other.timePointCount_)
    {
        other.assetCount_     = 0;
        other.timePointCount_ = 0;
    }

    IntermediateColumn& operator=(IntermediateColumn&& other) noexcept {
        if (this != &other) {
            data_           = std::move(other.data_);
            nullMasks_      = std::move(other.nullMasks_);
            assetCount_     = other.assetCount_;
            timePointCount_ = other.timePointCount_;
            other.assetCount_     = 0;
            other.timePointCount_ = 0;
        }
        return *this;
    }

    // ============================================================
    // Dimensions
    // ============================================================

    std::size_t assetCount()     const noexcept { return assetCount_; }
    std::size_t timePointCount() const noexcept { return timePointCount_; }
    bool empty()                 const noexcept { return assetCount_ == 0; }

    // ============================================================
    // Time-series access (per-asset, O(1), zero-copy)
    //
    // Returns a mutable pointer to the contiguous time-series buffer
    // for a single asset.  The buffer length is timePointCount().
    // Use this as the output target for per-asset expression evaluation.
    // ============================================================

    /// Mutable access: write computed expression results here.
    /// @pre assetIdx < assetCount()
    double* timeSeriesData(std::size_t assetIdx) {
        return data_[assetIdx].data();
    }

    /// Read-only access.
    const double* timeSeriesData(std::size_t assetIdx) const {
        return data_[assetIdx].data();
    }

    // ============================================================
    // Null mask management
    // ============================================================

    /// Get the null mask for an asset's time-series (non-owning pointer).
    /// Returns nullptr if no null mask has been set.
    const uint64_t* timeSeriesNullMask(std::size_t assetIdx) const {
        return nullMasks_[assetIdx];
    }

    /// Set the null mask for an asset.  The pointer is stored as-is
    /// (non-owning) and must remain valid for the lifetime of this
    /// IntermediateColumn or until overwritten.
    void setTimeSeriesNullMask(std::size_t assetIdx,
                                const uint64_t* mask) {
        nullMasks_[assetIdx] = mask;
    }

    /// Check whether a specific (asset, date) position is marked null.
    bool isNull(std::size_t assetIdx, std::size_t dateIdx) const {
        const uint64_t* mask = nullMasks_[assetIdx];
        if (!mask) return false;
        return (mask[dateIdx / 64] >> (dateIdx % 64)) & uint64_t{1};
    }

    // ============================================================
    // Cross-sectional access
    //
    // Gathers values from all assets at a single time point into a
    // caller-provided contiguous output buffer.  Null positions are
    // written as NaN (matching the CsOperator NaN-as-null convention).
    //
    // Returns the count of valid (non-null, non-NaN) values.
    // ============================================================

    /// Gather the cross-sectional vector at dateIdx.
    ///
    /// @param dateIdx    Time index to gather at (0 <= dateIdx < timePointCount).
    /// @param output     Destination buffer, must have space for at least
    ///                   min(assetCount, outputSize) doubles.
    /// @param outputSize Capacity of the output buffer.
    /// @return           Number of valid (non-null, non-NaN) values written.
    ///                   Null positions are written as quiet_NaN and NOT
    ///                   counted in the return value.
    std::size_t gatherCrossSection(std::size_t dateIdx,
                                   double* output,
                                   std::size_t outputSize) const {
        std::size_t n = std::min(assetCount_, outputSize);
        std::size_t validCount = 0;

        for (std::size_t a = 0; a < n; ++a) {
            const double* src = data_[a].data();
            if (isNull(a, dateIdx)) {
                output[a] = std::numeric_limits<double>::quiet_NaN();
            } else {
                double v = src[dateIdx];
                output[a] = v;
                if (!std::isnan(v)) {
                    ++validCount;
                }
            }
        }

        return validCount;
    }

    /// Gather and wrap in a ColView for direct use with CS operators.
    /// The output buffer must outlive the returned ColView.
    ColView<double> gatherCrossSectionView(std::size_t dateIdx,
                                           double* output,
                                           std::size_t outputSize) const {
        gatherCrossSection(dateIdx, output, outputSize);
        return ColView<double>(output,
                               std::min(assetCount_, outputSize));
    }

    // ============================================================
    // Computation status
    // ============================================================

    /// Whether a specific asset's data has been populated.
    /// Since buffers are eagerly allocated, this always returns true
    /// for valid indices in a non-empty column.  Provided for
    /// compatibility with lazy-allocation patterns in the future.
    bool isComputed(std::size_t assetIdx) const noexcept {
        return assetIdx < data_.size() && data_[assetIdx].data() != nullptr;
    }

    /// Whether all assets have been populated (always true for non-empty
    /// columns with eager allocation).
    bool isFullyComputed() const noexcept {
        for (std::size_t i = 0; i < assetCount_; ++i) {
            if (!isComputed(i)) return false;
        }
        return true;
    }

    // ============================================================
    // Bulk data access (for advanced use)
    // ============================================================

    /// Number of bytes allocated for this column.
    std::size_t memoryBytes() const noexcept {
        return assetCount_ * timePointCount_ * sizeof(double);
    }

private:
    std::vector<BufferHandle<double>> data_;        // per-asset, each length T
    std::vector<const uint64_t*>      nullMasks_;   // per-asset, non-owning
    std::size_t assetCount_     = 0;
    std::size_t timePointCount_ = 0;
};

// ============================================================
// Implementation (constructor)
// ============================================================

inline IntermediateColumn::IntermediateColumn(
        std::size_t nAssets,
        std::size_t nTimePoints,
        BufferPool& pool)
    : assetCount_(nAssets)
    , timePointCount_(nTimePoints)
{
    data_.reserve(nAssets);
    nullMasks_.resize(nAssets, nullptr);

    for (std::size_t i = 0; i < nAssets; ++i) {
        data_.push_back(pool.allocate<double>(nTimePoints));
    }
}

}  // namespace quantcore
