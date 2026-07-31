// CrossSectionWorkspace.h — reusable per-date workspace for cross-sectional operations
// Phase: 五期实现
//
// CrossSectionWorkspace provides pre-allocated, pool-backed buffers that can be
// reused across multiple date iterations in a cross-sectional computation loop.
// This avoids the overhead of allocate/deallocate per date.
//
// The workspace holds two buffers:
//   values  — M doubles, 64-byte aligned, for the gathered cross-sectional vector
//   indices — M size_t, for sorting-based operators (rank, quantile, winsorize)
//
// Usage:
//   auto ws = CrossSectionWorkspace::allocate(pool, assetCount);
//   for (std::size_t d = 0; d < nDates; ++d) {
//       intermediate.gatherCrossSection(d, ws.values.data(), assetCount);
//       ColView<double> input(ws.values.data(), assetCount);
//       reg.invokeCs(op, input, ws.values.data(), extraParams);
//       // scatter ws.values[a] to per-asset output at date d
//   }
//   // ws auto-releases buffers back to pool on scope exit
//
// Capacity: for 5000 assets, values=40KB, indices=40KB → 80KB total.
// This fits comfortably within L2 cache on modern CPUs.
#pragma once

#include <cstddef>
#include <cstdint>

#include "quantcore/engine/BufferHandle.h"
#include "quantcore/engine/BufferPool.h"

namespace quantcore {

struct CrossSectionWorkspace {
    BufferHandle<double> values;        // M doubles, 64-byte aligned
    BufferHandle<std::size_t> indices;  // M indices for sorting-based ops
    std::size_t capacity = 0;           // allocated element count

    // ============================================================
    // Move semantics — explicit to zero primitive members
    // ============================================================

    CrossSectionWorkspace() = default;

    CrossSectionWorkspace(CrossSectionWorkspace&& other) noexcept
        : values(std::move(other.values))
        , indices(std::move(other.indices))
        , capacity(other.capacity)
    {
        other.capacity = 0;
    }

    CrossSectionWorkspace& operator=(CrossSectionWorkspace&& other) noexcept {
        if (this != &other) {
            values  = std::move(other.values);
            indices = std::move(other.indices);
            capacity = other.capacity;
            other.capacity = 0;
        }
        return *this;
    }

    CrossSectionWorkspace(const CrossSectionWorkspace&) = delete;
    CrossSectionWorkspace& operator=(const CrossSectionWorkspace&) = delete;

    // ============================================================
    // Factory: allocate both buffers from the pool
    // ============================================================

    /// Allocate workspace for up to `nAssets` assets.
    /// Both buffers are 64-byte aligned (pool guarantee).
    static CrossSectionWorkspace allocate(BufferPool& pool,
                                           std::size_t nAssets) {
        CrossSectionWorkspace ws;
        ws.values  = pool.allocate<double>(nAssets);
        ws.indices = pool.allocate<std::size_t>(nAssets);
        ws.capacity = nAssets;
        return ws;
    }

    // ============================================================
    // Capacity management
    // ============================================================

    /// Ensure the workspace can hold at least `nAssets` elements.
    /// Reallocates only if current capacity is insufficient; otherwise
    /// a no-op (preserves existing data).
    void ensureCapacity(BufferPool& pool, std::size_t nAssets) {
        if (capacity >= nAssets) return;
        values  = pool.allocate<double>(nAssets);
        indices = pool.allocate<std::size_t>(nAssets);
        capacity = nAssets;
    }

    // ============================================================
    // Diagnostics
    // ============================================================

    /// Total allocated memory in bytes.
    std::size_t memoryBytes() const noexcept {
        return capacity * (sizeof(double) + sizeof(std::size_t));
    }
};

}  // namespace quantcore
