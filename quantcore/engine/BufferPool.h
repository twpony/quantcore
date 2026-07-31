// BufferPool.h — slab-based allocator for aligned temporary buffers
// Phase: 二期必实现
//
// Allocates 64-byte aligned memory in size-class slabs.  Memory is NOT
// zeroed.  Deallocated blocks are reused via per-size-class free lists.
//
// Each slab is 64 KB (one cache-line-aligned page).  Blocks within a
// slab are tracked by a bitmap (1 = free, 0 = allocated).
//
// Overflow (>256 KB) uses raw aligned_alloc/free and is not pooled.
//
// Thread safety: BufferPool is NOT thread-safe.  Create one per thread.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "quantcore/engine/BufferHandle.h"

namespace quantcore {

class BufferPool {
public:
    // ============================================================
    // Size classes
    // ============================================================

    enum class SizeClass : uint8_t {
        kTiny   = 0,   // 256 B    (32 doubles)
        kSmall  = 1,   // 4 KB     (512 doubles)
        kMedium = 2,   // 16 KB    (2048 doubles)
        kLarge  = 3,   // 64 KB    (8192 doubles)
        kHuge   = 4,   // 256 KB   (32768 doubles)
        kCount  = 5,
    };

    static constexpr std::size_t kSlabSize          = 64 * 1024;   // 64 KB
    static constexpr std::size_t kOverflowThreshold = 256 * 1024;  // 256 KB
    static constexpr std::size_t kAlignment         = 64;

    // ============================================================
    // Construction
    // ============================================================

    BufferPool() = default;
    ~BufferPool();

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    BufferPool(BufferPool&&) = delete;
    BufferPool& operator=(BufferPool&&) = delete;

    // ============================================================
    // Public typed interface
    // ============================================================

    /// Allocate `n` aligned elements of type T.
    template <typename T>
    BufferHandle<T> allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        void* ptr = allocateRaw(bytes);
        return BufferHandle<T>(static_cast<T*>(ptr), n, this);
    }

    /// Deallocate a buffer returned by allocate<T>().  n must match the
    /// original allocation size exactly.
    template <typename T>
    void deallocate(T* ptr, std::size_t n) noexcept {
        deallocateRaw(static_cast<void*>(ptr), n * sizeof(T));
    }

    // ============================================================
    // Statistics
    // ============================================================

    std::size_t totalAllocated()   const noexcept { return totalAllocated_;   }
    std::size_t totalDeallocated() const noexcept { return totalDeallocated_; }
    std::size_t slabCount()        const noexcept { return slabs_.size();     }
    std::size_t freeListSize(SizeClass sc) const noexcept {
        auto idx = static_cast<int>(sc);
        return (idx >= 0 && idx < static_cast<int>(SizeClass::kCount))
            ? freeLists_[idx].size() : 0;
    }

    /// Max number of blocks that can be served from currently free blocks
    /// (no new slab allocation).
    std::size_t availableBlocks(SizeClass sc) const noexcept {
        return freeListSize(sc);
    }

private:
    friend class BufferHandle<double>;
    friend class BufferHandle<uint64_t>;

    // ============================================================
    // Size classification
    // ============================================================

    static SizeClass classify(std::size_t bytes) noexcept;
    static std::size_t blockSize(SizeClass sc) noexcept;

    // ============================================================
    // Raw (untyped) allocation / deallocation
    // ============================================================

    void* allocateRaw(std::size_t bytes);
    void  deallocateRaw(void* ptr, std::size_t bytes) noexcept;

    // ============================================================
    // Slab management
    // ============================================================

    /// Allocate a new slab for the given size class and push all its
    /// blocks onto the free list.
    void allocateSlab(SizeClass sc);

    struct Slab {
        void*                memory;     // 64-byte aligned chunk (kSlabSize)
        std::size_t          blockSize;  // per-block bytes
        std::size_t          numBlocks;
        std::vector<uint8_t> bitmap;     // 1 bit per block: 1 = free
        std::size_t          freeCount;
    };

    std::vector<Slab>  slabs_;
    std::vector<void*> freeLists_[static_cast<int>(SizeClass::kCount)];

    std::size_t totalAllocated_   = 0;
    std::size_t totalDeallocated_ = 0;
};

// ============================================================
// BufferHandle<T>::release() — defined here because it needs
// the full BufferPool definition (avoids circular include).
// ============================================================

template <typename T>
void BufferHandle<T>::release() noexcept {
    if (data_ && pool_) {
        pool_->deallocate(data_, size_);
    }
    data_ = nullptr;
    size_ = 0;
    pool_ = nullptr;
}

}  // namespace quantcore
