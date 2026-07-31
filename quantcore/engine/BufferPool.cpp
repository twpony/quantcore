// BufferPool.cpp — slab-based allocator implementation
// Phase: 二期必实现
//
// See BufferPool.h for the public interface and design rationale.

#include "quantcore/engine/BufferPool.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "quantcore/core/ErrorHandling.h"

namespace quantcore {

// ============================================================
// Size classification
// ============================================================

BufferPool::SizeClass BufferPool::classify(std::size_t bytes) noexcept {
    if (bytes <= 256)       return SizeClass::kTiny;
    if (bytes <= 4 * 1024)  return SizeClass::kSmall;
    if (bytes <= 16 * 1024) return SizeClass::kMedium;
    if (bytes <= 64 * 1024) return SizeClass::kLarge;
    return SizeClass::kHuge;  // up to 256 KB; overflow handled separately
}

std::size_t BufferPool::blockSize(SizeClass sc) noexcept {
    switch (sc) {
        case SizeClass::kTiny:   return 256;
        case SizeClass::kSmall:  return 4 * 1024;
        case SizeClass::kMedium: return 16 * 1024;
        case SizeClass::kLarge:  return 64 * 1024;
        case SizeClass::kHuge:   return 256 * 1024;
        default:                 return 0;
    }
}

// ============================================================
// Slab allocation
// ============================================================

void BufferPool::allocateSlab(SizeClass sc) {
    std::size_t bsize = blockSize(sc);
    std::size_t nBlocks = kSlabSize / bsize;

    // Allocate 64-byte aligned slab
    void* mem = std::aligned_alloc(kAlignment, kSlabSize);
    if (!mem) {
        throw std::bad_alloc();
    }

    Slab slab;
    slab.memory    = mem;
    slab.blockSize = bsize;
    slab.numBlocks = nBlocks;
    slab.bitmap.resize((nBlocks + 7) / 8, 0xFF);  // all free
    slab.freeCount = nBlocks;

    slabs_.push_back(slab);

    // Push all block pointers onto the free list
    auto& freeList = freeLists_[static_cast<int>(sc)];
    freeList.reserve(freeList.size() + nBlocks);
    uint8_t* base = static_cast<uint8_t*>(mem);
    for (std::size_t i = 0; i < nBlocks; ++i) {
        freeList.push_back(base + i * bsize);
    }

    totalAllocated_ += kSlabSize;
}

// ============================================================
// Raw allocation
// ============================================================

void* BufferPool::allocateRaw(std::size_t bytes) {
    if (bytes == 0) return nullptr;

    // Overflow path: allocate directly, do not pool
    if (bytes > kOverflowThreshold) {
        void* ptr = std::aligned_alloc(kAlignment, bytes);
        if (!ptr) throw std::bad_alloc();
        totalAllocated_ += bytes;
        return ptr;
    }

    SizeClass sc = classify(bytes);
    auto& freeList = freeLists_[static_cast<int>(sc)];

    // If the free list is empty, allocate a new slab
    if (freeList.empty()) {
        allocateSlab(sc);
    }

    // Pop a block from the free list
    void* ptr = freeList.back();
    freeList.pop_back();

    // Mark the block as used in the bitmap
    // Find which slab this block belongs to
    for (auto& slab : slabs_) {
        uint8_t* base = static_cast<uint8_t*>(slab.memory);
        uint8_t* end  = base + kSlabSize;
        uint8_t* blk  = static_cast<uint8_t*>(ptr);
        if (blk >= base && blk < end) {
            std::size_t idx = static_cast<std::size_t>(blk - base) / slab.blockSize;
            std::size_t word = idx / 8;
            std::size_t bit  = idx % 8;
            slab.bitmap[word] &= ~(uint8_t{1} << bit);
            --slab.freeCount;
            break;
        }
    }

    return ptr;
}

// ============================================================
// Raw deallocation
// ============================================================

void BufferPool::deallocateRaw(void* ptr, std::size_t bytes) noexcept {
    if (!ptr) return;

    // Overflow path: free directly
    if (bytes > kOverflowThreshold) {
        std::free(ptr);
        totalDeallocated_ += bytes;
        return;
    }

    SizeClass sc = classify(bytes);

    // Find which slab this block belongs to and mark it free
    for (auto& slab : slabs_) {
        uint8_t* base = static_cast<uint8_t*>(slab.memory);
        uint8_t* end  = base + kSlabSize;
        uint8_t* blk  = static_cast<uint8_t*>(ptr);
        if (blk >= base && blk < end) {
            std::size_t idx = static_cast<std::size_t>(blk - base) / slab.blockSize;
            std::size_t word = idx / 8;
            std::size_t bit  = idx % 8;
            slab.bitmap[word] |= (uint8_t{1} << bit);
            ++slab.freeCount;

            // Push back to free list
            freeLists_[static_cast<int>(sc)].push_back(ptr);
            totalDeallocated_ += slab.blockSize;
            return;
        }
    }

    // If we reach here, the pointer didn't belong to any slab.
    // This is a logic error — the pointer was not allocated by this pool.
    // In release builds, silently ignore to avoid crashing.
}

// ============================================================
// Destruction
// ============================================================

BufferPool::~BufferPool() {
    // Free all slabs.  We do NOT check for leaks — any outstanding
    // BufferHandles should have been released before the pool is
    // destroyed.  Outstanding handles after pool destruction will
    // have dangling pool_ pointers; their release() becomes a no-op
    // (detectable in debug builds by checking if this pool is alive).
    for (auto& slab : slabs_) {
        std::free(slab.memory);
    }
}

}  // namespace quantcore
