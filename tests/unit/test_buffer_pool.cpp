// test_buffer_pool.cpp — unit tests for BufferHandle and BufferPool
// Phase: 二期必实现

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "quantcore/engine/BufferHandle.h"
#include "quantcore/engine/BufferPool.h"

using namespace quantcore;

// ============================================================
// BufferHandle tests
// ============================================================

TEST(BufferHandleTest, DefaultConstruction) {
    BufferHandle<double> h;
    EXPECT_EQ(h.data(), nullptr);
    EXPECT_EQ(h.size(), 0);
    EXPECT_TRUE(h.empty());
}

TEST(BufferHandleTest, WrapExternalMemory) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto handle = BufferHandle<double>::wrap(data.data(), data.size());

    EXPECT_EQ(handle.data(), data.data());
    EXPECT_EQ(handle.size(), 5);
    EXPECT_FALSE(handle.empty());

    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(handle[i], data[i]);
    }

    // Release on wrapped memory should be a no-op (pool_==nullptr)
    handle.release();
    EXPECT_EQ(handle.data(), nullptr);
    EXPECT_EQ(handle.size(), 0);
}

TEST(BufferHandleTest, MoveConstructor) {
    std::vector<double> data = {1.0, 2.0};
    auto h1 = BufferHandle<double>::wrap(data.data(), data.size());

    BufferHandle<double> h2(std::move(h1));

    EXPECT_EQ(h2.data(), data.data());
    EXPECT_EQ(h2.size(), 2);

    // h1 should be empty after move
    EXPECT_EQ(h1.data(), nullptr);
    EXPECT_EQ(h1.size(), 0);
}

TEST(BufferHandleTest, MoveAssignment) {
    std::vector<double> data1 = {1.0};
    std::vector<double> data2 = {2.0, 3.0};

    auto h1 = BufferHandle<double>::wrap(data1.data(), data1.size());
    auto h2 = BufferHandle<double>::wrap(data2.data(), data2.size());

    h1 = std::move(h2);

    EXPECT_EQ(h1.data(), data2.data());
    EXPECT_EQ(h1.size(), 2);
}

TEST(BufferHandleTest, Detach) {
    std::vector<double> data = {7.0, 8.0, 9.0};
    auto handle = BufferHandle<double>::wrap(data.data(), data.size());

    auto [ptr, sz] = handle.detach();

    EXPECT_EQ(ptr, data.data());
    EXPECT_EQ(sz, 3);
    EXPECT_EQ(handle.data(), nullptr);
    EXPECT_EQ(handle.size(), 0);
}

TEST(BufferHandleTest, RangeBasedFor) {
    std::vector<double> data = {10.0, 20.0, 30.0};
    auto handle = BufferHandle<double>::wrap(data.data(), data.size());

    double sum = 0.0;
    for (double v : handle) {
        sum += v;
    }
    EXPECT_DOUBLE_EQ(sum, 60.0);
}

// ============================================================
// BufferPool tests
// ============================================================

TEST(BufferPoolTest, AllocateAndDeallocate) {
    BufferPool pool;

    auto h1 = pool.allocate<double>(100);
    EXPECT_NE(h1.data(), nullptr);
    EXPECT_EQ(h1.size(), 100);

    // Write to the buffer to verify it's usable
    for (std::size_t i = 0; i < 100; ++i) {
        h1[i] = static_cast<double>(i);
    }
    for (std::size_t i = 0; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(h1[i], static_cast<double>(i));
    }

    // Release returns memory to pool
    h1.release();
    EXPECT_EQ(h1.data(), nullptr);
}

TEST(BufferPoolTest, Alignment) {
    BufferPool pool;

    for (int i = 0; i < 10; ++i) {
        auto h = pool.allocate<double>(50 + i * 10);
        EXPECT_NE(h.data(), nullptr);
        // Verify 64-byte alignment
        EXPECT_EQ(reinterpret_cast<uintptr_t>(h.data()) % 64, 0);
    }
}

TEST(BufferPoolTest, ReuseAfterFree) {
    BufferPool pool;

    // Allocate and free several buffers of the same size, then re-allocate.
    // Each handle releases on scope exit, returning memory to the free list.
    for (int i = 0; i < 5; ++i) {
        auto h = pool.allocate<double>(32);  // 256 bytes → Tiny class
        EXPECT_NE(h.data(), nullptr);
    }

    // Next allocation should succeed (either from free list or new slab)
    auto h = pool.allocate<double>(32);
    EXPECT_NE(h.data(), nullptr);
}

TEST(BufferPoolTest, DifferentSizeClasses) {
    BufferPool pool;

    // Tiny (256 B): 32 doubles
    auto hTiny = pool.allocate<double>(32);
    EXPECT_NE(hTiny.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(hTiny.data()) % 64, 0);

    // Small (4 KB): 512 doubles
    auto hSmall = pool.allocate<double>(512);
    EXPECT_NE(hSmall.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(hSmall.data()) % 64, 0);

    // Medium (16 KB): 2048 doubles
    auto hMed = pool.allocate<double>(2048);
    EXPECT_NE(hMed.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(hMed.data()) % 64, 0);

    // Large (64 KB): 8192 doubles
    auto hLarge = pool.allocate<double>(8192);
    EXPECT_NE(hLarge.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(hLarge.data()) % 64, 0);
}

TEST(BufferPoolTest, OverflowAllocation) {
    BufferPool pool;

    // >256 KB → direct aligned_alloc, not pooled
    auto h = pool.allocate<double>(40000);  // 320 KB
    EXPECT_NE(h.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(h.data()) % 64, 0);

    // Should still be writable
    for (std::size_t i = 0; i < 100; ++i) {
        h[i] = static_cast<double>(i);
    }
}

TEST(BufferPoolTest, ZeroSizeAllocation) {
    BufferPool pool;

    auto h = pool.allocate<double>(0);
    EXPECT_EQ(h.data(), nullptr);
    EXPECT_EQ(h.size(), 0);
}

TEST(BufferPoolTest, Statistics) {
    BufferPool pool;

    EXPECT_EQ(pool.totalAllocated(), 0);
    EXPECT_EQ(pool.totalDeallocated(), 0);

    {
        auto h = pool.allocate<double>(1000);
        EXPECT_GT(pool.totalAllocated(), 0);
    }
    // After release, totalDeallocated should be > 0
    EXPECT_GT(pool.totalDeallocated(), 0);
}

TEST(BufferPoolTest, SlabCount) {
    BufferPool pool;

    EXPECT_EQ(pool.slabCount(), 0);

    // Allocate enough buffers to force slab creation
    // Tiny: 256 bytes/block, 64KB slab = 256 blocks
    {
        std::vector<BufferHandle<double>> handles;
        for (int i = 0; i < 10; ++i) {
            handles.push_back(pool.allocate<double>(32));
        }
        EXPECT_GT(pool.slabCount(), 0);
    }
}

TEST(BufferPoolTest, Uint64Allocation) {
    BufferPool pool;

    // Allocate null mask buffer
    auto h = pool.allocate<uint64_t>(4);
    EXPECT_NE(h.data(), nullptr);
    EXPECT_EQ(h.size(), 4);

    h[0] = 0xDEADBEEF;
    h[1] = 0xCAFEBABE;
    EXPECT_EQ(h[0], 0xDEADBEEF);
    EXPECT_EQ(h[1], 0xCAFEBABE);
}
