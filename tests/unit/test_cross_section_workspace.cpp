// test_cross_section_workspace.cpp — unit tests for CrossSectionWorkspace
// Phase: 五期实现

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>

#include "quantcore/engine/BufferPool.h"
#include "quantcore/engine/CrossSectionWorkspace.h"

using namespace quantcore;

class CrossSectionWorkspaceTest : public ::testing::Test {
protected:
    BufferPool pool_;
};

// ============================================================
// Basic allocation
// ============================================================

TEST_F(CrossSectionWorkspaceTest, AllocateZeroAssets) {
    auto ws = CrossSectionWorkspace::allocate(pool_, 0);
    EXPECT_EQ(ws.capacity, 0u);
    EXPECT_EQ(ws.values.size(), 0u);
    EXPECT_EQ(ws.indices.size(), 0u);
}

TEST_F(CrossSectionWorkspaceTest, AllocateSmall) {
    constexpr std::size_t M = 10;
    auto ws = CrossSectionWorkspace::allocate(pool_, M);

    EXPECT_EQ(ws.capacity, M);
    EXPECT_EQ(ws.values.size(), M);
    EXPECT_EQ(ws.indices.size(), M);
    EXPECT_NE(ws.values.data(), nullptr);
    EXPECT_NE(ws.indices.data(), nullptr);

    // Buffers should be writable
    for (std::size_t i = 0; i < M; ++i) {
        ws.values[i] = static_cast<double>(i) * 1.5;
        ws.indices[i] = i;
    }
    for (std::size_t i = 0; i < M; ++i) {
        EXPECT_DOUBLE_EQ(ws.values[i], static_cast<double>(i) * 1.5);
        EXPECT_EQ(ws.indices[i], i);
    }
}

TEST_F(CrossSectionWorkspaceTest, AllocateLarge) {
    // Typical real-world: 5000 assets
    constexpr std::size_t M = 5000;
    auto ws = CrossSectionWorkspace::allocate(pool_, M);

    EXPECT_EQ(ws.capacity, M);
    EXPECT_EQ(ws.values.size(), M);
    EXPECT_EQ(ws.indices.size(), M);
}

// ============================================================
// ensureCapacity — reuse vs reallocation
// ============================================================

TEST_F(CrossSectionWorkspaceTest, EnsureCapacityNoReallocWhenSufficient) {
    auto ws = CrossSectionWorkspace::allocate(pool_, 100);
    double* origValues = ws.values.data();
    std::size_t* origIndices = ws.indices.data();
    std::size_t origCap = ws.capacity;

    // Same capacity: should be no-op
    ws.ensureCapacity(pool_, 100);
    EXPECT_EQ(ws.values.data(), origValues);
    EXPECT_EQ(ws.indices.data(), origIndices);
    EXPECT_EQ(ws.capacity, origCap);

    // Smaller capacity: should also be no-op
    ws.ensureCapacity(pool_, 50);
    EXPECT_EQ(ws.values.data(), origValues);
    EXPECT_EQ(ws.indices.data(), origIndices);
    EXPECT_EQ(ws.capacity, origCap);
}

TEST_F(CrossSectionWorkspaceTest, EnsureCapacityReallocWhenInsufficient) {
    auto ws = CrossSectionWorkspace::allocate(pool_, 100);
    double* origValues = ws.values.data();

    // Larger capacity: should reallocate
    ws.ensureCapacity(pool_, 200);
    EXPECT_EQ(ws.capacity, 200u);
    // New buffer should be different after reallocation
    // (note: pool may reuse freed blocks, so this is not a strict guarantee,
    //  but with LIFO free list it's likely the same block is reused)
    EXPECT_EQ(ws.values.size(), 200u);
    EXPECT_EQ(ws.indices.size(), 200u);
    EXPECT_NE(ws.values.data(), nullptr);
}

// ============================================================
// Memory bytes
// ============================================================

TEST_F(CrossSectionWorkspaceTest, MemoryBytes) {
    auto ws = CrossSectionWorkspace::allocate(pool_, 1000);
    std::size_t expected = 1000 * (sizeof(double) + sizeof(std::size_t));
    EXPECT_EQ(ws.memoryBytes(), expected);
}

TEST_F(CrossSectionWorkspaceTest, MemoryBytesZero) {
    auto ws = CrossSectionWorkspace::allocate(pool_, 0);
    EXPECT_EQ(ws.memoryBytes(), 0u);
}

// ============================================================
// Move semantics
// ============================================================

TEST_F(CrossSectionWorkspaceTest, MoveConstruct) {
    auto ws1 = CrossSectionWorkspace::allocate(pool_, 50);
    double* origValues = ws1.values.data();
    std::size_t* origIndices = ws1.indices.data();

    auto ws2 = std::move(ws1);

    EXPECT_EQ(ws2.values.data(), origValues);
    EXPECT_EQ(ws2.indices.data(), origIndices);
    EXPECT_EQ(ws2.capacity, 50u);

    // ws1 should be empty after move
    EXPECT_EQ(ws1.capacity, 0u);
    EXPECT_EQ(ws1.values.data(), nullptr);
    EXPECT_EQ(ws1.indices.data(), nullptr);
}
