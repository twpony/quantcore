// test_column.cpp — unit tests for Column<T> and ColView<T>
// Phase: 一期必实现
//
// Coverage:
//   Column<T>: construction, copy, move, bounds, null mask, alignment,
//              empty column, large capacity, mmap wrapping, iteration,
//              resize, comparison operators
//   ColView<T>: zero-copy semantics, slicing, sub-view, null mask forwarding,
//               empty view, factory helpers

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "quantcore/storage/Column.h"
#include "quantcore/storage/ColView.h"

using namespace quantcore;

// ============================================================
// Test fixture: pre-built columns for sharing across tests
// ============================================================

class ColumnTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Build a small test column with known values
        smallDouble_ = Column<double>(5);
        for (std::size_t i = 0; i < 5; ++i) {
            smallDouble_[i] = static_cast<double>(i) * 1.5;
        }

        smallInt64_ = Column<int64_t>(4);
        for (std::size_t i = 0; i < 4; ++i) {
            smallInt64_[i] = static_cast<int64_t>(i) * 100;
        }
    }

    Column<double>  smallDouble_;
    Column<int64_t> smallInt64_;
};

// ============================================================
// Column<T> — Construction
// ============================================================

TEST_F(ColumnTest, DefaultConstructEmpty) {
    Column<double> col;
    EXPECT_EQ(col.size(), 0u);
    EXPECT_TRUE(col.empty());
    EXPECT_FALSE(col.hasNullMask());
}

TEST_F(ColumnTest, SizedConstruct) {
    Column<double> col(10);
    EXPECT_EQ(col.size(), 10u);
    EXPECT_FALSE(col.empty());
    EXPECT_FALSE(col.hasNullMask());
    // Default-initialized doubles should be 0.0
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(col[i], 0.0);
    }
}

TEST_F(ColumnTest, SizedWithValue) {
    Column<double> col(5, 3.14);
    EXPECT_EQ(col.size(), 5u);
    for (std::size_t i = 0; i < 5; ++i) {
        EXPECT_DOUBLE_EQ(col[i], 3.14);
    }
}

TEST_F(ColumnTest, FromRawPointer) {
    double raw[] = {1.0, 2.0, 3.0, 4.0};
    Column<double> col(raw, 4);
    EXPECT_EQ(col.size(), 4u);
    EXPECT_DOUBLE_EQ(col[0], 1.0);
    EXPECT_DOUBLE_EQ(col[3], 4.0);

    // Verify deep copy — modifying raw should not affect column
    raw[0] = 99.0;
    EXPECT_DOUBLE_EQ(col[0], 1.0);
}

TEST_F(ColumnTest, FromInitializerList) {
    Column<double> col = {1.0, 2.0, 3.0};
    EXPECT_EQ(col.size(), 3u);
    EXPECT_DOUBLE_EQ(col[0], 1.0);
    EXPECT_DOUBLE_EQ(col[1], 2.0);
    EXPECT_DOUBLE_EQ(col[2], 3.0);
}

TEST_F(ColumnTest, Int64Column) {
    Column<int64_t> col = {100, 200, 300};
    EXPECT_EQ(col.size(), 3u);
    EXPECT_EQ(col[0], 100);
    EXPECT_EQ(col[1], 200);
    EXPECT_EQ(col[2], 300);
}

// ============================================================
// Column<T> — Copy & Move
// ============================================================

TEST_F(ColumnTest, CopyConstruct) {
    Column<double> original = {1.0, 2.0, 3.0};
    Column<double> copy(original);

    EXPECT_EQ(copy.size(), original.size());
    EXPECT_DOUBLE_EQ(copy[0], 1.0);
    EXPECT_DOUBLE_EQ(copy[1], 2.0);

    // Modify original, copy should be independent
    original[0] = 99.0;
    EXPECT_DOUBLE_EQ(copy[0], 1.0);
}

TEST_F(ColumnTest, CopyAssign) {
    Column<double> a = {1.0, 2.0};
    Column<double> b = {3.0, 4.0, 5.0};
    b = a;
    EXPECT_EQ(b.size(), 2u);
    EXPECT_DOUBLE_EQ(b[0], 1.0);
}

TEST_F(ColumnTest, MoveConstruct) {
    Column<double> original = {1.0, 2.0, 3.0};
    const double* origData = original.data();
    Column<double> moved(std::move(original));

    EXPECT_EQ(moved.size(), 3u);
    EXPECT_EQ(moved.data(), origData);  // pointer should be reused
    EXPECT_DOUBLE_EQ(moved[0], 1.0);
}

TEST_F(ColumnTest, MoveAssign) {
    Column<double> a = {1.0, 2.0};
    Column<double> b;
    b = std::move(a);
    EXPECT_EQ(b.size(), 2u);
}

// ============================================================
// Column<T> — Null mask
// ============================================================

TEST_F(ColumnTest, NullMaskLazyAllocation) {
    Column<double> col(10);
    EXPECT_FALSE(col.hasNullMask());
    EXPECT_EQ(col.nullCount(), 0u);

    // First null → allocates mask
    col.setNull(3);
    EXPECT_TRUE(col.hasNullMask());
    EXPECT_EQ(col.nullCount(), 1u);
    EXPECT_TRUE(col.isNull(3));
    EXPECT_FALSE(col.isNull(0));
    EXPECT_FALSE(col.isNull(9));
}

TEST_F(ColumnTest, NullMaskMultiplePositions) {
    Column<double> col(100);
    col.setNull(0);
    col.setNull(63);   // boundary within first uint64
    col.setNull(64);   // first bit of second uint64
    col.setNull(99);   // last element

    EXPECT_TRUE(col.isNull(0));
    EXPECT_TRUE(col.isNull(63));
    EXPECT_TRUE(col.isNull(64));
    EXPECT_TRUE(col.isNull(99));
    EXPECT_EQ(col.nullCount(), 4u);
}

TEST_F(ColumnTest, ClearNull) {
    Column<double> col(10);
    col.setNull(5);
    EXPECT_TRUE(col.isNull(5));

    col.clearNull(5);
    EXPECT_FALSE(col.isNull(5));
}

TEST_F(ColumnTest, ClearAllNulls) {
    Column<double> col(10);
    col.setNull(2);
    col.setNull(7);
    EXPECT_TRUE(col.hasNullMask());

    col.clearAllNulls();
    EXPECT_FALSE(col.hasNullMask());
    EXPECT_EQ(col.nullCount(), 0u);
    EXPECT_FALSE(col.isNull(2));
}

TEST_F(ColumnTest, NullCountAccurate) {
    Column<double> col(200);
    for (std::size_t i = 0; i < 200; i += 3) {
        col.setNull(i);
    }
    // Every 3rd element: 0, 3, 6, ..., 198 → 67 nulls
    EXPECT_EQ(col.nullCount(), 67u);
}

// ============================================================
// Column<T> — Alignment
// ============================================================

TEST_F(ColumnTest, Alignment64Byte) {
    Column<double> col(100);
    EXPECT_TRUE(col.isAligned());

    // Verify alignment by checking the pointer
    auto ptrVal = reinterpret_cast<std::uintptr_t>(col.data());
    EXPECT_EQ(ptrVal % 64, 0u);
}

TEST_F(ColumnTest, AlignmentAfterMove) {
    Column<double> a(50);
    EXPECT_TRUE(a.isAligned());
    Column<double> b(std::move(a));
    EXPECT_TRUE(b.isAligned());
}

// ============================================================
// Column<T> — Mmap
// ============================================================

TEST_F(ColumnTest, MmapWrap) {
    double raw[] = {10.0, 20.0, 30.0, 40.0, 50.0};
    auto col = Column<double>::fromMmap(raw, 5);

    EXPECT_EQ(col.size(), 5u);
    EXPECT_TRUE(col.isMmap());
    EXPECT_DOUBLE_EQ(col[2], 30.0);

    // Modify underlying memory → visible through column (zero-copy)
    raw[2] = 99.0;
    EXPECT_DOUBLE_EQ(col[2], 99.0);

    col.releaseMmap();
    EXPECT_FALSE(col.isMmap());
    EXPECT_EQ(col.size(), 0u);
}

TEST_F(ColumnTest, MmapWithNullMask) {
    double raw[] = {1.0, 2.0, 3.0, 4.0};
    uint64_t mask[] = {0b00000101};  // bits 0 and 2 set → elements 0, 2 are null

    auto col = Column<double>::fromMmap(raw, 4, mask);
    EXPECT_TRUE(col.hasNullMask());
    EXPECT_TRUE(col.isNull(0));
    EXPECT_FALSE(col.isNull(1));
    EXPECT_TRUE(col.isNull(2));
    EXPECT_FALSE(col.isNull(3));
}

// ============================================================
// Column<T> — Resize & mutation
// ============================================================

TEST_F(ColumnTest, ResizeGrow) {
    Column<double> col = {1.0, 2.0, 3.0};
    col.resize(5);
    EXPECT_EQ(col.size(), 5u);
    EXPECT_DOUBLE_EQ(col[0], 1.0);
    EXPECT_DOUBLE_EQ(col[1], 2.0);
    EXPECT_DOUBLE_EQ(col[2], 3.0);
    EXPECT_DOUBLE_EQ(col[3], 0.0);  // default-init
    EXPECT_DOUBLE_EQ(col[4], 0.0);
}

TEST_F(ColumnTest, ResizeShrink) {
    Column<double> col = {1.0, 2.0, 3.0, 4.0, 5.0};
    col.resize(3);
    EXPECT_EQ(col.size(), 3u);
    EXPECT_DOUBLE_EQ(col[0], 1.0);
    EXPECT_DOUBLE_EQ(col[2], 3.0);
}

TEST_F(ColumnTest, Clear) {
    Column<double> col = {1.0, 2.0, 3.0};
    col.setNull(1);
    col.clear();
    EXPECT_EQ(col.size(), 0u);
    EXPECT_TRUE(col.empty());
    EXPECT_FALSE(col.hasNullMask());
}

// ============================================================
// Column<T> — Comparison
// ============================================================

TEST_F(ColumnTest, EqualColumns) {
    Column<double> a = {1.0, 2.0, 3.0};
    Column<double> b = {1.0, 2.0, 3.0};
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
}

TEST_F(ColumnTest, NotEqualDifferentSize) {
    Column<double> a = {1.0, 2.0};
    Column<double> b = {1.0, 2.0, 3.0};
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
}

TEST_F(ColumnTest, NotEqualDifferentValues) {
    Column<double> a = {1.0, 2.0, 3.0};
    Column<double> b = {1.0, 2.0, 99.0};
    EXPECT_FALSE(a == b);
}

TEST_F(ColumnTest, NotEqualDifferentNulls) {
    Column<double> a = {1.0, 2.0, 3.0};
    Column<double> b = {1.0, 2.0, 3.0};
    b.setNull(2);  // Mark same value as null
    EXPECT_FALSE(a == b);
}

// ============================================================
// Column<T> — Large capacity
// ============================================================

TEST_F(ColumnTest, LargeColumn) {
    constexpr std::size_t N = 100000;
    Column<double> col(N);
    EXPECT_EQ(col.size(), N);
    EXPECT_TRUE(col.isAligned());

    // Write and read back
    for (std::size_t i = 0; i < N; ++i) {
        col[i] = static_cast<double>(i) * 0.5;
    }
    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(col[i], static_cast<double>(i) * 0.5);
    }
}

// ============================================================
// Column<T> — Memory info
// ============================================================

TEST_F(ColumnTest, MemoryBytes) {
    Column<double> col(100);
    EXPECT_EQ(col.memoryBytes(), 100u * sizeof(double));

    Column<int64_t> col2(50);
    EXPECT_EQ(col2.memoryBytes(), 50u * sizeof(int64_t));
}

// ============================================================
// ColView<T> — Construction & zero-copy
// ============================================================

TEST_F(ColumnTest, ColViewFromColumn) {
    Column<double> col = {10.0, 20.0, 30.0, 40.0, 50.0};
    ColView<double> view(col);

    EXPECT_EQ(view.size(), 5u);
    EXPECT_EQ(view.data(), col.data());  // zero-copy: same pointer
    EXPECT_DOUBLE_EQ(view[0], 10.0);
    EXPECT_DOUBLE_EQ(view[4], 50.0);
}

TEST_F(ColumnTest, ColViewZeroCopySemantics) {
    Column<double> col = {1.0, 2.0, 3.0};
    ColView<double> view(col);

    // Modify column → visible through view
    col[1] = 99.0;
    EXPECT_DOUBLE_EQ(view[1], 99.0);
}

TEST_F(ColumnTest, ColViewSubRange) {
    Column<double> col = {0.0, 1.0, 2.0, 3.0, 4.0};
    ColView<double> view(col, 1, 4);  // [1, 4) = elements 1,2,3

    EXPECT_EQ(view.size(), 3u);
    EXPECT_DOUBLE_EQ(view[0], 1.0);
    EXPECT_DOUBLE_EQ(view[1], 2.0);
    EXPECT_DOUBLE_EQ(view[2], 3.0);
}

TEST_F(ColumnTest, ColViewEmpty) {
    ColView<double> view;
    EXPECT_TRUE(view.empty());
    EXPECT_EQ(view.size(), 0u);
    EXPECT_EQ(view.data(), nullptr);
}

TEST_F(ColumnTest, ColViewFromRawPointer) {
    double raw[] = {100.0, 200.0, 300.0};
    ColView<double> view(raw, 3);
    EXPECT_EQ(view.size(), 3u);
    EXPECT_DOUBLE_EQ(view[0], 100.0);
}

// ============================================================
// ColView<T> — Sub-view slicing
// ============================================================

TEST_F(ColumnTest, ColViewSubView) {
    Column<double> col = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    ColView<double> full(col);
    ColView<double> sub = full.subView(2, 5);  // [2, 5) = elements 2,3,4

    EXPECT_EQ(sub.size(), 3u);
    EXPECT_DOUBLE_EQ(sub[0], 2.0);
    EXPECT_DOUBLE_EQ(sub[1], 3.0);
    EXPECT_DOUBLE_EQ(sub[2], 4.0);
}

TEST_F(ColumnTest, ColViewHead) {
    Column<double> col = {10.0, 20.0, 30.0, 40.0, 50.0};
    ColView<double> view(col);
    auto h = view.head(3);
    EXPECT_EQ(h.size(), 3u);
    EXPECT_DOUBLE_EQ(h[0], 10.0);
    EXPECT_DOUBLE_EQ(h[2], 30.0);
}

TEST_F(ColumnTest, ColViewTail) {
    Column<double> col = {10.0, 20.0, 30.0, 40.0, 50.0};
    ColView<double> view(col);
    auto t = view.tail(2);
    EXPECT_EQ(t.size(), 2u);
    EXPECT_DOUBLE_EQ(t[0], 40.0);
    EXPECT_DOUBLE_EQ(t[1], 50.0);
}

// ============================================================
// ColView<T> — Null mask forwarding
// ============================================================

TEST_F(ColumnTest, ColViewNullMaskForwarding) {
    Column<double> col = {1.0, 2.0, 3.0, 4.0, 5.0};
    col.setNull(1);
    col.setNull(3);

    ColView<double> view(col);
    EXPECT_TRUE(view.hasNullMask());
    EXPECT_FALSE(view.isNull(0));
    EXPECT_TRUE(view.isNull(1));
    EXPECT_FALSE(view.isNull(2));
    EXPECT_TRUE(view.isNull(3));
    EXPECT_FALSE(view.isNull(4));
}

TEST_F(ColumnTest, ColViewNullMaskAfterSlice) {
    Column<double> col(10);
    col.setNull(0);   // position 0 is null
    col.setNull(65);  // in second uint64

    // Take a sub-view starting after element 0
    ColView<double> view(col, 1, 10);
    EXPECT_TRUE(view.hasNullMask());
    EXPECT_FALSE(view.isNull(0));  // col[1] — was not set to null
    EXPECT_FALSE(view.isNull(1));  // col[2]
    EXPECT_FALSE(view.isNull(3));  // col[4]
    EXPECT_FALSE(view.isNull(8));  // col[9]
}

// ============================================================
// ColView<T> — Iterator support
// ============================================================

TEST_F(ColumnTest, ColViewRangeForLoop) {
    Column<double> col = {1.0, 2.0, 3.0};
    ColView<double> view(col);

    double expected = 1.0;
    for (const double& val : view) {
        EXPECT_DOUBLE_EQ(val, expected);
        expected += 1.0;
    }
}

// ============================================================
// Factory helpers
// ============================================================

TEST_F(ColumnTest, MakeColViewHelpers) {
    Column<double> col = {5.0, 6.0, 7.0, 8.0};

    auto v1 = makeColView(col);
    EXPECT_EQ(v1.size(), 4u);

    auto v2 = makeColView(col, 1, 3);
    EXPECT_EQ(v2.size(), 2u);
    EXPECT_DOUBLE_EQ(v2[0], 6.0);
    EXPECT_DOUBLE_EQ(v2[1], 7.0);

    double raw[] = {9.0, 10.0};
    auto v3 = makeColView(raw, 2);
    EXPECT_EQ(v3.size(), 2u);
    EXPECT_DOUBLE_EQ(v3[0], 9.0);
}

// ============================================================
// Edge cases
// ============================================================

TEST_F(ColumnTest, SingleElement) {
    Column<double> col = {42.0};
    EXPECT_EQ(col.size(), 1u);
    EXPECT_DOUBLE_EQ(col[0], 42.0);

    col.setNull(0);
    EXPECT_TRUE(col.isNull(0));
    EXPECT_EQ(col.nullCount(), 1u);
}

TEST_F(ColumnTest, Uint8ColumnForBool) {
    // Use Column<uint8_t> for boolean/signal storage
    // (Column<bool> is intentionally unsupported — std::vector<bool> is a bitset)
    Column<uint8_t> col = {0, 1, 1, 0, 1};
    EXPECT_EQ(col.size(), 5u);
    EXPECT_EQ(col[0], 0u);
    EXPECT_EQ(col[1], 1u);
    EXPECT_EQ(col[2], 1u);
    EXPECT_EQ(col[3], 0u);
    EXPECT_EQ(col[4], 1u);
}

TEST_F(ColumnTest, Uint8Column) {
    Column<uint8_t> col = {0, 128, 255};
    EXPECT_EQ(col.size(), 3u);
    EXPECT_EQ(col[0], 0);
    EXPECT_EQ(col[1], 128);
    EXPECT_EQ(col[2], 255);
}

TEST_F(ColumnTest, NullMaskAfterResize) {
    Column<double> col(5);
    col.setNull(2);
    col.resize(10);
    // After grow, null mask should be resized; position 2 still null
    EXPECT_TRUE(col.hasNullMask());
    EXPECT_TRUE(col.isNull(2));
    EXPECT_FALSE(col.isNull(7));  // new elements not null
}

TEST_F(ColumnTest, NullMaskSurvivesCopy) {
    Column<double> original = {1.0, 2.0, 3.0};
    original.setNull(1);

    Column<double> copy(original);
    EXPECT_TRUE(copy.hasNullMask());
    EXPECT_TRUE(copy.isNull(1));
    EXPECT_FALSE(copy.isNull(0));

    // Verify deep copy of null mask
    original.clearNull(1);
    EXPECT_TRUE(copy.isNull(1));  // copy unaffected
}
