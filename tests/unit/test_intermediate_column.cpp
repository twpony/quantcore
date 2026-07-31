// test_intermediate_column.cpp — unit tests for IntermediateColumn
// Phase: 五期实现

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "quantcore/engine/BufferPool.h"
#include "quantcore/storage/IntermediateColumn.h"

using namespace quantcore;

class IntermediateColumnTest : public ::testing::Test {
protected:
    static constexpr std::size_t kNAssets = 3;
    static constexpr std::size_t kNDates  = 10;

    BufferPool pool_;

    // Helper: fill the intermediate column with known values:
    //   data[a][d] = base + a * 10.0 + d * 1.0
    void fillWithKnownValues(IntermediateColumn& col) {
        for (std::size_t a = 0; a < kNAssets; ++a) {
            double* out = col.timeSeriesData(a);
            double base = 100.0;
            for (std::size_t d = 0; d < kNDates; ++d) {
                out[d] = base + static_cast<double>(a) * 10.0
                               + static_cast<double>(d) * 1.0;
            }
        }
    }
};

// ============================================================
// Construction and dimensions
// ============================================================

TEST_F(IntermediateColumnTest, DefaultConstructEmpty) {
    IntermediateColumn col;
    EXPECT_TRUE(col.empty());
    EXPECT_EQ(col.assetCount(), 0u);
    EXPECT_EQ(col.timePointCount(), 0u);
}

TEST_F(IntermediateColumnTest, ConstructWithDimensions) {
    IntermediateColumn col(kNAssets, kNDates, pool_);

    EXPECT_FALSE(col.empty());
    EXPECT_EQ(col.assetCount(), kNAssets);
    EXPECT_EQ(col.timePointCount(), kNDates);
    EXPECT_TRUE(col.isFullyComputed());  // eager allocation
}

TEST_F(IntermediateColumnTest, ConstructZeroAssets) {
    IntermediateColumn col(0, 10, pool_);
    EXPECT_TRUE(col.empty());
    EXPECT_EQ(col.assetCount(), 0u);
    EXPECT_EQ(col.timePointCount(), 10u);
}

TEST_F(IntermediateColumnTest, ConstructZeroDates) {
    IntermediateColumn col(3, 0, pool_);
    EXPECT_EQ(col.assetCount(), 3u);
    EXPECT_EQ(col.timePointCount(), 0u);
}

// ============================================================
// Time-series access — write and read
// ============================================================

TEST_F(IntermediateColumnTest, TimeSeriesDataMutable) {
    IntermediateColumn col(kNAssets, kNDates, pool_);
    fillWithKnownValues(col);

    // Read back and verify
    for (std::size_t a = 0; a < kNAssets; ++a) {
        const double* data = col.timeSeriesData(a);
        double base = 100.0 + static_cast<double>(a) * 10.0;
        for (std::size_t d = 0; d < kNDates; ++d) {
            EXPECT_DOUBLE_EQ(data[d], base + static_cast<double>(d) * 1.0);
        }
    }
}

TEST_F(IntermediateColumnTest, TimeSeriesDataIsContiguous) {
    IntermediateColumn col(kNAssets, kNDates, pool_);
    double* data = col.timeSeriesData(0);

    // All kNDates elements should be contiguous in memory
    for (std::size_t d = 0; d < kNDates; ++d) {
        data[d] = static_cast<double>(d);
    }
    for (std::size_t d = 0; d < kNDates; ++d) {
        EXPECT_EQ(&data[d], col.timeSeriesData(0) + d);
        EXPECT_DOUBLE_EQ(data[d], static_cast<double>(d));
    }
}

TEST_F(IntermediateColumnTest, DifferentAssetsHaveDifferentBuffers) {
    IntermediateColumn col(kNAssets, kNDates, pool_);

    double* d0 = col.timeSeriesData(0);
    double* d1 = col.timeSeriesData(1);
    double* d2 = col.timeSeriesData(2);

    // Each asset should have its own buffer
    EXPECT_NE(d0, d1);
    EXPECT_NE(d1, d2);
    EXPECT_NE(d0, d2);

    // Writing to one should not affect others
    for (std::size_t d = 0; d < kNDates; ++d) {
        d0[d] = 1.0;
        d1[d] = 2.0;
        d2[d] = 3.0;
    }
    for (std::size_t d = 0; d < kNDates; ++d) {
        EXPECT_DOUBLE_EQ(d0[d], 1.0);
        EXPECT_DOUBLE_EQ(d1[d], 2.0);
        EXPECT_DOUBLE_EQ(d2[d], 3.0);
    }
}

// ============================================================
// Null mask
// ============================================================

TEST_F(IntermediateColumnTest, NullMaskDefaultsToNullptr) {
    IntermediateColumn col(kNAssets, kNDates, pool_);
    for (std::size_t a = 0; a < kNAssets; ++a) {
        EXPECT_EQ(col.timeSeriesNullMask(a), nullptr);
        EXPECT_FALSE(col.isNull(a, 0));
    }
}

TEST_F(IntermediateColumnTest, SetAndCheckNullMask) {
    IntermediateColumn col(kNAssets, kNDates, pool_);

    // Manually create a null mask using the bit helpers from Column.h
    std::size_t words = (kNDates + 63) / 64;
    std::vector<uint64_t> mask(words, 0);
    // Mark date 3 as null for asset 1
    mask[3 / 64] |= (uint64_t{1} << (3 % 64));

    col.setTimeSeriesNullMask(1, mask.data());

    EXPECT_EQ(col.timeSeriesNullMask(1), mask.data());
    EXPECT_TRUE(col.isNull(1, 3));
    EXPECT_FALSE(col.isNull(1, 0));
    EXPECT_FALSE(col.isNull(0, 3));  // asset 0 unaffected
}

// ============================================================
// Cross-sectional gather
// ============================================================

TEST_F(IntermediateColumnTest, GatherCrossSection) {
    IntermediateColumn col(kNAssets, kNDates, pool_);
    fillWithKnownValues(col);

    // Gather at date 5
    std::vector<double> gathered(kNAssets);
    std::size_t valid = col.gatherCrossSection(5, gathered.data(), kNAssets);

    EXPECT_EQ(valid, kNAssets);  // no nulls

    // Verify each asset's value at date 5
    for (std::size_t a = 0; a < kNAssets; ++a) {
        double expected = 100.0 + static_cast<double>(a) * 10.0 + 5.0;
        EXPECT_DOUBLE_EQ(gathered[a], expected);
    }
}

TEST_F(IntermediateColumnTest, GatherCrossSectionFirstDate) {
    IntermediateColumn col(kNAssets, kNDates, pool_);
    fillWithKnownValues(col);

    std::vector<double> gathered(kNAssets);
    col.gatherCrossSection(0, gathered.data(), kNAssets);

    for (std::size_t a = 0; a < kNAssets; ++a) {
        double expected = 100.0 + static_cast<double>(a) * 10.0 + 0.0;
        EXPECT_DOUBLE_EQ(gathered[a], expected);
    }
}

TEST_F(IntermediateColumnTest, GatherCrossSectionLastDate) {
    IntermediateColumn col(kNAssets, kNDates, pool_);
    fillWithKnownValues(col);

    std::vector<double> gathered(kNAssets);
    col.gatherCrossSection(kNDates - 1, gathered.data(), kNAssets);

    for (std::size_t a = 0; a < kNAssets; ++a) {
        double expected = 100.0 + static_cast<double>(a) * 10.0 + static_cast<double>(kNDates - 1);
        EXPECT_DOUBLE_EQ(gathered[a], expected);
    }
}

TEST_F(IntermediateColumnTest, GatherWithNullValues) {
    IntermediateColumn col(kNAssets, kNDates, pool_);
    fillWithKnownValues(col);

    // Mark asset 1, date 3 as null
    std::size_t words = (kNDates + 63) / 64;
    std::vector<uint64_t> mask(words, 0);
    mask[3 / 64] |= (uint64_t{1} << (3 % 64));
    col.setTimeSeriesNullMask(1, mask.data());

    // Gather at date 3
    std::vector<double> gathered(kNAssets);
    std::size_t valid = col.gatherCrossSection(3, gathered.data(), kNAssets);

    // Asset 1 should be NaN, counted as invalid
    EXPECT_EQ(valid, kNAssets - 1);
    EXPECT_FALSE(std::isnan(gathered[0]));
    EXPECT_TRUE(std::isnan(gathered[1]));
    EXPECT_FALSE(std::isnan(gathered[2]));
}

TEST_F(IntermediateColumnTest, GatherOutputSmallerThanAssetCount) {
    IntermediateColumn col(5, kNDates, pool_);
    fillWithKnownValues(col);  // only fills first 3

    // Request output for only 3 assets
    std::vector<double> gathered(3);
    std::size_t valid = col.gatherCrossSection(0, gathered.data(), 3);
    EXPECT_EQ(valid, 3u);
}

// ============================================================
// Gather with NaN in source data (not null, but NaN)
// ============================================================

TEST_F(IntermediateColumnTest, GatherIgnoresSourceNaN) {
    IntermediateColumn col(kNAssets, kNDates, pool_);

    // Asset 0: normal, Asset 1: NaN at date 2, Asset 2: normal
    for (std::size_t a = 0; a < kNAssets; ++a) {
        double* out = col.timeSeriesData(a);
        for (std::size_t d = 0; d < kNDates; ++d) {
            out[d] = static_cast<double>(a * 10 + d);
        }
    }
    col.timeSeriesData(1)[2] = std::numeric_limits<double>::quiet_NaN();

    std::vector<double> gathered(kNAssets);
    std::size_t valid = col.gatherCrossSection(2, gathered.data(), kNAssets);

    EXPECT_EQ(valid, kNAssets - 1);  // one NaN
    EXPECT_FALSE(std::isnan(gathered[0]));
    EXPECT_TRUE(std::isnan(gathered[1]));
    EXPECT_FALSE(std::isnan(gathered[2]));
}

// ============================================================
// gatherCrossSectionView — wraps in ColView
// ============================================================

TEST_F(IntermediateColumnTest, GatherCrossSectionView) {
    IntermediateColumn col(kNAssets, kNDates, pool_);
    fillWithKnownValues(col);

    std::vector<double> buffer(kNAssets);
    ColView<double> view = col.gatherCrossSectionView(5, buffer.data(), kNAssets);

    EXPECT_EQ(view.size(), kNAssets);
    for (std::size_t a = 0; a < kNAssets; ++a) {
        double expected = 100.0 + static_cast<double>(a) * 10.0 + 5.0;
        EXPECT_DOUBLE_EQ(view[a], expected);
    }
}

// ============================================================
// Memory diagnostics
// ============================================================

TEST_F(IntermediateColumnTest, MemoryBytes) {
    IntermediateColumn col(100, 2500, pool_);
    EXPECT_EQ(col.memoryBytes(), 100u * 2500u * sizeof(double));
}

// ============================================================
// Move semantics
// ============================================================

TEST_F(IntermediateColumnTest, MoveConstruct) {
    IntermediateColumn col1(kNAssets, kNDates, pool_);
    fillWithKnownValues(col1);

    double* origPtr = col1.timeSeriesData(0);

    IntermediateColumn col2 = std::move(col1);

    EXPECT_EQ(col2.assetCount(), kNAssets);
    EXPECT_EQ(col2.timePointCount(), kNDates);
    EXPECT_EQ(col2.timeSeriesData(0), origPtr);  // same underlying buffer

    // Original should be empty
    EXPECT_TRUE(col1.empty());
    EXPECT_EQ(col1.assetCount(), 0u);
}

TEST_F(IntermediateColumnTest, MoveAssign) {
    IntermediateColumn col1(kNAssets, kNDates, pool_);
    fillWithKnownValues(col1);

    IntermediateColumn col2;
    col2 = std::move(col1);

    EXPECT_EQ(col2.assetCount(), kNAssets);
    EXPECT_TRUE(col1.empty());
}
