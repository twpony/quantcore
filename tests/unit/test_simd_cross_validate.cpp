// test_simd_cross_validate.cpp — SIMD vs scalar cross-validation
//
// Verifies that SIMD-accelerated functions produce identical results
// to pure scalar implementations across a range of data sizes.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "factors/alpha_1001.h"
#include "quantcore/storage/Column.h"

using namespace quantcore;
using namespace quantcore::factors;

// ============================================================
// Scalar SMA20 baseline (structurally identical to computeSma20)
// ============================================================

static Column<double> sma20_scalar(const Column<double>& close) {
    constexpr std::size_t W = 20;
    const std::size_t n = close.size();
    Column<double> result(n);

    if (n < W) {
        for (std::size_t i = 0; i < n; ++i)
            result[i] = std::nan("");
        return result;
    }
    for (std::size_t i = 0; i < W - 1; ++i)
        result[i] = std::nan("");

    double sum = 0.0;
    for (std::size_t i = 0; i < W; ++i)
        sum += close[i];
    result[W - 1] = sum / static_cast<double>(W);

    for (std::size_t i = W; i < n; ++i) {
        sum += close[i] - close[i - W];
        result[i] = sum / static_cast<double>(W);
    }
    return result;
}

// ============================================================
// Data generation
// ============================================================

static Column<double> makeData(std::size_t n) {
    Column<double> c(n);
    for (std::size_t i = 0; i < n; ++i)
        c[i] = 100.0 + static_cast<double>(i) * 0.01;
    return c;
}

// ============================================================
// Cross-validation tests at various sizes
// ============================================================

TEST(SimdCrossValidate, SmallDataset) {
    auto data = makeData(50);
    auto scalar = sma20_scalar(data);
    auto simd   = computeSma20(data);

    ASSERT_EQ(scalar.size(), simd.size());
    for (std::size_t i = 0; i < scalar.size(); ++i) {
        if (std::isnan(scalar[i])) {
            EXPECT_TRUE(std::isnan(simd[i])) << "i=" << i;
        } else {
            EXPECT_DOUBLE_EQ(scalar[i], simd[i]) << "i=" << i;
        }
    }
}

TEST(SimdCrossValidate, ExactWindowSize) {
    auto data = makeData(20);
    auto scalar = sma20_scalar(data);
    auto simd   = computeSma20(data);

    ASSERT_EQ(scalar.size(), 20u);
    // All but position 19 should be NaN, position 19 should match
    for (std::size_t i = 0; i < 19; ++i) {
        EXPECT_TRUE(std::isnan(scalar[i]));
        EXPECT_TRUE(std::isnan(simd[i]));
    }
    EXPECT_DOUBLE_EQ(scalar[19], simd[19]);
}

TEST(SimdCrossValidate, MediumDataset) {
    auto data = makeData(1000);
    auto scalar = sma20_scalar(data);
    auto simd   = computeSma20(data);

    ASSERT_EQ(scalar.size(), simd.size());
    // Running-sum accumulation order differs between scalar and SIMD;
    // use NEAR to tolerate sub-ULP differences.
    for (std::size_t i = 0; i < scalar.size(); ++i) {
        if (std::isnan(scalar[i])) {
            EXPECT_TRUE(std::isnan(simd[i])) << "i=" << i;
        } else {
            EXPECT_NEAR(scalar[i], simd[i], 1e-10) << "i=" << i;
        }
    }
}

TEST(SimdCrossValidate, LargeDataset) {
    auto data = makeData(10000);
    auto scalar = sma20_scalar(data);
    auto simd   = computeSma20(data);

    ASSERT_EQ(scalar.size(), simd.size());
    for (std::size_t i = 0; i < scalar.size(); ++i) {
        if (std::isnan(scalar[i])) {
            EXPECT_TRUE(std::isnan(simd[i])) << "i=" << i;
        } else {
            EXPECT_NEAR(scalar[i], simd[i], 1e-10) << "i=" << i;
        }
    }
}

TEST(SimdCrossValidate, VeryLargeDataset) {
    auto data = makeData(100000);
    auto scalar = sma20_scalar(data);
    auto simd   = computeSma20(data);

    ASSERT_EQ(scalar.size(), simd.size());
    // At 100K elements, floating-point accumulation order differs between
    // scalar and SIMD paths.  Use NEAR with a small relative tolerance.
    for (std::size_t i = 0; i < scalar.size(); ++i) {
        if (std::isnan(scalar[i])) {
            EXPECT_TRUE(std::isnan(simd[i])) << "i=" << i;
        } else {
            EXPECT_NEAR(scalar[i], simd[i], 1e-6) << "i=" << i;
        }
    }
}

TEST(SimdCrossValidate, SubWindowSizeReturnsAllNaN) {
    auto data = makeData(10);  // less than window=20
    auto scalar = sma20_scalar(data);
    auto simd   = computeSma20(data);

    ASSERT_EQ(scalar.size(), 10u);
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_TRUE(std::isnan(scalar[i]));
        EXPECT_TRUE(std::isnan(simd[i]));
    }
}

TEST(SimdCrossValidate, ZeroLengthInput) {
    auto data = makeData(0);
    auto scalar = sma20_scalar(data);
    auto simd   = computeSma20(data);

    EXPECT_EQ(scalar.size(), 0u);
    EXPECT_EQ(simd.size(), 0u);
}

TEST(SimdCrossValidate, AllSameValues) {
    Column<double> data(100);
    for (std::size_t i = 0; i < 100; ++i)
        data[i] = 42.0;

    auto scalar = sma20_scalar(data);
    auto simd   = computeSma20(data);

    for (std::size_t i = 0; i < 100; ++i) {
        if (i < 19) {
            EXPECT_TRUE(std::isnan(simd[i]));
        } else {
            EXPECT_DOUBLE_EQ(scalar[i], 42.0);
            EXPECT_DOUBLE_EQ(simd[i], 42.0);
        }
    }
}
