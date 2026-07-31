// test_panel_data.cpp — unit tests for PanelData and cross-sectional factors
// Phase: 五期实现
//
// Tests PanelData construction, asset access, crossSection() correctness,
// timestamp validation, and FactorCalculator cross-sectional factor
// registration and evaluation.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "quantcore/core/FactorCalculator.h"
#include "quantcore/core/Types.h"
#include "quantcore/expression/ColumnRef.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/expression/UnaryExpr.h"
#include "quantcore/registry/OperatorRegistry.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/IntermediateColumn.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/PanelData.h"
#include "quantcore/storage/TimestampIndex.h"

using namespace quantcore;

// ============================================================
// Test fixture — builds a synthetic 3-asset × 10-date panel
// ============================================================

class PanelDataTest : public ::testing::Test {
protected:
    static constexpr std::size_t kNStocks = 3;
    static constexpr std::size_t kNDates  = 10;

    void SetUp() override {
        // Common timestamp axis: 10 sequential dates.
        std::vector<int64_t> timestamps(kNDates);
        for (std::size_t i = 0; i < kNDates; ++i) {
            timestamps[i] = static_cast<int64_t>(i);
        }
        TimestampIndex tsIdx(timestamps.data(), kNDates);

        // 3 assets: "S1", "S2", "S3"
        // Each field = base + stock_offset + date_offset
        // So at date d, stock s: value = base[field] + s * 10.0 + d * 1.0
        std::vector<MarketData> assets;
        for (std::size_t s = 0; s < kNStocks; ++s) {
            std::string id = "S" + std::to_string(s + 1);
            MarketData md(id, tsIdx);

            for (std::size_t f = 0; f < static_cast<std::size_t>(Field::kFieldCount); ++f) {
                auto field = static_cast<Field>(f);
                Column<double> col(kNDates);
                double base = 0.0;
                switch (field) {
                    case Field::OPEN:   base = 10.0; break;
                    case Field::HIGH:   base = 15.0; break;
                    case Field::LOW:    base =  8.0; break;
                    case Field::CLOSE:  base = 12.0; break;
                    case Field::VOLUME: base = 100.0; break;
                    case Field::AMOUNT: base = 1000.0; break;
                    case Field::VWAP:   base = 11.5; break;
                    default: break;
                }
                for (std::size_t d = 0; d < kNDates; ++d) {
                    col[d] = base + static_cast<double>(s) * 10.0
                                      + static_cast<double>(d) * 1.0;
                }
                md.setColumn(field, std::move(col));
            }
            assets.push_back(std::move(md));
        }

        panel_ = std::make_unique<PanelData>(std::move(assets));
    }

    /// Build a fresh assets vector for tests that need one.
    std::vector<MarketData> buildAssets() const {
        std::vector<int64_t> timestamps(kNDates);
        for (std::size_t i = 0; i < kNDates; ++i)
            timestamps[i] = static_cast<int64_t>(i);
        TimestampIndex tsIdx(timestamps.data(), kNDates);

        std::vector<MarketData> assets;
        for (std::size_t s = 0; s < kNStocks; ++s) {
            std::string id = "S" + std::to_string(s + 1);
            MarketData md(id, tsIdx);
            for (std::size_t f = 0; f < static_cast<std::size_t>(Field::kFieldCount); ++f) {
                auto field = static_cast<Field>(f);
                Column<double> col(kNDates);
                double base = 0.0;
                switch (field) {
                    case Field::OPEN:   base = 10.0; break;
                    case Field::HIGH:   base = 15.0; break;
                    case Field::LOW:    base =  8.0; break;
                    case Field::CLOSE:  base = 12.0; break;
                    case Field::VOLUME: base = 100.0; break;
                    case Field::AMOUNT: base = 1000.0; break;
                    case Field::VWAP:   base = 11.5; break;
                    default: break;
                }
                for (std::size_t d = 0; d < kNDates; ++d) {
                    col[d] = base + static_cast<double>(s) * 10.0
                                      + static_cast<double>(d) * 1.0;
                }
                md.setColumn(field, std::move(col));
            }
            assets.push_back(std::move(md));
        }
        return assets;
    }

    // Expected value at (stockIdx, dateIdx, field)
    double expectedValue(std::size_t stockIdx,
                         std::size_t dateIdx,
                         Field field) const {
        double base = 0.0;
        switch (field) {
            case Field::OPEN:   base = 10.0; break;
            case Field::HIGH:   base = 15.0; break;
            case Field::LOW:    base =  8.0; break;
            case Field::CLOSE:  base = 12.0; break;
            case Field::VOLUME: base = 100.0; break;
            case Field::AMOUNT: base = 1000.0; break;
            case Field::VWAP:   base = 11.5; break;
            default: break;
        }
        return base + static_cast<double>(stockIdx) * 10.0
                    + static_cast<double>(dateIdx) * 1.0;
    }

    std::unique_ptr<PanelData> panel_;
    BufferPool pool_;
};

// ============================================================
// Construction tests
// ============================================================

TEST_F(PanelDataTest, ConstructEmpty) {
    PanelData empty;
    EXPECT_TRUE(empty.empty());
    EXPECT_EQ(empty.assetCount(), 0);
    EXPECT_EQ(empty.dateCount(), 0);
}

TEST_F(PanelDataTest, ConstructFromAssets) {
    EXPECT_EQ(panel_->assetCount(), kNStocks);
    EXPECT_EQ(panel_->dateCount(), kNDates);
    EXPECT_FALSE(panel_->empty());
    EXPECT_TRUE(panel_->isValid());
}

TEST_F(PanelDataTest, AssetMetadata) {
    EXPECT_EQ(panel_->assetId(0), "S1");
    EXPECT_EQ(panel_->assetId(1), "S2");
    EXPECT_EQ(panel_->assetId(2), "S3");

    EXPECT_EQ(panel_->assetIndex("S1"), 0);
    EXPECT_EQ(panel_->assetIndex("S3"), 2);

    EXPECT_THROW(panel_->assetIndex("UNKNOWN"), std::out_of_range);
}

TEST_F(PanelDataTest, TimestampQueries) {
    for (std::size_t d = 0; d < kNDates; ++d) {
        EXPECT_EQ(panel_->timestampAt(d), static_cast<int64_t>(d));
    }
}

// ============================================================
// Time-series access
// ============================================================

TEST_F(PanelDataTest, AssetAccess) {
    const MarketData& s1 = panel_->asset(0);
    EXPECT_EQ(s1.assetId(), "S1");
    EXPECT_EQ(s1.rowCount(), kNDates);

    // Verify data integrity via MarketData access.
    const auto& close = s1.column<double>(Field::CLOSE);
    for (std::size_t d = 0; d < kNDates; ++d) {
        EXPECT_DOUBLE_EQ(close[d], expectedValue(0, d, Field::CLOSE));
    }
}

TEST_F(PanelDataTest, AssetById) {
    const MarketData& s2 = panel_->assetById("S2");
    EXPECT_EQ(s2.assetId(), "S2");
}

// ============================================================
// Cross-sectional access
// ============================================================

TEST_F(PanelDataTest, CrossSectionBasic) {
    // Get cross-section of CLOSE at date 5 for all 3 assets.
    auto cs = panel_->crossSection(5, Field::CLOSE, pool_);

    EXPECT_EQ(cs.view.size(), kNStocks);

    // S0: 12.0 + 0*10 + 5 = 17.0
    // S1: 12.0 + 1*10 + 5 = 27.0
    // S2: 12.0 + 2*10 + 5 = 37.0
    EXPECT_DOUBLE_EQ(cs.view[0], expectedValue(0, 5, Field::CLOSE));
    EXPECT_DOUBLE_EQ(cs.view[1], expectedValue(1, 5, Field::CLOSE));
    EXPECT_DOUBLE_EQ(cs.view[2], expectedValue(2, 5, Field::CLOSE));
}

TEST_F(PanelDataTest, CrossSectionAllDates) {
    // Verify crossSection at every date for VOLUME.
    for (std::size_t d = 0; d < kNDates; ++d) {
        auto cs = panel_->crossSection(d, Field::VOLUME, pool_);
        EXPECT_EQ(cs.view.size(), kNStocks);
        for (std::size_t s = 0; s < kNStocks; ++s) {
            EXPECT_DOUBLE_EQ(cs.view[s], expectedValue(s, d, Field::VOLUME));
        }
    }
}

TEST_F(PanelDataTest, CrossSectionAllFields) {
    // Verify crossSection works for every field.
    for (std::size_t f = 0; f < static_cast<std::size_t>(Field::kFieldCount); ++f) {
        auto field = static_cast<Field>(f);
        auto cs = panel_->crossSection(3, field, pool_);
        EXPECT_EQ(cs.view.size(), kNStocks);
        for (std::size_t s = 0; s < kNStocks; ++s) {
            EXPECT_DOUBLE_EQ(cs.view[s], expectedValue(s, 3, field));
        }
    }
}

// ============================================================
// Validation tests
// ============================================================

TEST_F(PanelDataTest, MismatchedTimestampRejected) {
    // Create an asset with different number of timestamps.
    std::vector<int64_t> shortTs(5);
    for (std::size_t i = 0; i < 5; ++i) shortTs[i] = static_cast<int64_t>(i);
    TimestampIndex shortIdx(shortTs.data(), 5);
    MarketData badMd("BAD", shortIdx);

    EXPECT_THROW(panel_->addAsset(std::move(badMd)), std::invalid_argument);
}

TEST_F(PanelDataTest, IsValid) {
    EXPECT_TRUE(panel_->isValid());
}

TEST_F(PanelDataTest, NullCount) {
    EXPECT_EQ(panel_->nullCount(0, Field::CLOSE), 0);
    EXPECT_EQ(panel_->nullCount(5, Field::VOLUME), 0);
}

// ============================================================
// FactorCalculator CS factor tests
// ============================================================

class CsFactorTest : public PanelDataTest {
protected:
    void SetUp() override {
        PanelDataTest::SetUp();
    }

    FactorCalculator calc_;
};

TEST_F(CsFactorTest, RegisterAndEvaluateCsRank) {
    calc_.registerCrossSectionalFactor(
        "rank_close", CsOpCode::CS_RANK, Field::CLOSE);

    EXPECT_TRUE(calc_.isCrossSectionalFactor("rank_close"));
    EXPECT_FALSE(calc_.isCrossSectionalFactor("nonexistent"));

    auto results = calc_.evaluateCS("rank_close", *panel_);

    // results[a][d] = rank of asset a at date d
    EXPECT_EQ(results.size(), kNStocks);
    for (auto& col : results) {
        EXPECT_EQ(col.size(), kNDates);
    }

    // At each date, the CLOSE values are:
    //   S0: 12 + 0*10 + d = 12 + d  (smallest)
    //   S1: 12 + 1*10 + d = 22 + d
    //   S2: 12 + 2*10 + d = 32 + d  (largest)
    // Ranks should be: S0=1, S1=2, S2=3 (1-based, average for ties)
    // Actually the rank depends on the CS_RANK implementation
    // For 3 distinct values, ranks should be distinct.
    for (std::size_t d = 0; d < kNDates; ++d) {
        // All ranks should be >= 1 and <= 3
        for (std::size_t a = 0; a < kNStocks; ++a) {
            EXPECT_GE(results[a][d], 1.0);
            EXPECT_LE(results[a][d], 3.0);
        }
        // S0 < S1 < S2, so rank(S0) == 1, rank(S1) == 2, rank(S2) == 3
        EXPECT_DOUBLE_EQ(results[0][d], 1.0);
        EXPECT_DOUBLE_EQ(results[1][d], 2.0);
        EXPECT_DOUBLE_EQ(results[2][d], 3.0);
    }
}

TEST_F(CsFactorTest, EvaluateCsDirectly) {
    // Evaluate CS_ZSCORE without registration.
    auto results = calc_.evaluateCS(
        CsOpCode::CS_ZSCORE, Field::CLOSE, *panel_);

    EXPECT_EQ(results.size(), kNStocks);
    for (auto& col : results) {
        EXPECT_EQ(col.size(), kNDates);
    }

    // Z-scores should be mean ~= 0 and sum ~= 0 for each date.
    for (std::size_t d = 0; d < kNDates; ++d) {
        double sum = 0.0;
        for (std::size_t a = 0; a < kNStocks; ++a) {
            sum += results[a][d];
        }
        EXPECT_NEAR(sum, 0.0, 1e-10);
    }
}

TEST_F(CsFactorTest, Unregistration) {
    calc_.registerCrossSectionalFactor(
        "temp_cs", CsOpCode::CS_RANK, Field::VOLUME);
    EXPECT_TRUE(calc_.isCrossSectionalFactor("temp_cs"));

    calc_.unregisterCrossSectionalFactor("temp_cs");
    EXPECT_FALSE(calc_.isCrossSectionalFactor("temp_cs"));

    EXPECT_THROW(calc_.evaluateCS("temp_cs", *panel_), std::runtime_error);
}

TEST_F(CsFactorTest, CsFactorNames) {
    calc_.registerCrossSectionalFactor("cs_a", CsOpCode::CS_RANK, Field::CLOSE);
    calc_.registerCrossSectionalFactor("cs_b", CsOpCode::CS_ZSCORE, Field::VOLUME);

    auto names = calc_.csFactorNames();
    EXPECT_EQ(names.size(), 2);
}

TEST_F(CsFactorTest, CsFactorHasDescription) {
    calc_.registerCrossSectionalFactor("my_rank", CsOpCode::CS_RANK, Field::CLOSE);
    std::string desc = calc_.formulaExpression("my_rank");
    // Description should be non-empty and start with "CS_".
    EXPECT_FALSE(desc.empty());
    EXPECT_NE(desc.find("CS_"), std::string::npos);
}

TEST_F(CsFactorTest, EmptyPanelReturnsEmpty) {
    PanelData emptyPanel;
    auto results = calc_.evaluateCS(
        CsOpCode::CS_RANK, Field::CLOSE, emptyPanel);
    EXPECT_TRUE(results.empty());
}

// ============================================================
// Integration: time-series + cross-sectional workflow
// ============================================================

TEST_F(CsFactorTest, TsAndCsWorkflow) {
    // 1. Register a time-series factor (momentum-like).
    calc_.registerFormula("simple_ma", "close");

    // 2. Evaluate time-series factor per asset.
    std::vector<Column<double>> tsResults;
    for (std::size_t a = 0; a < panel_->assetCount(); ++a) {
        tsResults.push_back(calc_.evaluate("simple_ma", panel_->asset(a)));
    }
    EXPECT_EQ(tsResults.size(), kNStocks);
    for (auto& col : tsResults) {
        EXPECT_EQ(col.size(), kNDates);
    }

    // 3. Register and evaluate a cross-sectional factor.
    calc_.registerCrossSectionalFactor(
        "rank_vol", CsOpCode::CS_RANK, Field::VOLUME);
    auto csResults = calc_.evaluateCS("rank_vol", *panel_);
    EXPECT_EQ(csResults.size(), kNStocks);
    for (auto& col : csResults) {
        EXPECT_EQ(col.size(), kNDates);
    }
}

// ============================================================
// Move semantics
// ============================================================

TEST_F(PanelDataTest, MoveConstruct) {
    auto assets = buildAssets();
    PanelData src(std::move(assets));
    PanelData moved(std::move(src));

    EXPECT_EQ(moved.assetCount(), kNStocks);
    EXPECT_EQ(moved.dateCount(), kNDates);
    EXPECT_FALSE(moved.empty());
    EXPECT_EQ(moved.assetId(0), "S1");
    auto cs = moved.crossSection(0, Field::CLOSE, pool_);
    EXPECT_EQ(cs.view.size(), kNStocks);
}

// ============================================================
// CS expression evaluation tests — cs_rank(log(volume)) etc.
// ============================================================

class CsExprEvalTest : public PanelDataTest {
protected:
    void SetUp() override {
        PanelDataTest::SetUp();
    }

    FactorCalculator calc_;
};

TEST_F(CsExprEvalTest, EvaluateCSExpressionRankLogVolume) {
    // cs_rank(log(volume)): rank all assets by log(volume) at each date.
    //
    // Volume values:  S0=100+d, S1=110+d, S2=120+d
    // log(volume) is monotonic, so ranks are S0=1, S1=2, S2=3
    auto results = calc_.evaluateCSExpression(
        "log(volume)", *panel_, CsOpCode::CS_RANK);

    EXPECT_EQ(results.size(), kNStocks);
    for (auto& col : results) {
        EXPECT_EQ(col.size(), kNDates);
    }

    for (std::size_t d = 0; d < kNDates; ++d) {
        EXPECT_DOUBLE_EQ(results[0][d], 1.0);  // smallest log(volume)
        EXPECT_DOUBLE_EQ(results[1][d], 2.0);
        EXPECT_DOUBLE_EQ(results[2][d], 3.0);  // largest log(volume)
    }
}

TEST_F(CsExprEvalTest, EvaluateCSWithExprNode) {
    // Build expression tree: log(volume) manually
    auto innerExpr = std::make_unique<UnaryExpr>(
        UnaryOpCode::LOG,
        std::make_unique<ColumnRef>(Field::VOLUME));

    auto results = calc_.evaluateCS(
        CsOpCode::CS_RANK, *innerExpr, *panel_);

    EXPECT_EQ(results.size(), kNStocks);
    for (auto& col : results) {
        EXPECT_EQ(col.size(), kNDates);
    }

    // Same expected ranks as above
    for (std::size_t d = 0; d < kNDates; ++d) {
        EXPECT_DOUBLE_EQ(results[0][d], 1.0);
        EXPECT_DOUBLE_EQ(results[1][d], 2.0);
        EXPECT_DOUBLE_EQ(results[2][d], 3.0);
    }
}

TEST_F(CsExprEvalTest, EvaluateCSExpressionZScore) {
    // cs_zscore(log(volume)): z-score should sum to ~0 per date
    auto results = calc_.evaluateCSExpression(
        "log(volume)", *panel_, CsOpCode::CS_ZSCORE);

    EXPECT_EQ(results.size(), kNStocks);

    for (std::size_t d = 0; d < kNDates; ++d) {
        double sum = 0.0;
        for (std::size_t a = 0; a < kNStocks; ++a) {
            sum += results[a][d];
        }
        EXPECT_NEAR(sum, 0.0, 1e-10);
    }
}

TEST_F(CsExprEvalTest, EmptyPanelReturnsEmpty) {
    PanelData emptyPanel;
    auto results = calc_.evaluateCSExpression(
        "log(volume)", emptyPanel, CsOpCode::CS_RANK);
    EXPECT_TRUE(results.empty());
}

TEST_F(CsExprEvalTest, EmptyPanelExprNodeReturnsEmpty) {
    PanelData emptyPanel;
    auto innerExpr = std::make_unique<ColumnRef>(Field::VOLUME);
    auto results = calc_.evaluateCS(
        CsOpCode::CS_RANK, *innerExpr, emptyPanel);
    EXPECT_TRUE(results.empty());
}

TEST_F(CsExprEvalTest, ExpressionCachingWorks) {
    // First call parses and caches
    EXPECT_EQ(calc_.cacheSize(), 0u);
    calc_.evaluateCSExpression("log(volume)", *panel_, CsOpCode::CS_RANK);
    EXPECT_EQ(calc_.cacheSize(), 1u);

    // Second call uses cache
    calc_.evaluateCSExpression("log(volume)", *panel_, CsOpCode::CS_RANK);
    EXPECT_EQ(calc_.cacheSize(), 1u);
}

// ============================================================
// IntermediateColumn integration test
// ============================================================

TEST_F(CsExprEvalTest, IntermediateColumnDirectUsage) {
    // Manually use IntermediateColumn + CrossSectionWorkspace
    // to replicate what evaluateCSImplExpr does internally.
    std::size_t M = panel_->assetCount();
    std::size_t T = panel_->dateCount();
    auto& pool = calc_.engine().pool();

    // Phase 1: compute log(volume) per-asset
    IntermediateColumn intermediate(M, T, pool);
    for (std::size_t a = 0; a < M; ++a) {
        const auto& md = panel_->asset(a);
        const auto& vol = md.column<double>(Field::VOLUME);
        double* out = intermediate.timeSeriesData(a);
        for (std::size_t d = 0; d < T; ++d) {
            out[d] = std::log(vol[d]);
        }
    }

    // Verify intermediate data
    for (std::size_t a = 0; a < M; ++a) {
        const auto& md = panel_->asset(a);
        const auto& vol = md.column<double>(Field::VOLUME);
        const double* data = intermediate.timeSeriesData(a);
        for (std::size_t d = 0; d < T; ++d) {
            EXPECT_DOUBLE_EQ(data[d], std::log(vol[d]));
        }
    }

    // Verify cross-sectional gather
    std::vector<double> gathered(M);
    for (std::size_t d = 0; d < T; ++d) {
        std::size_t valid = intermediate.gatherCrossSection(d, gathered.data(), M);
        EXPECT_EQ(valid, M);  // no nulls
        for (std::size_t a = 0; a < M; ++a) {
            const auto& vol = panel_->asset(a).column<double>(Field::VOLUME);
            EXPECT_DOUBLE_EQ(gathered[a], std::log(vol[d]));
        }
    }
}
