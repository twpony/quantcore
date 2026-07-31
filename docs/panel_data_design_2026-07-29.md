# PanelData 设计方案与现有架构融合

**日期**: 2026-07-29
**状态**: 设计草案
**已有基础**: [[panel_data_storage_design_2026-07-29]], [[tile_storage_cache_analysis_2026-07-29]]

---

## 一、现状分析：融合的关键障碍

### 1.1 已有基础设施

| 组件 | 能力 | 限制 |
|------|------|------|
| `MarketData` | 单股票 7 字段时间序列 | 只存一行（一只股票） |
| `Column<T>` / `ColView<T>` | 连续数组 + 零拷贝视图 | 一维 |
| `ExprNode::evaluate(MarketData&)` | 对单股票求值表达式 | 签名绑定 `MarketData` |
| `ExecutionEngine::evaluate(ExprNode&, MarketData&)` | 表达式求值 + 算子融合 + SIMD | 同上 |
| `CsExpr / RedExpr` | 对 `ColView<double>` 做截面变换/归约 | 已正确实现，但数据来源是单股票时间序列 |
| `CsOperator / RedOperator` | N→N 变换 或 N→scalar 归约 | 与数据来源无关，只需 `ColView<double>` |
| `MarketDataBundle` | **空壳 stub** | 只有 `assetCount() == 0`，无实现 |

### 1.2 当前 CsExpr/RedExpr 的实际行为

```cpp
// CsExpr::evaluate(MarketData& md, double* output, size_t n)
//   → child_->evaluate(md, ...)  → 得到 单股票时间序列
//   → invokeCs(op_, timeSeriesView, output, ...)
//   → 对时间序列做 rank/zscore/normalize

// 这其实是 TIME-SERIES rank，不是 CROSS-SECTIONAL rank：
CS_RANK(CLOSE)  // 当前行为：每天 close 在该股票全部历史中的排名（时序rank）
CS_RANK(CLOSE)  // 期望行为（截面）：某日期下所有股票的 close 排名
```

**核心矛盾**：`CsExpr` 和 `RedExpr` 的计算逻辑是通用的（对任意 `ColView<double>` 做统计变换），但数据来源目前只能是时间序列。要让它们支持真正的跨股票截面，只需要换数据来源——这正是 `PanelData` 要解决的问题。

### 1.3 结论：需要改动的地方极少

```
PanelData 只需要做一件事：为 CsExpr/RedExpr 提供截面向量作为数据来源。
其余一切（MarketData、ExecutionEngine、FusedLoopGenerator、UnaryExpr/BinaryExpr/RollingExpr）
完全不变。

改动清单：
  新增: PanelData 类（替换 MarketDataBundle 空壳）
  新增: PanelData::crossSection() 方法
  新增: ExecutionEngine 的截面求值重载
  改动: FactorCalculator 增加截面因子注册方法
  不改: MarketData, Column<T>, ColView<T>, 所有 ExprNode, 所有 Operator
```

---

## 二、PanelData 设计

### 2.1 设计原则

1. **不推翻，只补充**：`MarketData` + `Column<T>` 保持不变。`PanelData` 是上层容器。
2. **两种访问方向**：时序走 `asset(i)` → `MarketData`（现有管线）；截面走 `crossSection(date, field)` → `ColView<double>`（新管线）
3. **存储可选方案**：Phase 1 直接存 `vector<MarketData>`；Phase 2 内部切换为分块存储，API 不变
4. **时间轴对齐**：所有资产共享同一个 `TimestampIndex`（构建时校验）

### 2.2 核心接口

```cpp
// === 文件: quantcore/storage/PanelData.h ===

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/engine/BufferPool.h"
#include "quantcore/storage/ColView.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"
#include "quantcore/storage/TimestampIndex.h"

namespace quantcore {

class PanelData {
public:
    // ============================================================
    // Construction
    // ============================================================

    PanelData() = default;

    /// Construct from a vector of MarketData objects.
    /// All assets must have identical TimestampIndex (validated).
    /// @throws std::invalid_argument if timestamps are misaligned.
    explicit PanelData(std::vector<MarketData> assets);

    /// Add a single asset.  Timestamps must match existing assets.
    /// @throws std::invalid_argument if timestamps don't match.
    void addAsset(MarketData asset);

    /// Build from a vector.  Clears any existing data.
    void buildFrom(std::vector<MarketData> assets);

    // ============================================================
    // Dimensions
    // ============================================================

    /// Number of assets in the panel.
    std::size_t assetCount() const noexcept { return assets_.size(); }

    /// Number of time points (same for all assets).
    std::size_t dateCount() const noexcept { return timestamps_.size(); }

    /// Number of fields (always 7 for OHLCV).
    static constexpr std::size_t fieldCount() noexcept { return 7; }

    /// True if the panel is empty.
    bool empty() const noexcept { return assets_.empty(); }

    // ============================================================
    // Asset metadata
    // ============================================================

    const std::string& assetId(std::size_t i) const;
    std::size_t assetIndex(const std::string& id) const;
    const TimestampIndex& timestamps() const noexcept { return timestamps_; }

    /// Timestamp value at a date index.
    int64_t timestampAt(std::size_t dateIdx) const;

    /// Date (YYYYMMDD) at a date index.
    int64_t dateAt(std::size_t dateIdx) const;

    // ============================================================
    // Time-series access (时序方向)
    //
    // Returns a reference to the existing MarketData for one asset.
    // This is the entry point for time-series factor computation.
    // Zero-copy — no materialization needed.
    // ============================================================

    const MarketData& asset(std::size_t i) const;

    /// Find asset by ID.  @throws std::out_of_range if not found.
    const MarketData& assetById(const std::string& id) const;

    // ============================================================
    // Cross-sectional access (截面方向)
    //
    // Materializes all assets' values for a single (date, field)
    // into a contiguous buffer allocated from the given BufferPool.
    //
    // The returned ColView is valid until the buffer is returned to
    // the pool.  Typical usage:
    //
    //   auto view = panel.crossSection(dateIdx, Field::CLOSE, pool);
    //   // ... use view (compute rank, zscore, etc.) ...
    //   // BufferHandle auto-releases when leaving scope.
    //
    // @param dateIdx  Index into the common time axis [0, dateCount()).
    // @param field    Which field to extract.
    // @param pool     BufferPool for the temporary allocation.
    // @return A (BufferHandle, ColView<double>) pair.  The handle keeps
    //         the buffer alive; the view is a lightweight reference.
    // ============================================================

    struct CrossSection {
        BufferHandle<double> handle;  // Owns the materialized buffer
        ColView<double>      view;    // Lightweight reference into handle
    };

    CrossSection crossSection(std::size_t dateIdx,
                              Field field,
                              BufferPool& pool) const;

    /// Convenience: materialize cross-section and return just the ColView.
    /// The buffer must remain alive (stored in the returned handle).
    /// Prefer the CrossSection struct version for clarity.

    // ============================================================
    // Validation & diagnostics
    // ============================================================

    /// Verify all assets have aligned timestamps and non-empty fields.
    bool isValid() const;

    /// Number of null values at a specific (dateIdx, field) across
    /// all assets.
    std::size_t nullCount(std::size_t dateIdx, Field field) const;

private:
    /// Validate that `asset` has the same timestamp axis as the panel.
    void validateAsset(const MarketData& asset) const;

    std::vector<MarketData> assets_;
    std::unordered_map<std::string, std::size_t> assetIndex_;
    TimestampIndex timestamps_;
};

}  // namespace quantcore
```

### 2.3 关键方法实现要点

#### crossSection() — Phase 1 实现（直接 gather）

```cpp
PanelData::CrossSection PanelData::crossSection(
        std::size_t dateIdx, Field field, BufferPool& pool) const {

    std::size_t nAssets = assets_.size();

    // 从 BufferPool 分配连续缓冲区
    auto handle = pool.allocate<double>(nAssets);

    // Gather：从每只股票的 MarketData 取 dateIdx 位置的值
    for (std::size_t a = 0; a < nAssets; ++a) {
        const auto& col = assets_[a].column<double>(field);
        handle[a] = col[dateIdx];
    }

    // 构建 ColView（null mask 暂不处理）
    ColView<double> view(handle.data(), nAssets);

    return {std::move(handle), view};
}
```

性能特征（Phase 1）：
- 5000 只股票 × 每只 1 次跨 `Column<double>` 读取
- 每只股票的 `Column<double>` 是独立堆分配 → TLB miss + cache miss
- 延迟约 0.2-0.5ms（5000 只股票）
- 对于大多数场景可接受（截面因子占比 < 30% 时）

#### crossSection() — Phase 2 实现（分块存储）

内部存储从 `vector<MarketData>` 改为分块布局后，`crossSection()` 的实现变为：

```cpp
PanelData::CrossSection PanelData::crossSection(
        std::size_t dateIdx, Field field, BufferPool& pool) const {

    std::size_t nAssets = assets_.size();
    auto handle = pool.allocate<double>(nAssets);

    // 从 tiles 中 gather：每个 tile 提供 64 个连续 stock 的数据
    std::size_t nTileRows = (nAssets + kTileRows - 1) / kTileRows;
    std::size_t tileCol = dateIdx / kTileCols;
    std::size_t colInTile = dateIdx % kTileCols;

    for (std::size_t tr = 0; tr < nTileRows; ++tr) {
        const auto& tile = tiles_[tr * nTileCols_ + tileCol];
        std::size_t stockStart = tr * kTileRows;
        std::size_t stockEnd = std::min(stockStart + kTileRows, nAssets);

        // tile 内部：行优先，一行 64 个 dates
        // 第 s 行第 dateIdx 列: tile.data[s * kTileCols + colInTile]
        for (std::size_t s = stockStart; s < stockEnd; ++s) {
            std::size_t sLocal = s - stockStart;
            handle[s] = tile.data[sLocal * kTileCols + colInTile];
        }
    }

    ColView<double> view(handle.data(), nAssets);
    return {std::move(handle), view};
}
```

性能提升：
- tile 32KB 在 L1 cache 内 → 访问延迟 ~4 cycles
- 5000 只股票的 gather → 约 20,000 cycles ≈ 6μs（vs Phase 1 的 ~0.3ms）

---

## 三、与现存类型的融合方案

### 3.1 融合总览

```
                     ┌────────────────────────┐
                     │     FactorCalculator     │
                     │  registerTS()           │
                     │  registerCS()  ← NEW    │
                     │  evaluateTS()           │
                     │  evaluateCS()  ← NEW    │
                     └────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                                ▼
    ┌──────────────────┐            ┌────────────────────┐
    │ ExecutionEngine   │            │ ExecutionEngine     │
    │ .evaluate(expr,   │  ← 不变    │ .evaluateCS(expr,   │  ← NEW
    │  MarketData&)     │            │  PanelData&,        │
    │                   │            │  dateIdx, pool)     │
    └──────┬───────────┘            └────────┬───────────┘
           │                                 │
           ▼                                 ▼
    ┌──────────────┐              ┌──────────────────┐
    │  MarketData   │              │  PanelData        │
    │  (单股票)      │              │  .crossSection()  │
    │  ← 不修改     │              │  → ColView<double>│
    └──────────────┘              └──────────────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │  CsExpr / RedExpr  │
                                  │  (不修改)          │
                                  │  直接接收          │
                                  │  ColView<double>   │
                                  └──────────────────┘
```

### 3.2 MarketData — 不需要修改

`MarketData` 保持不变。它继续作为单股票时间序列的容器。时序因子的计算管道完全不碰 `PanelData`。

### 3.3 ExecutionEngine — 增加截面求值重载

```cpp
// === ExecutionEngine.h 新增 ===

class ExecutionEngine {
public:
    // ...existing API unchanged...

    /// Evaluate a cross-sectional expression for a single date.
    ///
    /// The expression tree is applied to the cross-sectional ColView
    /// obtained from panel.crossSection(dateIdx, field, pool).
    ///
    /// This is used for factors like "cs_rank(close)" — ranking stocks
    /// within a single date's cross-section.
    Column<double> evaluateCS(const ExprNode& expr,
                              PanelData& panel,
                              std::size_t dateIdx,
                              Field field);
};
```

实现逻辑：
```cpp
Column<double> ExecutionEngine::evaluateCS(
        const ExprNode& expr,
        PanelData& panel,
        std::size_t dateIdx,
        Field field) {

    std::size_t n = panel.assetCount();
    
    // 从 PanelData 获取截面向量（从 BufferPool 分配）
    auto cs = panel.crossSection(dateIdx, field, pool_);
    
    // 分配输出缓冲区
    auto resultHandle = pool_.allocate<double>(n);
    
    // 对截面向量求值 CsExpr/RedExpr
    // 注意：此处 expr 应该只包含 Cs/Red 操作（已经是融合边界，
    // 不能做 FusedLoop），直接用标准求值路径
    const uint64_t* nullMask = expr.evaluate(
        /* MarketData& — 需要适配 */,
        resultHandle.data(), n, &pool_);
    
    // 物化结果
    Column<double> result(n);
    std::memcpy(result.data(), resultHandle.data(), n * sizeof(double));
    return result;
}
```

**等等——这里有个问题**：`ExprNode::evaluate()` 的第一个参数是 `const MarketData&`。对于截面求值，我们没有 `MarketData`（或者说我们有的只是截面向量）。

这里有两条路：

**方案 A**：给 `ExprNode` 增加一个新的 `evaluate` 重载，接收 `ColView<double>`。

**方案 B**：在表达式构建时，让 CsExpr 的子节点固定为 `ColumnRef`（即要操作的字段），求值时 ColumnRef 不从 `MarketData` 读数据，而是从 PanelData 读。

**方案 C**：为截面因子创建专门的表达式节点 `CsFactorExpr`，它内部持有 `CsOpCode` 和 `Field`，不通过通用的 `ExprNode::evaluate(MarketData&)` 路径。

我认为**方案 A 最简单**：

```cpp
// === ExprNode.h 新增 ===

class ExprNode {
public:
    // ...existing...
    
    /// Evaluate this subtree against a raw ColView (cross-sectional path).
    /// For cross-sectional factors, the "row count" is the number of assets.
    /// ColumnRef nodes in this path resolve against the provided ColView
    /// instead of MarketData.
    virtual const uint64_t* evaluateCS(const double* input,
                                       double* output,
                                       std::size_t n,
                                       BufferPool* pool = nullptr) const {
        // Default: fall back to no-op (subclasses override)
        return nullptr;
    }
};
```

但这样需要修改 `ColumnRef` — 它需要知道何时从 `MarketData` 读（时序路径）vs 何时从 `ColView` 读（截面路径）。这会导致条件分支污染。

**更好的思路：不要让 CsExpr 的 child 是 ColumnRef。**

对于截面因子，表达式的叶子节点不应该引用 `MarketData` 的字段，而应该引用 `PanelData` 的截面。所以截面因子的表达式树是：

```
CS_RANK(CS_INPUT)          — CS_INPUT 是特殊的叶子节点，表示"截面向量"
而不是：
CS_RANK(COLUMN(CLOSE))     — 这会试图从 MarketData 读时间序列
```

**但更实际的方案是**：为截面因子创建一种**不同的求值路径**，完全不走 `ExprNode::evaluate(MarketData&)`：

```cpp
// === 新增: CrossSectionEvaluator ===

class CrossSectionEvaluator {
public:
    /// Evaluate a CS expression for all dates in the panel.
    /// Returns a Column<double> per asset, each of length dateCount().
    std::vector<Column<double>> evaluateAllDates(
        CsOpCode op,
        Field field,
        const PanelData& panel,
        BufferPool& pool,
        const std::vector<double>& extraParams = {}) const;
    
    /// Evaluate for a single date.
    void evaluateSingleDate(
        CsOpCode op,
        const double* input,    // crossSection(dateIdx, field)
        double* output,          // result for this date
        std::size_t nAssets,
        const std::vector<double>& extraParams = {}) const;
};
```

这样 `CsExpr` 和 `RedExpr` 保持不动（用于时序 rank/zscore），截面因子的求值与通用表达式树解耦。

**实际上，我意识到这里有一个更根本的问题需要澄清**。

让我重新审视现有的 `CsExpr`。它的文档说：
> "CsExpr applies a cross-section transform operator (RANK, ZSCORE, ...) to its child expression."

它设计成 Child 可以是任意表达式，然后对 Child 的输出做截面变换。但在单股票 MarketData 上下文中，Child 的输出是**时间序列**。所以它实际上是"对时间序列做统计变换"。

这两种用法**都合法且有用**：
1. 时序 rank：close 在过去 N 天中的排名 → 时序因子
2. 截面 rank：close 在所有股票中的排名 → 截面因子

所以 `CsExpr` 的代码不需要改——它只是"对输入数组做统计变换"。数据来源决定它是时序还是截面。

### 3.4 最终集成方案

对于**截面因子**的求值，采用以下路径：

```
PanelData::crossSection(date, field, pool)
    → ColView<double> (N_stocks 个值)
    → 直接传给 CsOperator::evaluate(ColView, output)
    → 结果写入 output (N_stocks 个值)
    → 拼接：每个日期的结果拼回每只股票的 Column<double>
```

**核心洞察**：截面因子的求值不需要走 `ExprNode` 树。`CsOperator` 已经可以直接接收 `ColView<double>`。截面求值就是：循环日期 → gather 截面向量 → 调用 CsOperator。

### 3.5 FactorCalculator — 增加截面因子支持

```cpp
// === FactorCalculator.h 新增 ===

class FactorCalculator {
public:
    // ...existing API unchanged...

    /// Register a cross-sectional factor.
    ///
    /// Cross-sectional factors are computed per-date across all assets,
    /// rather than per-asset across all dates.
    ///
    /// @param name        Factor name.
    /// @param op          CS operator code (CS_RANK, CS_ZSCORE, etc.).
    /// @param field       Which field to operate on (e.g. CLOSE, VOLUME).
    /// @param extraParams Optional operator parameters (e.g. quantile q).
    void registerCrossSectionalFactor(
        const std::string& name,
        CsOpCode op,
        Field field,
        const std::vector<double>& extraParams = {});

    /// Evaluate a cross-sectional factor on the full panel.
    ///
    /// Returns one Column<double> per asset, each containing the factor
    /// values at every date.
    ///
    /// @param name   Registered cross-sectional factor name.
    /// @param panel  The multi-asset panel.
    /// @return       Vector of Columns, one per asset, aligned with panel
    ///               asset ordering.
    std::vector<Column<double>> evaluateCS(
        const std::string& name,
        const PanelData& panel);
};
```

实现：
```cpp
std::vector<Column<double>> FactorCalculator::evaluateCS(
        const std::string& name,
        const PanelData& panel) {

    auto& reg = OperatorRegistry::instance();
    auto it = csFactors_.find(name);
    if (it == csFactors_.end()) throw std::runtime_error("Unknown CS factor: " + name);

    const auto& cfg = it->second;  // {op, field, extraParams}
    std::size_t nAssets = panel.assetCount();
    std::size_t nDates  = panel.dateCount();

    // 每只股票一个输出列，长度为日期数
    std::vector<Column<double>> results(nAssets);
    for (auto& col : results) col = Column<double>(nDates);

    // 为每个日期计算截面
    for (std::size_t d = 0; d < nDates; ++d) {
        // Gather 截面向量
        auto cs = panel.crossSection(d, cfg.field, engine_.pool());

        // 分配输出缓冲区
        auto outHandle = engine_.pool().allocate<double>(nAssets);

        // 直接调用 CsOperator（不经过 ExprNode）
        reg.invokeCs(cfg.op, cs.view, outHandle.data(), cfg.extraParams);

        // 将结果分散写回各股票的对应日期位置
        for (std::size_t a = 0; a < nAssets; ++a) {
            results[a][d] = outHandle[a];
        }
    }

    return results;
}
```

---

## 四、完整数据流

### 4.1 时序因子（不变）

```
for each stock in panel.assets():
    result[stock] = engine.evaluate(ast, panel.asset(stock))
                      │
                      ▼
              MarketData (单股票时间序列)
                      │
                      ▼
              ExprNode::evaluate(MarketData&, ...)
                      │
                      ▼
              Column<double> (时间序列结果)
```

### 4.2 截面因子（新增）

```
for each date in panel.timestamps():
    cs = panel.crossSection(date, field, pool)
         │
         ▼
    ColView<double> (N_stocks 个值：所有股票在该日期的 field 值)
         │
         ▼
    OperatorRegistry::invokeCs(op, cs, output, params)
         │
         ▼
    double[N_stocks] (每只股票一个值)
         │
         ▼
    分散写回: results[stock][date] = output[stock]
```

---

## 五、实施计划

### Phase 1：最小可行实现（本周可完成）

```
新增文件:
  quantcore/storage/PanelData.h        (~120 行)
  quantcore/storage/PanelData.cpp      (~80 行)

修改文件:
  quantcore/storage/MarketDataBundle.h → 删除或用 PanelData 替代
  quantcore/core/FactorCalculator.h    → 增加 registerCS() + evaluateCS()
  quantcore/core/FactorCalculator.cpp  → 增加截面求值循环

改动量: ~250 行新增代码，~50 行修改
```

**Phase 1 的核心逻辑非常短**：

```cpp
// PanelData::crossSection — 核心就这几行
for (size_t a = 0; a < assets_.size(); ++a) {
    handle[a] = assets_[a].column<double>(field)[dateIdx];
}

// FactorCalculator::evaluateCS — 核心也就这几行  
for (size_t d = 0; d < nDates; ++d) {
    auto cs = panel.crossSection(d, field, pool);
    reg.invokeCs(op, cs.view, outHandle.data(), params);
    for (size_t a = 0; a < nAssets; ++a)
        results[a][d] = outHandle[a];
}
```

### Phase 2：分块存储优化（性能提升 10x-50x）

触发条件：截面因子占比 > 20%，或单个因子求值超过 1 秒/日期。

```
新增文件:
  quantcore/storage/TiledColumn.h      (~200 行)  分块存储原语

修改文件:
  quantcore/storage/PanelData.h/.cpp   → 内部存储从 vector<MarketData> 
                                         改为 TiledColumn<double>[7]
  PanelData::crossSection()            → 从 tile gather
  PanelData::asset()                   → 从 tile materialize MarketData

API 不变，实现替换。
```

### Phase 3：多线程并行

```
for each date in parallel:
    cs = panel.crossSection(date, field, pool)
    reg.invokeCs(op, cs.view, outHandle.data(), params)
```

tile 之间无数据依赖，天然可并行。

---

## 六、不变清单（不需要改的部分）

| 文件 | 原因 |
|------|------|
| `MarketData` / `MarketDataView` | 单股票时间序列的完美容器，时序因子管线依赖它 |
| `Column<T>` / `ColView<T>` | 通用列式原语，两个方向都依赖 |
| `ExprNode` / `ColumnRef` / `Scalar` / `UnaryExpr` / `BinaryExpr` | 时序因子的表达式树，完全不变 |
| `RollingExpr` / `BinaryRollingExpr` | 纯粹时序操作，与截面无关 |
| `CsExpr` / `RedExpr` | 保留时序语义（时间序列上的 rank/zscore）不变 |
| `CsOperator` / `RedOperator` | 只接收 `ColView<double>`，与数据来源无关 |
| `OperatorRegistry` | 已有的 `invokeCs` / `invokeRed` 签名满足需求 |
| `FusedLoopGenerator` | 仅用于时序融合，截面无融合需求 |
| `BufferPool` / `BufferHandle` | 截面 gather 直接使用现有分配器 |
| `ExecutionEngine::evaluate()` | 时序求值不变 |
| `Lexer` / `Parser` | 表达式解析不变 |
| `alpha_0001..0005` | 现有 AOT 因子不变 |

---

## 七、与 MarketDataBundle 的关系

`MarketDataBundle` 当前是空壳。方案：

- **删除** `MarketDataBundle.h`，用 `PanelData.h` 替代
- 或者在 `MarketDataBundle.h` 的位置放入 `PanelData` 的实现，保持文件名向后兼容

推荐直接用 `PanelData` 替代 `MarketDataBundle`，清晰表达其职责。

---

## 八、使用示例

### 构建 PanelData

```cpp
// 从 I/O 层加载数据
std::vector<MarketData> assets;
for (const auto& stockId : stockList) {
    auto md = ParquetDailyReader::read("data/" + stockId + ".parquet");
    assets.push_back(std::move(md));
}

PanelData panel(std::move(assets));
// panel.assetCount() == 5000
// panel.dateCount()  == 2500
```

### 时序因子求值

```cpp
FactorCalculator calc;

// 时序因子：每只股票独立计算
calc.registerFormula("momentum", "close / rolling_mean(close, 60) - 1");

std::vector<Column<double>> momentum(panel.assetCount());
for (size_t a = 0; a < panel.assetCount(); ++a) {
    momentum[a] = calc.evaluate("momentum", panel.asset(a));
}
// momentum[a][d] = 股票a在日期d的动量值
```

### 截面因子求值

```cpp
// 截面因子：每个日期独立计算
calc.registerCrossSectionalFactor("rank_volume", CsOpCode::CS_RANK, Field::VOLUME);

auto rankVolume = calc.evaluateCS("rank_volume", panel);
// rankVolume[a][d] = 股票a在日期d的成交量全市场排名
```

### 组合：先算时序再截面

```cpp
// 1. 先算每只股票的时序因子（动量）
std::vector<Column<double>> momentum(panel.assetCount());
for (size_t a = 0; a < panel.assetCount(); ++a) {
    momentum[a] = calc.evaluate("momentum", panel.asset(a));
}

// 2. 将结果作为新字段注入（或构建临时 PanelData）
// 3. 对动量值做截面 rank
// → 这就是经典的 "截面动量因子"
```
