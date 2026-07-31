# MarketData 与 PanelData 数据结构说明

**日期**: 2026-07-29
**状态**: 已实现
**相关文档**: [[panel_data_design_2026-07-29]], [[tile_storage_cache_analysis_2026-07-29]]

---

## 一、整体关系

```
PanelData  ←── 多资产面板容器
├── assets_[0] = MarketData("000001.SZ")  ←── 单股票时间序列
├── assets_[1] = MarketData("000002.SZ")
├── ...
└── assets_[N-1] = MarketData(...)
```

`MarketData` 是"一只股票 × 7 个字段 × 连续时间序列"的列式容器。`PanelData` 是 N 个 `MarketData` 的集合，不重复存储数据，只把它们管理在一起并提供截面切片能力。

---

## 二、MarketData — 单股票时间序列

### 结构

```cpp
MarketData("000001.SZ") {
    assetId_        = "000001.SZ"
    timestamps_     = TimestampIndex { dates: [20240101, 20240102, ...] }
    columns_[0..6]  = std::array<ColumnDataVariant, 7>   // 7 个字段
}

// ColumnDataVariant = std::variant<Column<double>, Column<int64_t>>
// 当前所有字段统一用 Column<double>
```

### 七个字段

```
columns_[OPEN]   → Column<double> { data: [10.0, 10.5,  9.8, ...] }  ← 连续 64 字节对齐
columns_[HIGH]   → Column<double> { data: [15.0, 15.3, 14.9, ...] }
columns_[LOW]    → Column<double> { data: [ 8.0,  8.2,  7.9, ...] }
columns_[CLOSE]  → Column<double> { data: [12.0, 12.3, 12.1, ...] }
columns_[VOLUME] → Column<double> { data: [100k,  95k, 120k, ...] }
columns_[AMOUNT] → Column<double> { data: [1.2M, 1.1M, 1.4M, ...] }
columns_[VWAP]   → Column<double> { data: [11.5, 11.8, 11.3, ...] }
```

每个 `Column<double>` 是一维连续数组，长度为日期数（如 2500），元素按时间先后排列。

### 关键特性

| 特性 | 说明 |
|------|------|
| **移构不可拷** | `MarketData(MarketData&&)` = default，`MarketData(const MarketData&)` = delete |
| **惰性 null bitmask** | `Column<T>` 仅在调用 `setNull()` 时才分配 bitmask，无 null 时零开销 |
| **零拷贝 mmap** | `Column::fromMmap(ptr, size, nullMask)` 直接包装外部内存 |
| **零拷贝切片** | `MarketData::slice(start, end) → MarketDataView`，返回 `ColView<T>` 引用 |
| **SIMD 友好** | `Column<T>` 内存 64 字节对齐（AVX-512 cache line），一次 load 8 个 double |

---

## 三、PanelData — 多资产面板

### 结构

```cpp
PanelData {
    assets_      = std::vector<MarketData>         // [5000 只股票]
    assetIndex_  = std::unordered_map<string→idx>  // "000001.SZ" → 0
    timestamps_  = TimestampIndex                  // 从首只股票复制，所有资产共用
}
```

### 内部存储布局（当前 Phase 1）

```
assets_[0] = MarketData("000001.SZ") { OPEN: [D0,D1,D2,...], CLOSE: [D0,D1,D2,...], ... }
assets_[1] = MarketData("000002.SZ") { OPEN: [D0,D1,D2,...], CLOSE: [D0,D1,D2,...], ... }
assets_[2] = MarketData("000003.SZ") { OPEN: [D0,D1,D2,...], CLOSE: [D0,D1,D2,...], ... }
...
assets_[4999] = MarketData("600000.SH") { ... }
```

每只股票的每个字段是一块独立的连续堆内存。不同股票的数据散布在地址空间各处。

### 约束

- 所有资产的 `TimestampIndex` 必须完全相同（构建时校验，不匹配则抛异常）
- 所有资产的行数必须相同（= 日期数）
- PanelData 移构不可拷（同 MarketData）

---

## 四、两种访问路径

### 4.1 时序访问（零拷贝，走现有管线）

```
panel.asset(0)
    → const MarketData&          直接返回 assets_[0] 的引用，零拷贝
    → engine.evaluate(ast, md)   执行表达式树
    → ExprNode::evaluate(MarketData&, ...)
    → Column<double> result      单股票的时间序列结果 (长度 = dateCount)
```

特点：
- **零拷贝**：直接返回已存在的 `MarketData` 引用
- **stride=1**：`Column<T>` 内数据连续，cache 友好
- **SIMD + 算子融合**：`FusedLoopGenerator` 将多条表达式融合为单次遍历
- **使用场景**：所有时序因子（SMA, EMA, 动量, 波动率, MACD 等）

### 4.2 截面访问（物化 gather，走新管线）

```
panel.crossSection(dateIdx=100, field=CLOSE, pool)
    │
    ├── 1. pool.allocate<double>(assetCount)    从 BufferPool 分配 40KB
    │
    ├── 2. for each asset in assets_:           遍历所有股票
    │       buf[a] = asset.column<double>(CLOSE)[100]   从每只股票的 Column 取第 100 个值
    │
    ├── 3. 返回 CrossSection {
    │         handle: BufferHandle<double>       持有 buffer 生命周期
    │         view:   ColView<double>(buf, 5000)  连续截面向量
    │       }
    │
    ├── 4. 调用方:
    │       OperatorRegistry::invokeCs(CS_RANK, cs.view, output, {})
    │       → 对 [S0_CLOSE_D100, S1_CLOSE_D100, ..., S4999_CLOSE_D100] 做 rank
    │
    └── 5. CrossSection 析构 → BufferHandle 归还 pool
```

特点：
- **物化一次**：gather 开销 O(assetCount)，5000 只股票约 0.2-0.5ms
- **返回连续 ColView**：物化后是连续内存，CsOperator 可直接 SIMD 向量化
- **BufferPool 管理**：临时 buffer 用完即还，不占用持久内存
- **使用场景**：所有截面因子（全市场 rank, zscore, normalize, 行业中性化等

### 4.3 两种路径对比

| | 时序 | 截面 |
|---|---|---|
| 入口 | `panel.asset(i)` | `panel.crossSection(d, field, pool)` |
| 返回类型 | `const MarketData&` | `CrossSection { BufferHandle, ColView }` |
| 数据量 | 1 只股票 × 2500 天 | 5000 只股票 × 1 天 |
| 拷贝 | 零拷贝 | 物化 gather（临时） |
| 内存连续性 | stride=1 完美 | gather 后连续 |
| 走 ExprNode 树 | ✅ | ❌（直接调 CsOperator） |
| 走 FusedLoop | ✅ | N/A（截面无融合需求） |

---

## 五、内存占用量化

以 A 股 5000 只股票 × 2500 个交易日为例：

| 组件 | 大小 |
|------|------|
| 数据本身（7 fields × 5000 × 2500 × 8B） | ~700MB |
| Column null-bitmask（无 null 时为零） | 0 |
| PanelData 索引（assetIndex_） | ~200KB |
| PanelData timestamps_ | ~40KB |
| 截面物化缓冲（临时，BufferPool 复用） | ~40KB/次 |
| **总计** | **~700MB** |

---

## 六、代码示例

### 构建

```cpp
// 从 I/O 加载
std::vector<MarketData> assets;
for (const auto& stockId : stockList)
    assets.push_back(loadFromParquet(stockId));

PanelData panel(std::move(assets));
// panel.assetCount() == 5000
// panel.dateCount()  == 2500
```

### 时序因子

```cpp
FactorCalculator calc;
calc.registerFormula("momentum", "close / rolling_mean(close, 60) - 1");

// 每只股票独立求值
for (size_t a = 0; a < panel.assetCount(); ++a)
    results[a] = calc.evaluate("momentum", panel.asset(a));
```

### 截面因子

```cpp
calc.registerCrossSectionalFactor(
    "rank_volume", CsOpCode::CS_RANK, Field::VOLUME);

// 每个日期独立求值，结果按股票组织
auto rankVol = calc.evaluateCS("rank_volume", panel);
// rankVol[a][d] = 股票 a 在日期 d 的成交量全市场排名
```
