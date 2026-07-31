# Panel 数据存储与双方向因子计算设计

**日期**: 2026-07-29
**状态**: 设计讨论
**相关文档**: [[phase1_expression_ast_plan_2026-07-27]], [[phase2_engine_bufferpool_plan_2026-07-27]], [[tile_storage_cache_analysis_2026-07-29]]

---

## 一、问题背景

### 数据特征

量化因子的底层数据是典型的 Panel 结构（股票 × 日期矩阵）：

```
               D1       D2       D3      ...   D_N
Stock_A  VOL   100k      95k     120k    ...
Stock_A  OPEN   10.0     10.5      9.8   ...
...
Stock_B  VOL    50k      55k      48k    ...
...
```

每个字段（OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT, VWAP）是一个独立的 2D 矩阵：行为股票，列为日期。

### 两种因子方向

| 方向 | 含义 | 例子 | 访问模式 |
|------|------|------|---------|
| **时序因子** | 单只股票沿时间轴计算 | SMA, 动量, 波动率, MACD | 沿行（固定股票，遍历日期） |
| **截面因子** | 单个日期跨全市场计算 | rank, zscore, 行业中性化, 分位数 | 沿列（固定日期，遍历股票） |

同一组数据需要同时支持两种访问方向，这是存储设计的核心挑战。

---

## 二、当前架构分析

### 现有设计：MarketData = 单股票时间序列

```cpp
MarketData("000001.SZ") {
    timestamps: [D1, D2, D3, ...]
    OPEN:   [10.0, 10.5,  9.8, ...]   // Column<double>, 连续内存, 64B对齐
    CLOSE:  [12.0, 12.3, 12.1, ...]
    VOLUME: [100k,  95k, 120k, ...]
}
```

**优点**（针对时序因子）：
- `Column<T>` 是连续内存，cache 友好
- SIMD 向量化可直接处理 8 个连续日期
- `FusedLoopGenerator` 将多条表达式融合为单次遍历
- `BufferPool` 提供 64 字节对齐的零拷贝管理

**当前适配度**：

| 能力 | 适配度 | 说明 |
|------|--------|------|
| 时序因子 (SMA, EMA, momentum) | ✅ 完美 | `RollingExpr` + SIMD 极致优化 |
| 一元/二元表达式 (log return, ratio) | ✅ 完美 | `UnaryExpr`/`BinaryExpr` + 算子融合 |
| 截面因子 (rank, zscore, normalize) | ⚠️ 有缺口 | `CsExpr` 代码已存在，但求值路径仅接收单只 `MarketData` |
| 多资产管理 | ⚠️ 有缺口 | `MarketDataBundle` 为空壳 stub |

**关键问题**：`CsExpr::evaluate(MarketData& md)` 接收单只股票数据，但截面算子需要同一日期所有股票的值才能计算 rank/zscore。

---

## 三、pandas 的启示与局限

### pandas 的 API 设计

```python
# axis=0: 沿日期方向（每只股票独立）— 时序因子
df['close'].rolling(20).mean()           # SMA
df.pct_change()                           # 收益率

# axis=1: 沿股票方向（每个日期独立）— 截面因子
df.rank(axis=1)                           # 全市场排名
df.apply(lambda row: (row - row.mean()) / row.std(), axis=1)  # zscore
```

pandas 用一个 `axis` 参数**统一了 API**，调用方不需要关心底层存储方向。这是值得借鉴的。

### pandas 的性能不对称

pandas 底层按列存储（每列是连续 numpy array）：

- `axis=0`（列方向，时序）→ **快**：连续内存，SIMD 友好
- `axis=1`（行方向，截面）→ **慢**：跨 N 列的非连续内存 gather

在 Python 中，解释器开销远超内存跳转，所以这种不对称往往被掩盖。但在 C++ SIMD 向量化引擎中，**两种方向的性能差距可达 10x-100x**。

### 核心矛盾

```
时序优先布局（当前）：
  Stock_A CLOSE: [D1, D2, D3, D4, ...]   ← 连续，时序快
  Stock_B CLOSE: [D1, D2, D3, D4, ...]
  Stock_C CLOSE: [D1, D2, D3, D4, ...]

截面优先布局（转置）：
  Date_D1 CLOSE: [Stock_A, Stock_B, Stock_C, ...]  ← 连续，截面快
  Date_D2 CLOSE: [Stock_A, Stock_B, Stock_C, ...]

没有单一布局能同时优化两个方向。
```

---

## 四、C++ 的差异化优势：分块存储

C++ 可以做得比 pandas 更好：**通过分块（tile/chunk）策略在中间地带找到平衡**。

### 分块存储原理

```
Tile (4 stocks × 4 dates) 的 CLOSE 值:
     D1    D2    D3    D4
S_A  10.0  10.5   9.8  10.2
S_B   5.0   5.5   5.2   5.8
S_C  20.0  21.0  19.0  22.0
S_D   8.0   7.5   8.2   7.8

内存布局（行优先平铺）：
  [10.0, 10.5, 9.8, 10.2, 5.0, 5.5, ...]

时序遍历（同一行）：stride=1, 完全连续 → cache命中率最高
截面遍历（同一列）：stride=4, 但在 L1 cache 内 → 仍然很快
```

关键设计点：
- 一个 tile 小到可以装入 **L1 cache**（如 64×64 = 4096 doubles = 32KB）
- tile 内部两个方向都接近连续访问
- tile 之间可以并行处理（多线程各处理一个 tile）
- 这是 Apache Arrow、DuckDB 等现代分析引擎采用的标准策略

---

## 五、推荐架构：保持现有骨架 + 增加 PanelData 层

### 设计原则

1. **不推翻现有架构**：`MarketData` + `Column<T>` + SIMD 管线是时序方向的极致优化，保持不动
2. **在上层补充截面支持**：新增 `PanelData` 作为多资产容器
3. **两种求值路径**：时序走现有 pipeline，截面走新增的截面路径

### 架构图

```
                      API 层
         ┌────────────┼────────────┐
         │   时序因子              截面因子            │
         │   factor(stock)        factor(date)       │
         └────────────┼────────────┘
                      │
         ┌────────────┼────────────┐
         │       PanelData (tile-based)              │
         │   内部: 分块存储 stocks × dates           │
         └────────────┼────────────┘
                      │
         ┌────────────┴────────────┐
         │  ColView<T> (单块内连续) │
         │  - 时序切片: stride=1   │
         │  - 截面切片: stride=N   │
         └─────────────────────────┘
                      │
         ┌────────────┴────────────┐
         │     MarketData           │
         │   (单股票时间序列)        │
         │   SIMD + FusedLoop 管线  │
         └─────────────────────────┘
```

### 两种求值路径

```cpp
// === 时序因子求值（保留现有流程） ===
// for each stock:
//   result[stock] = engine.evaluate(ast, panelData.asset(i))

// === 截面因子求值（新增路径） ===
// for each date:
//   auto slice = panelData.crossSection(date, Field::CLOSE, pool);
//   // slice 是 ColView<double>, 包含所有股票在该日期的 close 值
//   CsExpr::evaluate(slice, output, ...)  // rank / zscore / normalize
```

### PanelData 核心接口（草案）

```cpp
class PanelData {
public:
    // --- 构造 ---
    // 从一组 MarketData 构建（要求时间轴对齐）
    static PanelData fromAssets(std::vector<MarketData> assets);

    // --- 资产访问（时序方向） ---
    size_t assetCount() const;
    const MarketData& asset(size_t i) const;
    const MarketData& assetById(const std::string& id) const;

    // --- 截面访问（截面方向） ---
    // 返回某日期下所有股票某个字段的视图
    // 内部可能从 BufferPool 物化（如果存储布局不匹配）
    ColView<double> crossSection(int64_t date, Field field, BufferPool& pool) const;

    // --- 公共时间轴 ---
    const TimestampIndex& commonTimestamps() const;

    // --- 分块存储（远期，如果截面因子占比 > 30%） ---
    // 内部将 stocks × dates 按 tile 组织
    // tileSize 取决于 L1 cache 大小（典型 64×64）
};
```

### 实施路线

| 阶段 | 内容 | 截面因子占比阈值 |
|------|------|-----------------|
| **当前** | `unordered_map<string, MarketData>` + 外部循环。时序因子直接用，截面因子临时 gather | < 10% |
| **Phase 2** | 实现 `PanelData` + `crossSection()` 方法（从多只 MarketData 按日期切片物化） | 10%-30% |
| **Phase 3** | 内部切换为分块存储（tile-based），两个方向都接近最优 | > 30% |
| **Phase 4** | 多线程并行：tile 间并行 + SIMD tile 内向量化 | 任意 |

---

## 六、总结

| 维度 | pandas | QuantCore 当前 | QuantCore 目标 |
|------|--------|---------------|---------------|
| API 统一性 | ✅ `axis` 参数 | ❌ 仅支持时序 | ✅ 时序 + 截面双路径 |
| 时序性能 | 中（Python 开销） | ✅ 高（SIMD + 融合） | ✅ 高（保持不变） |
| 截面性能 | 低（非连续 gather） | ❌ 不支持 | ✅ 高（分块存储） |
| 双向最优 | ❌ 无法做到 | N/A | ✅ tile-based |

核心结论：
- pandas 用 `axis` **统一了 API**，但没有统一底层性能（axis=1 就是慢）
- C++ 可以做到 API 统一 + 底层通过**分块存储**让两个方向都接近最优
- 现有 `MarketData` + `Column<T>` 是时序最优的叶子节点，**不需要改**
- 需要在上层增加 `PanelData` 分块容器来做截面方向的高效数据供给
- 不推翻现有架构，只是补充截面支持层
