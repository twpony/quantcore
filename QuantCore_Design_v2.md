# QuantCore 高性能 C++ 量化因子引擎设计规范（修订版 v2.0）

## 文档基本信息

| 项目 | 内容 |
|------|------|
| 文档名称 | QuantCore C++ High Performance Quant Factor Library 设计规范（修订版） |
| 适用场景 | 基于 K 线行情字段开发时序因子计算库，交由 AI 生成全套可编译 C++ 工程 |
| 当前阶段 | 一期基础版本（CPU SIMD，单线程串行调度） |
| 远期规划 | 多线程并行、GPU 后端、遗传规划因子生成 |
| 修订说明 | 基于 v1.0 评审意见全面修订，补充缺失设计、修正不合理约束、消解内部矛盾 |

---

## 一、项目整体目标

基于现代 C++ 构建**列式存储高性能量化因子计算引擎 QuantCore**，面向股票 K 线时序因子计算。

### 1.1 一期落地功能

| 模块 | 内容 |
|------|------|
| 数据存储层 | Column\<T\> 泛型列式容器、MarketData 多字段行情数据集（含时间轴与资产标识）、ColView\<T\> 零拷贝只读视图、空值 Mask 机制 |
| 算子层 | UnaryOperator 一元向量算子、BinaryOperator 二元算子（向量+标量 / 向量+向量） |
| 表达式系统 | 基于**表达式模板（Expression Templates）**的惰性求值链式 API，支持算子融合与单次遍历计算 |
| SIMD 向量化 | 分层策略：手写 intrinsics 路径（AVX2 / AVX-512）+ 编译器自动向量化 fallback + 运行时 CPU 特性派发 |
| 内存管理 | 分级 Slab Allocator 全局对齐缓冲池，RAII 句柄自动管理借用/归还 |
| 执行引擎 | 串行调度、BufferPool 管理、算子融合编译、列生命周期管理 |
| 数据 I/O | CSV 解析器 + 二进制列存格式（mmap 零拷贝加载）+ Python 互操作接口（pybind11 预留） |
| 可观测性 | 分级日志、算子级计时统计、内存池使用率监控 |

### 1.2 远期扩展规划

- 多线程并行调度（per-stock 数据并行 + per-operator 任务并行）
- DAG 表达式优化（公共子表达式消除、算子重排）
- CUDA / HIP / OpenCL GPU 算子后端
- RollingOperator 滚动窗口聚合算子（SMA / EMA / 滚动最大最小 / 标准差）
- CrossSectionOperator 横截面算子（多标的截面统计）
- ReductionOperator 归约聚合算子
- 字符串公式解析器（完整 AST → 表达式模板编译）
- 遗传规划因子生成模块

### 1.3 设计目标优先级

```
正确性 > 可复现性 > 性能 > 扩展性 > 代码简洁度
```

在金融计算领域，数值正确性和结果可复现性的优先级高于一切微优化。

---

## 二、顶层强制设计原则

### 2.1 核心原则（按优先级排序）

1. **正确性优先（Correctness First）**：任何优化不得以牺牲数值正确性为代价；SIMD 路径必须可对标量路径交叉验证
2. **面向数据设计（Data-Oriented Design）**：禁止逐根 K 线循环计算，全部采用批量向量运算
3. **列式存储 SoA 架构**：行情字段分离数组存储，禁用 AOS 单 K 线结构体数组
4. **惰性求值（Lazy Evaluation）**：表达式链式调用仅构建计算 DAG，调用 `.evaluate()` 时由执行引擎融合为单次遍历
5. **零拷贝（Zero Copy）**：ColView 视图访问列数据，中间结果通过 BufferPool 复用而非反复分配
6. **显式 SIMD（Explicit SIMD）**：优先手写 intrinsics 进行向量化，编译器自动向量化作为可移植 fallback
7. **分层可验证（Layered Verifiability）**：每一层提供标量参考实现，SIMD 优化路径必须通过交叉验证测试
8. **表达式驱动（Expression Oriented）**：支持多层嵌套复合数学表达式构建因子

### 2.2 原 v1.0 原则的修正说明

| 原原则 | 问题 | 修正 |
|--------|------|------|
| "零运行时开销" | 与后端抽象、算子注册机制矛盾 | 改为"关键路径零虚函数"，允许非关键路径存在可控开销 |
| "后端无关" | 过早抽象，缺乏 GPU 验证 | 一期不引入硬件后端抽象层；CPU SIMD 直接实现，待 GPU 需求明确后再提炼接口 |
| "SIMD 优先（仅编译器自动向量化）" | 策略过于脆弱 | 改为显式 SIMD intrinsics 优先 + 编译器向量化 fallback + 运行时派发 |

---

## 三、底层数据存储模型规范

### 3.1 字段类型体系

#### 3.1.1 行情字段枚举

```cpp
enum class Field : uint8_t {
    OPEN   = 0,   // 开盘价 — double
    HIGH   = 1,   // 最高价 — double
    LOW    = 2,   // 最低价 — double
    CLOSE  = 3,   // 收盘价 — double
    VOLUME = 4,   // 成交量 — int64_t（避免 double 精度损失）
    AMOUNT = 5,   // 成交额 — int64_t（避免 double 精度损失）
    VWAP   = 6,   // 均价   — double
    // === 远期可扩展 ===
    // TURNOVER, MARKET_CAP, LIMIT_UP, LIMIT_DOWN, ...
};
```

#### 3.1.2 字段类型映射

| 字段 | 存储类型 | 说明 |
|------|---------|------|
| OPEN / HIGH / LOW / CLOSE / VWAP | `double` | 价格类，天然浮点 |
| VOLUME / AMOUNT | `int64_t` | 数量/金额类，保持整数精度 |
| 计算中间结果 | `double` | 统一提升为 double |
| 信号/标记 | `bool` 或 `int8_t` | 远期扩展 |

**设计理由**：VOLUME 和 AMOUNT 在现实中是整数。double 只有 53 位有效精度，对于超大成交额会出现精度损失。使用 int64_t 存储原始值，仅在参与浮点运算时提升类型。

### 3.2 核心数据结构

#### 3.2.1 Column\<T\> — 泛型列式容器

```cpp
template <typename T>
class Column {
public:
    // 构造与初始化
    Column();                                          // 空列
    explicit Column(size_t size);                      // 定长分配（填充默认值）
    Column(const T* data, size_t size);                // 从原始指针拷贝构造
    static Column<T> fromMmap(const void* ptr, size_t size);  // mmap 零拷贝（只读）

    // 数据访问
    T&       operator[](size_t i);
    const T& operator[](size_t i) const;
    T*       data();
    const T* data() const;
    size_t   size() const;

    // 空值管理
    bool     hasNullMask() const;         // 是否存在空值
    void     setNull(size_t i);           // 标记位置 i 为空值
    bool     isNull(size_t i) const;      // 查询位置 i 是否为空

    // 内存信息
    size_t   capacity() const;
    bool     isAligned() const;           // 是否 64 字节对齐

private:
    std::vector<T, AlignedAllocator<T, 64>> data_;     // 64 字节对齐连续内存
    std::vector<uint64_t>                    nullMask_; // bitmask：1=空值，0=有效值（按需分配）
    bool                                     hasNull_ = false;
};
```

**关键设计决策**：
- `nullMask_` 采用惰性分配——仅当实际存在空值时才分配 bitmask 内存
- bitmask 使用 `uint64_t` 数组存储，便于 SIMD 批量检测
- 空值与 NaN 语义分离：NaN 是浮点计算的自然结果，null 表示数据源缺失
- `AlignedAllocator<T, 64>` 强制 64 字节对齐（适配 AVX-512）

#### 3.2.2 TimestampIndex — 时间轴索引

```cpp
class TimestampIndex {
public:
    // 从 Unix 时间戳数组构造
    explicit TimestampIndex(const int64_t* timestamps, size_t size);

    // 基本查询
    size_t  size() const;
    int64_t operator[](size_t i) const;          // 第 i 行的 Unix 时间戳

    // 日期感知查询
    bool    isConsecutive(size_t i) const;        // i 和 i-1 是否连续交易日
    int64_t dateAt(size_t i) const;               // 第 i 行的日期（YYYYMMDD）
    int64_t tradingDaysBetween(size_t from, size_t to) const;  // 间隔交易日数

    // 按日期范围切片
    std::pair<size_t, size_t> dateRange(int64_t startDate, int64_t endDate) const;

private:
    Column<int64_t> timestamps_;   // Unix 时间戳序列
    Column<int64_t> dates_;        // YYYYMMDD 格式日期序列（冗余，加速日期查询）
};

```

#### 3.2.3 MarketData — 单资产行情数据集

```cpp
class MarketData {
public:
    // === 构造 ===
    MarketData();
    MarketData(std::string assetId, TimestampIndex timestamps);

    // === 字段存取 ===
    template <typename T>
    Column<T>& column(Field field);

    template <typename T>
    const Column<T>& column(Field field) const;

    // === 资产信息 ===
    const std::string& assetId() const;
    const TimestampIndex& timestamps() const;

    // === 容量 ===
    size_t rowCount() const;
    bool   allColumnsAligned() const;    // 校验全部列长度一致

    // === 切片（生成零拷贝子视图 MarketDataView） ===
    MarketDataView slice(size_t start, size_t end) const;     // 按行下标切片
    MarketDataView sliceByDate(int64_t startDate, int64_t endDate) const;  // 按日期切片

    // === 字段管理（远期扩展接口）===
    void addField(Field field, ColumnVariant data);   // 动态添加字段

private:
    std::string assetId_;                              // 资产代码（如 "000001.SZ"）

    // 固定字段存储 — 使用 variant 支持不同类型
    std::array<ColumnDataVariant, kFieldCount> columns_;

    TimestampIndex timestamps_;                        // 时间轴
    bool           hasNullAnywhere_ = false;           // 快速判断是否有空值
};
```

**关键设计决策**：
- `Field` 到实际字段的映射使用 `std::array` + Field 枚举索引（零哈希开销），而非 `unordered_map`
- `assetId_` 作为字符串存储，支持各种交易所的代码格式
- `TimestampIndex` 是 MarketData 的必备组件，不存在没有时间轴的数据集
- `slice()` 返回 `MarketDataView`（零拷贝），而非拷贝新的 MarketData

#### 3.2.4 MarketDataView — 零拷贝子数据集视图

```cpp
class MarketDataView {
public:
    size_t rowCount() const;

    template <typename T>
    ColView<T> column(Field field) const;     // 返回零拷贝列切片

    const std::string& assetId() const;
    TimestampIndexView timestamps() const;     // 时间轴的零拷贝视图

private:
    std::string          assetId_;             // 拷贝（字符串短，可接受）
    std::array<ColVariant, kFieldCount> colViews_;  // 不持有数据所有权
    TimestampIndexView   timestampView_;
    size_t               rowCount_;
};
```

**生命周期规则**：
- `MarketDataView` 不持有数据所有权
- 底层 `MarketData` 必须比所有其派生的 `MarketDataView` 活得更久
- 违反时通过 assert 检测（Debug 模式）或产生未定义行为（Release 模式，类似 `std::string_view` 语义）

#### 3.2.5 ColView\<T\> — 零拷贝列视图

```cpp
template <typename T>
class ColView {
public:
    ColView();                                                   // 空视图
    ColView(const T* data, size_t start, size_t end);           // 引用外部数据
    ColView(const Column<T>& col);                               // 引用整个列
    ColView(const Column<T>& col, size_t start, size_t end);    // 引用列的子区间

    // 禁止拷贝赋值（只读视图，不参与所有权管理）
    ColView(const ColView&) = default;
    ColView& operator=(const ColView&) = delete;

    const T& operator[](size_t i) const;
    const T* data() const;
    size_t   size() const;
    bool     empty() const;
    ColView  subView(size_t start, size_t end) const;           // 嵌套切片

    // 空值查询（若底层 Column 有 nullMask）
    bool     hasNullMask() const;
    bool     isNull(size_t i) const;

private:
    const T*        data_ = nullptr;
    const uint64_t* nullMask_ = nullptr;
    size_t          start_ = 0;
    size_t          end_   = 0;
};
```

#### 3.2.6 MarketDataBundle — 多资产面板数据（远期）

```cpp
// 一期预留接口，不实现具体逻辑
class MarketDataBundle {
    // 多只股票的对齐面板数据
    // std::vector<MarketData> assets_;
    // 保证所有资产的 timestamp 对齐或者建立映射关系
};
```

### 3.3 数据模型层次总图

```
Column<T>                    — 单列连续内存（含空值 bitmask）
    │
    ├── TimestampIndex       — 时间轴索引（日期感知）
    │
    └── MarketData           — 单资产多字段数据集（assetId + TimestampIndex + 多 Column）
           │
           ├── MarketDataView  — 零拷贝切片视图
           │
           └── MarketDataBundle (远期) — 多资产对齐面板
```

---

## 四、表达式系统设计

### 4.1 核心策略：表达式模板（Expression Templates）

**设计目标**：一期即实现惰性求值 + 算子融合，消除链式 API 与零拷贝目标之间的根本矛盾。

**基本原理**：
- 每个算子调用不执行计算，而是返回一个轻量级**表达式节点对象**（编译期类型）
- 链式拼接构成**表达式树**（类型即为树的结构编码）
- 调用 `.evaluate()` 时，执行引擎展开表达式树为**单次循环**，逐元素完成全部嵌套运算
- 零中间内存分配——最终结果直接写入目标 Column

### 4.2 表达式节点类型

```cpp
// === 表达式节点基类（CRTP 静态多态，无虚函数） ===
template <typename Derived>
class ExprNode {
public:
    // 求值：对 [0, size) 范围内的每个 i 计算表达式值
    double evaluateAt(size_t i) const {
        return static_cast<const Derived&>(*this).evaluateAtImpl(i);
    }

    size_t size() const {
        return static_cast<const Derived&>(*this).sizeImpl();
    }

    // 类型编码（编译期常量，用于表达式模式匹配与优化）
    static constexpr ExprType typeCode = Derived::kTypeCode;
};

// === 叶节点 ===
class ColumnRef : public ExprNode<ColumnRef> {
    // 零拷贝引用某个 ColView<double>
    static constexpr ExprType kTypeCode = ExprType::COLUMN_REF;
};

class Scalar : public ExprNode<Scalar> {
    // 编译期/运行期标量常量
    static constexpr ExprType kTypeCode = ExprType::SCALAR;
};

// === 一元运算节点 ===
template <UnaryOp Op, typename Child>
class UnaryExpr : public ExprNode<UnaryExpr<Op, Child>> {
    // evaluateAt(i) = Op(Child.evaluateAt(i))
};

// === 二元运算节点 ===
template <BinaryOp Op, typename Lhs, typename Rhs>
class BinaryExpr : public ExprNode<BinaryExpr<Op, Lhs, Rhs>> {
    // evaluateAt(i) = Op(Lhs.evaluateAt(i), Rhs.evaluateAt(i))
};
```

### 4.3 链式 API（用户接口）

```cpp
// 用户编写代码示例：
auto close   = marketData.column<double>(Field::CLOSE);
auto vwap    = marketData.column<double>(Field::VWAP);
auto volume  = marketData.column<int64_t>(Field::VOLUME);

// 链式构建表达式 — 只构建树，不执行计算
auto factorExpr = abs(log(close) - log(vwap)) * volume.toDouble();

// 求值 — 此时执行引擎融合为单次循环
Column<double> result = engine.evaluate(factorExpr);
// 等价于单次遍历：
//   for i in 0..N:
//     result[i] = abs(log(close[i]) - log(vwap[i])) * double(volume[i])
```

### 4.4 算子融合编译流程

```
表达式树 (ExprNode DAG)
      │
      ▼
  FusedLoopGenerator（编译期/运行期代码生成）
      │
      ├── Step 1: 拓扑排序确定计算顺序
      ├── Step 2: 公共子表达式消除 (CSE)
      ├── Step 3: 生成融合循环体
      │     for i in 0..N:
      │       tmp = log(close[i]) - log(vwap[i])
      │       result[i] = abs(tmp) * double(volume[i])
      └── Step 4: 选择最优 SIMD 内核执行
```

### 4.5 分阶段实现

| 阶段 | 内容 |
|------|------|
| 一期 | 表达式模板链式 API + 算子融合求值 + 公共子表达式消除 + 标量循环 fuse 执行 + SIMD fuse 执行 |
| 远期 | 字符串公式解析器（"ABS(LOG(CLOSE)-LOG(VWAP))*VOLUME" → ExprNode）、更复杂的 DAG 优化器 |

**一期已经具备算子融合能力**，远期只需要换"输入端"（字符串解析 → 同一种 ExprNode），不需要重写表达式和执行引擎。

---

## 五、算子分层体系

### 5.1 算子分类

```
                    ┌─────────────────┐
                    │   OperatorBase  │  (CRTP 静态接口，无虚函数)
                    │   - evaluate()  │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┬──────────────────┐
          │                  │                  │                  │
   UnaryOperator      BinaryOperator     RollingOperator    CrossSectionOperator
   (一期实现)          (一期实现)          (远期)              (远期)
   - Log             - Add(Col,Col)     - SMA/EMA
   - Abs             - Sub(Col,Col)     - RollingMax
   - Sqrt            - Mul(Col,Scalar)  - RollingStd
   - Neg             - Div(Col,Scalar)  - RollingRank
   - Diff            - Add(Col,Scalar)  - ...
   - Shift           - ...
   - Sign
   - ...
```

### 5.2 一期算子清单

#### UnaryOperator（一元向量算子）

| 算子 | 数学形式 | 空值处理 |
|------|---------|---------|
| Abs | `abs(x)` | 传播空值 |
| Log | `ln(x)` | 需检查 x>0，否则 NaN + 日志警告 |
| Log10 | `log10(x)` | 同上 |
| Sqrt | `sqrt(x)` | 需检查 x>=0 |
| Neg | `-x` | 传播空值 |
| Diff | `x[i] - x[i-1]` | 首元素标记空值 |
| Shift | `x[i - offset]` | 越界位置标记空值 |
| Sign | `sign(x)` | 传播空值 |
| Square | `x²` | 传播空值 |
| Exp | `eˣ` | 传播空值 |

#### BinaryOperator（二元算子）

| 算子 | 数学形式 | 操作数类型 |
|------|---------|-----------|
| Add | `a + b` | Col+Col / Col+Scalar |
| Sub | `a - b` | Col+Col / Col+Scalar |
| Mul | `a × b` | Col+Col / Col+Scalar |
| Div | `a / b` | Col+Col / Col+Scalar（需检查除零） |
| Pow | `aᵇ` | Col+Scalar |
| Max | `max(a,b)` | Col+Col / Col+Scalar |
| Min | `min(a,b)` | Col+Col / Col+Scalar |
| Gt / Lt / Eq / Neq | 比较 | Col+Col / Col+Scalar，返回 bool 列 |

### 5.3 算子设计原则

1. **纯逻辑无内存**：算子类自身不持有内存，只描述运算逻辑
2. **输入为 ColView**：算子通过 ColView 引用输入数据，不拥有所有权
3. **输出由引擎注入**：计算结果写入引擎提供的目标缓冲区（来自 BufferPool 或最终输出 Column）
4. **标量参考实现必须存在**：每个算子提供一个 `evaluateScalar(T input) -> T` 方法，作为 SIMD 实现的对照基准

---

## 六、SIMD 向量化分层策略

### 6.1 分层架构

```
┌──────────────────────────────────────┐
│        ExecutionEngine (融合循环)      │
│  ┌──────────────────────────────────┐│
│  │  SIMD Dispatcher (运行时派发)      ││
│  │  ┌────────────┐ ┌─────────────┐  ││
│  │  │ AVX-512    │ │ AVX2        │  ││
│  │  │ intrinsics │ │ intrinsics  │  ││
│  │  └────────────┘ └─────────────┘  ││
│  │  ┌─────────────────────────────┐  ││
│  │  │ Compiler Auto-Vectorization│  ││
│  │  │ (#pragma omp simd fallback)│  ││
│  │  └─────────────────────────────┘  ││
│  │  ┌─────────────────────────────┐  ││
│  │  │ Scalar Loop (标量兜底)      │  ││
│  │  └─────────────────────────────┘  ││
│  └──────────────────────────────────┘│
└──────────────────────────────────────┘
```

### 6.2 指令集支持矩阵

| 层级 | 指令集 | 寄存器宽度 | 每次处理 double 数 | 实现方式 |
|------|--------|-----------|-------------------|---------|
| L3 | AVX-512F | 512 bit | 8 | `_mm512_*` intrinsics |
| L2 | AVX2 + FMA | 256 bit | 4 | `_mm256_*` intrinsics |
| L1 | SSE4.2 | 128 bit | 2 | `_mm_*` intrinsics |
| L0 | Scalar | N/A | 1 | 纯 C++ 标量循环 |

### 6.3 运行时派发

```cpp
// 启动时检测 CPU 能力，设置全局函数指针表
enum class SimdLevel { SCALAR, SSE42, AVX2, AVX512 };

class SimdDispatcher {
public:
    static SimdLevel detectCPU();   // CPUID 检测
    static SimdLevel bestLevel();   // 当前最优级别

    template <typename Op>
    static auto dispatch(Op&& op) -> /* 对应的 SIMD 内核 */;
};
```

### 6.4 SIMD 编码规范

1. **每条 SIMD 路径提供对标量路径的交叉验证测试**
2. intrinsics 代码使用 `#ifdef __AVX2__` / `#ifdef __AVX512F__` 条件编译
3. 循环尾部（不足一个向量宽度的元素）统一用标量循环处理
4. 所有计算缓冲区强制 64 字节对齐（`alignas(64)`）
5. 指针参数添加 `__restrict__` 关键字消除内存别名
6. 编译器自动向量化作为可移植性 fallback，不作为唯一 SIMD 策略

### 6.5 融合循环的 SIMD 执行

区别于传统逐算子执行模式，融合循环在单个 pass 内完成所有嵌套运算：

```
// 传统模式（多次循环，多次内存读写）
tmp1[i] = log(close[i])       // 循环 1
tmp2[i] = log(vwap[i])        // 循环 2
tmp3[i] = tmp1[i] - tmp2[i]   // 循环 3
tmp4[i] = abs(tmp3[i])        // 循环 4
result[i] = tmp4[i] * vol[i]  // 循环 5

// 融合模式（单次循环，零中间内存）
for i in 0..N:                 // 融合为 1 次循环
  result[i] = abs(log(close[i]) - log(vwap[i])) * double(volume[i])
```

---

## 七、ExecutionEngine 执行引擎

### 7.1 职责定义

```
┌─────────────────────────────────────────────────┐
│              ExecutionEngine                     │
│                                                  │
│  ┌─────────────┐  ┌──────────────┐              │
│  │ FusedLoop   │  │  BufferPool  │              │
│  │ Generator   │  │  Manager     │              │
│  │             │  │              │              │
│  │ 表达式树 →  │  │ 中间缓冲分配/│              │
│  │ 融合循环    │  │ 归还         │              │
│  └──────┬──────┘  └──────┬───────┘              │
│         │                │                      │
│         ▼                ▼                      │
│  ┌─────────────────────────────────────────┐    │
│  │           SIMD Dispatcher                │    │
│  │  运行时选择最优 SIMD 内核执行融合循环     │    │
│  └─────────────────────────────────────────┘    │
│                                                  │
│  ┌─────────────┐  ┌──────────────┐              │
│  │  Lifecycle  │  │   Metrics    │              │
│  │  Manager    │  │   Collector  │              │
│  │             │  │              │              │
│  │ Column/View │  │ 算子耗时统计 │              │
│  │ 生命周期管理│  │ 内存峰值追踪 │              │
│  └─────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────┘
```

### 7.2 核心接口

```cpp
class ExecutionEngine {
public:
    explicit ExecutionEngine(const EngineConfig& config);

    // === 核心求值接口 ===
    template <typename ExprType>
    Column<double> evaluate(const ExprNode<ExprType>& expr);

    template <typename ExprType>
    void evaluateInto(const ExprNode<ExprType>& expr, Column<double>& output);

    // === 表达式融合 + SIMD 执行 ===
    template <typename ExprType>
    Column<double> evaluateFused(const ExprNode<ExprType>& expr);

    // === 批量求值（多个因子共享输入，合并执行）===
    std::vector<Column<double>> evaluateBatch(
        const std::vector<ExprHandle>& expressions);

    // === BufferPool 管理 ===
    BufferPool& bufferPool();
    BufferHandle allocateBuffer(size_t elementCount);
    void        releaseBuffer(BufferHandle&& handle);

    // === 性能监控 ===
    const EngineMetrics& metrics() const;
    void                 resetMetrics();

private:
    BufferPool        bufferPool_;
    SimdDispatcher    simdDispatcher_;
    EngineMetrics     metrics_;
    EngineConfig      config_;
};
```

### 7.3 一期执行流程

```
1. 用户构建表达式树（链式 API）
       │
2. ExecutionEngine::evaluate(expr)
       │
3. 验证所有输入 ColView 长度一致性
       │
4. FusedLoopGenerator 展开表达式树
       │
5. SimdDispatcher 选择最优 SIMD 内核
       │
6. 从 BufferPool 获取输出缓冲区
       │
7. 执行融合向量化循环
       │
8. 返回 Column<double> 结果（所有权转移给调用方）
```

### 7.4 远期并行化预留

```cpp
// 一期串行实现，远期并行化时仅需替换 evaluateBatch 内部实现
// 接口设计已兼容 per-stock 数据并行 和 per-expression 任务并行
```

---

## 八、BufferPool 全局对齐缓冲池

### 8.1 设计策略：分级 Slab Allocator

```
BufferPool
  │
  ├── Slab 64B   (1 << 6)    — 8 个 double / 4 个 int64_t
  ├── Slab 256B  (1 << 8)    — 32 个 double
  ├── Slab 1KB   (1 << 10)   — 128 个 double（常用中间结果）
  ├── Slab 4KB   (1 << 12)   — 512 个 double
  ├── Slab 16KB  (1 << 14)   — 2048 个 double
  ├── Slab 64KB  (1 << 16)   — 8192 个 double
  ├── Slab 256KB (1 << 18)   — 32768 个 double（大列）
  └── Slab 1MB   (1 << 20)   — 131072 个 double（超大列）
```

### 8.2 核心接口

```cpp
class BufferPool {
public:
    struct Config {
        size_t totalMemoryLimitMB = 256;   // 内存上限
        size_t preallocPerSlab   = 4;      // 每级 slab 预分配块数
        bool   enableStats       = true;    // 启用统计
    };

    explicit BufferPool(const Config& config);

    // 分配与归还
    BufferHandle acquire(size_t elementCount);     // RAII 句柄
    void         release(BufferHandle&& handle);   // 归还到池中

    // 统计信息
    struct Stats {
        size_t totalAllocated;       // 已分配总字节数
        size_t peakAllocated;        // 峰值分配
        size_t hitCount;             // 缓存命中次数
        size_t missCount;            // 缓存未命中（需新分配）
        size_t activeHandles;        // 当前活跃句柄数
    };
    Stats stats() const;
    void  resetStats();

    // 预热：预分配指定级别的 slab
    void warmup();

private:
    // 每级 slab 一个 freelist
    std::array<SlabFreelist, kSlabLevels> slabs_;
    Config config_;
    Stats  stats_;
};
```

### 8.3 RAII 句柄

```cpp
class BufferHandle {
public:
    BufferHandle();                    // 空句柄
    BufferHandle(BufferPool* pool, double* ptr, size_t size, size_t slabLevel);
    ~BufferHandle();                   // 析构自动归还

    // 禁止拷贝，允许移动
    BufferHandle(const BufferHandle&) = delete;
    BufferHandle& operator=(const BufferHandle&) = delete;
    BufferHandle(BufferHandle&& other) noexcept;
    BufferHandle& operator=(BufferHandle&& other) noexcept;

    double*       data();
    const double* data() const;
    size_t        size() const;
    bool          valid() const;

private:
    BufferPool* pool_ = nullptr;
    double*     ptr_  = nullptr;
    size_t      size_ = 0;
    size_t      slabLevel_ = 0;
};
```

### 8.4 内存对齐保证

- 所有 slab 中的内存块均 **64 字节对齐**
- 使用 `AlignedAllocator<double, 64>` 底层分配
- 对齐要求同时满足 AVX-512 (64B) / AVX2 (32B) / SSE4.2 (16B)

### 8.5 线程安全预留（一期单线程，远期扩展）

```cpp
// 一期：无锁实现
// 远期：每线程一个 BufferPool 实例（thread_local）+
//       全局大块分配锁（std::mutex 或无锁队列）
// 接口本身不暴露线程语义，远期内部替换即可
```

---

## 九、硬件后端架构（一期简化）

### 9.1 设计决策：一期不引入硬件后端抽象层

原 v1.0 设计中的"后端无关"抽象层在本版中**取消**。理由：

1. 在没有 GPU 实现经验的情况下设计的抽象接口，几乎必然是错误的
2. 虚函数调用开销与"高性能"目标矛盾
3. 抽象本身增加代码复杂度，延缓一期交付

**替代方案**：一期直接在 ExecutionEngine 内部实现 CPU SIMD 路径。当二期需要接入 GPU 时，基于实际的 CPU SIMD 实现经验提炼后端接口。这符合"从具体到抽象"的工程规律。

### 9.2 远期 GPU 后端的接入点

以下位置预留扩展点（编译期 `#ifdef` 或运行期策略模式），但一期不创建抽象基类：

```
ExecutionEngine::evaluateFused()
  └── FusedLoopExecutor (具体类，一期仅有 CpuSimdExecutor)
        ├── CpuSimdExecutor   — 一期完整实现
        ├── CudaExecutor      — 远期实现
        └── OpenCLExecutor    — 远期实现
```

---

## 十、OperatorRegistry 算子注册中心

### 10.1 一期设计

```cpp
class OperatorRegistry {
public:
    static OperatorRegistry& instance();

    // 注册算子（启动时自动注册）
    template <typename OpType>
    void registerUnary(std::string name, UnaryOpCode code);

    template <typename OpType>
    void registerBinary(std::string name, BinaryOpCode code);

    // 查询算子
    UnaryOpCode  findUnary(const std::string& name) const;
    BinaryOpCode findBinary(const std::string& name) const;

    // 列出所有已注册算子
    std::vector<std::string> listUnary() const;
    std::vector<std::string> listBinary() const;

private:
    std::unordered_map<std::string, UnaryOpCode>  unaryRegistry_;
    std::unordered_map<std::string, BinaryOpCode> binaryRegistry_;
};
```

### 10.2 远期扩展接口（一期仅预留声明）

```cpp
// 远期：公式字符串解析时从注册中心查找算子
// std::unique_ptr<ExprNodeBase> parseFormula(const std::string& formula);

// 远期：遗传规划因子生成时枚举可用算子
// std::vector<UnaryOpCode> availableUnaryOps() const;
// std::vector<BinaryOpCode> availableBinaryOps() const;
```

---

## 十一、数据 I/O 层

### 11.1 CSV 解析器

```cpp
class CsvMarketDataReader {
public:
    struct Config {
        char    delimiter    = ',';
        bool    hasHeader    = true;
        int64_t dateColumn   = 0;    // 日期列索引（0-based）
        int64_t openColumn   = 1;
        int64_t highColumn   = 2;
        int64_t lowColumn    = 3;
        int64_t closeColumn  = 4;
        int64_t volumeColumn = 5;
        int64_t amountColumn = 6;
        // VWAP 可选：若不存在则从 AMOUNT/VOLUME 计算
        int64_t vwapColumn   = -1;   // -1 表示不存在
        int64_t skipRows     = 0;    // 跳过前 N 行
        int64_t maxRows      = -1;   // 最多读取行数（-1 为全部）
        std::string dateFormat = "%Y-%m-%d";  // 日期解析格式
    };

    explicit CsvMarketDataReader(const Config& config);

    // 从文件读取单只股票数据
    MarketData readFile(const std::string& filepath,
                        const std::string& assetId);

    // 从文件批量读取多只股票（每文件一只股票）
    std::vector<MarketData> readDirectory(const std::string& dirPath);

    // 读取过程中的错误统计
    struct ReadStats {
        size_t totalRows;
        size_t skippedRows;    // 因解析错误跳过的行
        size_t nullCells;      // 空单元格数量
        std::vector<std::string> warnings;
    };
    ReadStats lastReadStats() const;

private:
    Config config_;
    ReadStats lastStats_;
};
```

### 11.2 二进制列存格式（Zero-Copy Load）

```cpp
class BinaryMarketDataWriter {
public:
    // 将 MarketData 写入二进制文件（可直接 mmap 读取）
    static void write(const MarketData& data, const std::string& filepath);

    // 格式布局：
    // ┌──────────────────────────────────────┐
    // │ FileHeader (magic, version, fields)  │
    // ├──────────────────────────────────────┤
    // │ TimestampIndex (连续 int64_t 数组)   │
    // ├──────────────────────────────────────┤
    // │ OPEN    column (连续 double 数组)    │
    // ├──────────────────────────────────────┤
    // │ HIGH    column (连续 double 数组)    │
    // ├──────────────────────────────────────┤
    // │ LOW     column (连续 double 数组)    │
    // ├──────────────────────────────────────┤
    // │ CLOSE   column (连续 double 数组)    │
    // ├──────────────────────────────────────┤
    // │ VOLUME  column (连续 int64_t 数组)   │
    // ├──────────────────────────────────────┤
    // │ AMOUNT  column (连续 int64_t 数组)   │
    // ├──────────────────────────────────────┤
    // │ VWAP    column (连续 double 数组)    │
    // ├──────────────────────────────────────┤
    // │ NullMask 区域 (按需，每字段一组)     │
    // └──────────────────────────────────────┘
};

class BinaryMarketDataReader {
public:
    // mmap 文件到内存，返回零拷贝 MarketData
    // 注意：返回的 MarketData 内部 ColView 指向 mmap 区域
    //       调用方负责保证 MmapHandle 生命周期
    static std::pair<MarketData, MmapHandle>
    readFile(const std::string& filepath);
};
```

**设计理由**：二进制格式的内存布局与 `Column<T>` 的内存布局完全一致。通过 mmap 可以直接将文件内容映射为进程地址空间，然后在原地构造 ColView ——实现真正的"零拷贝加载"。这对生产环境加载大规模历史数据至关重要。

### 11.3 Python 互操作接口（预留）

```cpp
// 一期使用 pybind11 提供最小可用的 Python 绑定
// 远期可考虑纯 C 接口（ABI 稳定）或 Apache Arrow 兼容格式

// 最小绑定范围：
// - MarketData 构造与字段访问
// - ExecutionEngine::evaluate()
// - 基本算子（abs, log, +, -, *, /）
```

---

## 十二、错误处理策略

### 12.1 错误分类

| 错误类别 | 严重级别 | 处理策略 | 示例 |
|---------|---------|---------|------|
| **配置错误** | Fatal | 启动时 assert / 抛异常，快速失败 | 列长度不匹配、字段不存在、非法参数 |
| **数据错误** | Warning/Error | 按策略处理（skip/fill/propagate），记录日志 | NaN 输入、空值、除零 |
| **资源错误** | Fatal | 抛异常 + 日志 | BufferPool 耗尽、内存分配失败 |
| **数值错误** | Warning | 计算结果产生 NaN/Inf 时记录日志，继续执行 | log(-1)、除零产生 Inf |
| **内部错误** | Fatal | assert 终止（Debug）/ 抛异常（Release） | 违反不变量、空指针解引用 |

### 12.2 错误处理宏与工具

```cpp
// 带上下文的异常抛出
#define QUANTCORE_THROW(exceptionType, message) \
    throw exceptionType(std::string(__FILE__) + ":" + \
        std::to_string(__LINE__) + " [" + __func__ + "] " + message)

// 条件断言（Release 模式下编译为异常）
#define QUANTCORE_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            if constexpr (kIsDebugBuild) { \
                /* 记录日志后 abort */ \
            } else { \
                QUANTCORE_THROW(InternalError, message); \
            } \
        } \
    } while (0)

// 错误上下文（附加算子名、字段名、行号）
struct ErrorContext {
    std::string operatorName;
    std::string fieldName;
    size_t      rowIndex;
};
```

### 12.3 空值传播语义

算子计算时空值的传播规则：

| 算子类型 | 输入存在空值时的行为 |
|---------|-------------------|
| 一元算子 | 输入为 null → 输出为 null |
| 二元算子 (Col+Col) | 任一输入为 null → 输出为 null |
| 二元算子 (Col+Scalar) | Col输入为 null → 输出为 null |

所有算子检测到空值时，不执行计算，直接在输出对应位置标记空值。

---

## 十三、可观测性与日志系统

### 13.1 分级日志

```cpp
enum class LogLevel { TRACE, DEBUG, INFO, WARNING, ERROR, FATAL };

class Logger {
public:
    static Logger& instance();

    void setLevel(LogLevel level);
    void setOutput(std::ostream& stream);

    // 带上下文的日志宏
    void log(LogLevel level, const std::string& message,
             const std::string& file = "", int line = 0,
             const std::string& function = "");

    // 便捷宏
    #define QC_TRACE(msg)   Logger::instance().log(LogLevel::TRACE, msg, __FILE__, __LINE__, __func__)
    #define QC_DEBUG(msg)   Logger::instance().log(LogLevel::DEBUG, msg, __FILE__, __LINE__, __func__)
    #define QC_INFO(msg)    Logger::instance().log(LogLevel::INFO,  msg, __FILE__, __LINE__, __func__)
    #define QC_WARN(msg)    Logger::instance().log(LogLevel::WARNING, msg, __FILE__, __LINE__, __func__)
    #define QC_ERROR(msg)   Logger::instance().log(LogLevel::ERROR, msg, __FILE__, __LINE__, __func__)
    #define QC_FATAL(msg)   Logger::instance().log(LogLevel::FATAL, msg, __FILE__, __LINE__, __func__)
};
```

### 13.2 性能指标收集

```cpp
struct EngineMetrics {
    // 总体统计
    size_t      totalEvaluations;      // 总求值次数
    std::chrono::nanoseconds totalEvalTime;  // 总求值耗时

    // 算子级统计
    struct OperatorStats {
        std::string name;
        size_t      callCount;
        size_t      elementCount;            // 处理元素总数
        std::chrono::nanoseconds totalTime;
        std::chrono::nanoseconds minTime;
        std::chrono::nanoseconds maxTime;
    };
    std::vector<OperatorStats> operatorStats;

    // 内存统计
    size_t      peakBufferPoolBytes;   // BufferPool 峰值用量
    size_t      currentBufferPoolBytes;// BufferPool 当前用量
    size_t      bufferPoolAllocCount;  // 总分配次数
    size_t      bufferPoolHitCount;    // 缓存命中次数

    // SIMD 统计
    SimdLevel   activeSimdLevel;       // 当前使用的 SIMD 级别
    size_t      simdEvalCount;         // SIMD 路径执行次数
    size_t      scalarFallbackCount;   // 降级标量次数
};
```

### 13.3 调试模式

- **Debug 构建** (`-DCMAKE_BUILD_TYPE=Debug`)：
  - 启用所有 assert 检查
  - SIMD 路径与标量路径交叉验证（逐元素对比）
  - 数值越界检测
  - 空值传播跟踪
  - 日志级别默认为 DEBUG

- **Release 构建** (`-DCMAKE_BUILD_TYPE=Release`)：
  - 仅保留必要的运行时检查
  - 日志级别默认为 WARNING
  - 不执行交叉验证

---

## 十四、CMake 编译构建规范

### 14.1 基本要求

| 项目 | 要求 |
|------|------|
| C++ 标准 | C++17 最低，推荐启用 C++20 |
| 构建系统 | CMake ≥ 3.20 |
| 编译器 | GCC ≥ 9 / Clang ≥ 12 / MSVC ≥ 2019 |

### 14.2 编译配置

```cmake
# Debug 模式
target_compile_options(quantcore PRIVATE
    -g -O0
    -fsanitize=address,undefined   # 可选：AddressSanitizer + UBSan
)

# Release 模式（默认，保持数值正确性）
target_compile_options(quantcore PRIVATE
    -O3 -march=native -DNDEBUG
    -Wall -Wextra -Wpedantic
)

# ReleaseFast 模式（可选，允许微小精度损失）
# 用户需自行承担 -ffast-math 带来的数值差异风险
option(QUANTCORE_ENABLE_FAST_MATH "Enable -ffast-math" OFF)
if(QUANTCORE_ENABLE_FAST_MATH)
    target_compile_options(quantcore PRIVATE -ffast-math)
endif()
```

**关键变更**：`-ffast-math` **不是默认选项**，需要用户显式启用 `-DQUANTCORE_ENABLE_FAST_MATH=ON`。

### 14.3 编译开关

```cmake
option(QUANTCORE_BUILD_TESTS      "Build unit tests"         ON)
option(QUANTCORE_BUILD_BENCHMARKS "Build benchmarks"         ON)
option(QUANTCORE_BUILD_EXAMPLES   "Build example programs"   ON)
option(QUANTCORE_ENABLE_AVX512    "Enable AVX-512 intrinsics" ON)
option(QUANTCORE_ENABLE_AVX2      "Enable AVX2 intrinsics"   ON)
option(QUANTCORE_ENABLE_SSE42     "Enable SSE4.2 intrinsics" ON)
option(QUANTCORE_ENABLE_FAST_MATH "Enable -ffast-math"       OFF)
```

### 14.4 目录结构

```
QuantCore/
├── CMakeLists.txt                    # 顶层 CMake
├── cmake/
│   ├── FindGTest.cmake
│   ├── FindGBenchmark.cmake
│   └── CompilerSettings.cmake        # 编译器选项集中管理
├── quantcore/
│   ├── CMakeLists.txt                # 核心库
│   ├── core/                         # 基础类型与工具
│   │   ├── Types.h                   # Field 枚举、常量定义
│   │   ├── AlignedAllocator.h
│   │   ├── Logger.h / .cpp
│   │   └── ErrorHandling.h
│   ├── storage/                      # 存储层
│   │   ├── Column.h
│   │   ├── Column.cpp
│   │   ├── ColView.h
│   │   ├── TimestampIndex.h / .cpp
│   │   ├── MarketData.h / .cpp
│   │   └── MarketDataView.h
│   ├── expression/                   # 表达式系统
│   │   ├── ExprNode.h                # CRTP 基类
│   │   ├── ColumnRef.h
│   │   ├── Scalar.h
│   │   ├── UnaryExpr.h
│   │   ├── BinaryExpr.h
│   │   ├── FusedLoopGenerator.h
│   │   └── ExprTraits.h              # 表达式类型萃取
│   ├── operators/                    # 算子层
│   │   ├── UnaryOperator.h
│   │   ├── BinaryOperator.h
│   │   ├── unary/                    # 一元算子实现
│   │   │   ├── Abs.h
│   │   │   ├── Log.h
│   │   │   ├── Sqrt.h
│   │   │   ├── Diff.h
│   │   │   ├── Shift.h
│   │   │   └── ...
│   │   └── binary/                   # 二元算子实现
│   │       ├── Add.h
│   │       ├── Sub.h
│   │       ├── Mul.h
│   │       ├── Div.h
│   │       └── ...
│   ├── simd/                         # SIMD 向量化
│   │   ├── SimdDispatcher.h / .cpp
│   │   ├── SimdTraits.h
│   │   ├── avx512/
│   │   ├── avx2/
│   │   └── sse42/
│   ├── engine/                       # 执行引擎
│   │   ├── ExecutionEngine.h / .cpp
│   │   ├── BufferPool.h / .cpp
│   │   ├── BufferHandle.h
│   │   └── EngineMetrics.h
│   ├── io/                           # 数据 I/O
│   │   ├── CsvReader.h / .cpp
│   │   ├── BinaryWriter.h / .cpp
│   │   └── BinaryReader.h / .cpp
│   └── registry/                     # 算子注册中心
│       └── OperatorRegistry.h / .cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── unit/                         # 单元测试
│   │   ├── test_column.cpp
│   │   ├── test_market_data.cpp
│   │   ├── test_unary_ops.cpp
│   │   ├── test_binary_ops.cpp
│   │   ├── test_expression.cpp
│   │   ├── test_buffer_pool.cpp
│   │   ├── test_simd_cross_validate.cpp
│   │   └── ...
│   ├── integration/                  # 集成测试
│   │   ├── test_full_pipeline.cpp    # 端到端：加载 → 计算 → 验证
│   │   └── test_csv_roundtrip.cpp
│   └── regression/                   # 数值回归测试
│       ├── golden_factors.cpp        # 黄金标准输入输出对
│       └── data/                     # 固定测试数据集
├── benchmarks/
│   ├── CMakeLists.txt
│   ├── bench_unary_ops.cpp
│   ├── bench_binary_ops.cpp
│   ├── bench_expression_fusion.cpp
│   └── bench_buffer_pool.cpp
├── examples/
│   ├── CMakeLists.txt
│   ├── demo_basic_factors.cpp        # 基础因子计算演示
│   └── data/                         # 示例数据文件
│       └── sample_000001.csv
├── python/                           # Python 绑定（预留）
│   ├── CMakeLists.txt
│   └── quantcore_bindings.cpp
└── docs/
    └── QuantCore_Design_v2.md
```

---

## 十五、测试与性能基准规范

### 15.1 测试金字塔

```
         ┌─────────────┐
         │  集成测试     │  ~10%  — 端到端链路验证
         │  Integration │
         ├─────────────┤
         │  回归测试     │  ~15%  — 数值正确性 Golden File
         │  Regression  │
         ├─────────────┤
         │  属性测试     │  ~10%  — 数学恒等式随机验证
         │  Property    │
         ├─────────────┤
         │  单元测试     │  ~65%  — 覆盖边界值、空值、异常场景
         │  Unit Tests  │
         └─────────────┘
```

### 15.2 单元测试规范（GoogleTest）

| 测试类别 | 覆盖内容 |
|---------|---------|
| Column\<T\> | 构造/拷贝/移动/越界/空值标记/aligned 属性/空列/大容量 |
| ColView\<T\> | 零拷贝验证/切片正确性/越界/嵌套切片/空视图 |
| TimestampIndex | 日期解析/连续交易日判断/日期范围切片 |
| MarketData | 7 字段读写/长度对齐校验/切片/assetId/空值传播 |
| MarketDataView | 零拷贝语义/生命周期/嵌套切片 |
| UnaryOperator | 每种算子的正确性、NaN 输入、空值输入、边界值（+Inf/-Inf/0） |
| BinaryOperator | 长度不匹配断言、除零行为、空值传播、标量+向量一致性 |
| Expression | 嵌套表达式正确性、算子融合等价性、CSE 正确性 |
| BufferPool | 分配/归还/RAII 句柄/多级 slab/峰值追踪/对齐保证 |
| OperatorRegistry | 注册/查询/重复注册/未注册查询 |

### 15.3 SIMD 交叉验证测试（重点）

```cpp
// 对于每个算子，必须验证 SIMD 路径与标量路径结果一致
TEST(UnaryOpSimd, AbsCrossValidate) {
    // 生成随机测试数据（包含边界值和特殊值）
    auto input = generateTestData({
        .size = 1024,
        .includeNaN    = true,
        .includeInf    = true,
        .includeNeg    = true,
        .includeZero   = true,
        .includeSubnormal = true,
    });

    auto scalarResult = evaluateScalar<AbsOp>(input);
    auto avx2Result   = evaluateAvx2<AbsOp>(input);
    auto avx512Result = evaluateAvx512<AbsOp>(input);

    // 逐元素对比，允许 1 ULP 误差（仅 -ffast-math 模式下）
    EXPECT_COLUMNS_NEAR(scalarResult, avx2Result, kMaxUlpDiff);
    EXPECT_COLUMNS_NEAR(scalarResult, avx512Result, kMaxUlpDiff);
}
```

### 15.4 数值回归测试

```cpp
// Golden File 测试：确保重构不改变数值结果
TEST(RegressionTest, StandardFactorSet) {
    auto data = loadGoldenMarketData("test/data/golden_input.csv");
    auto engine = ExecutionEngine(EngineConfig::debugMode());

    // 计算标准因子集
    auto momentum = engine.evaluate(close / shift(close, 20) - 1.0);
    auto amplitude = engine.evaluate((high - low) / close);
    auto logRet = engine.evaluate(log(close) - log(shift(close, 1)));

    // 与 Golden File 对比（精确到 1e-12）
    assertNearGolden("momentum_20d.csv", momentum);
    assertNearGolden("amplitude.csv", amplitude);
    assertNearGolden("log_return.csv", logRet);
}
```

### 15.5 属性测试（Property-Based Testing）

```cpp
// 数学恒等式验证 — 对随机输入始终保持
TEST(PropertyTest, AbsNonNegative) {
    for (int i = 0; i < kRandomRounds; ++i) {
        auto input = randomColumn(/*size=*/500);
        auto result = engine.evaluate(abs(col(input)));
        ASSERT_TRUE(allNonNegative(result));   // ABS 结果必须全部 ≥ 0
    }
}

TEST(PropertyTest, LogProductRule) {
    for (int i = 0; i < kRandomRounds; ++i) {
        auto a = randomPositiveColumn(500);
        auto b = randomPositiveColumn(500);
        auto lhs = engine.evaluate(log(a * b));
        auto rhs = engine.evaluate(log(a) + log(b));
        EXPECT_COLUMNS_NEAR(lhs, rhs, /*maxUlp=*/4);  // 允许微小浮点误差
    }
}
```

### 15.6 性能基准测试规范（GoogleBenchmark）

| 基准测试 | 测试内容 | 数据规模 |
|---------|---------|---------|
| bench_unary_ops | 每种一元算子的 SIMD vs Scalar 吞吐量对比 | 1K / 10K / 100K / 1M 行 |
| bench_binary_ops | 每种二元算子的 SIMD vs Scalar 吞吐量对比 | 同上 |
| bench_expression_fusion | 嵌套表达式的融合 vs 非融合性能对比 | 同上 |
| bench_buffer_pool | 缓冲池分配/归还吞吐量、命中率 | 1000 次操作 |
| bench_csv_load | CSV 加载吞吐量 | 1MB / 10MB / 100MB 文件 |
| bench_binary_load | 二进制格式 mmap 加载吞吐量 | 同上 |

### 15.7 标准测试数据

所有测试使用同一份标准数据集：
- 至少包含 250 个交易日（一年的交易日数）
- 覆盖连续交易、节假日间隔、停牌日（含空值）
- 包含极端行情场景（涨跌停、大幅跳空、除权除息）
- 文件名：`test/data/standard_kline_250.csv`

---

## 十六、AI 代码交付硬性要求

基于本文档规范输出完整可独立编译运行的 C++ QuantCore 工程代码。交付清单如下：

### 16.1 目录与文件

严格按照第十四节 14.4 的目录结构组织代码。

### 16.2 一期必实现功能

| 模块 | 交付内容 |
|------|---------|
| 存储层 | Column\<T\>（泛型列式容器）、ColView\<T\>（零拷贝视图）、TimestampIndex（时间轴）、MarketData（单资产数据集）、MarketDataView（零拷贝切片）、AlignedAllocator\<T, 64\>（对齐分配器） |
| 空值机制 | bitmask 空值标记、惰性分配、空值传播语义 |
| 表达式系统 | ExprNode CRTP 基类、ColumnRef/Scalar 叶节点、UnaryExpr/BinaryExpr 模板节点、FusedLoopGenerator 融合循环生成器 |
| 算子层 | 第十节第 5.2 条列出的全部一元、二元算子，每个算子包含标量参考实现 |
| SIMD | AVX2 intrinsics 实现路径 + SSE4.2 fallback + 标量兜底 + SimdDispatcher 运行时派发 |
| 执行引擎 | ExecutionEngine（evaluate / evaluateFused / evaluateBatch）、BufferPool（分级 Slab Allocator + RAII BufferHandle） |
| 算子注册 | OperatorRegistry（一元/二元算子注册与查询） |
| 数据 I/O | CsvMarketDataReader、BinaryMarketDataWriter / BinaryMarketDataReader（含 mmap 零拷贝加载） |
| 日志 | Logger（分级日志，默认 WARNING 级别） |
| 错误处理 | 异常体系 + QUANTCORE_ASSERT 宏 + 空值传播 |

### 16.3 远期预留接口（仅声明，不实现）

- CUDA / HIP / OpenCL GPU 后端（不创建抽象基类，仅在注释中标注扩展点）
- RollingOperator 基类和具体实现
- CrossSectionOperator 基类和具体实现
- ReductionOperator 基类和具体实现
- 字符串公式解析器
- MarketDataBundle 多资产面板
- 遗传规划因子生成模块

### 16.4 示例程序

提供 `examples/demo_basic_factors.cpp`：

1. 从 CSV 加载示例数据（附带 `examples/data/sample_000001.csv`）
2. 构建多个演示因子：
   - 日内振幅：`(HIGH - LOW) / CLOSE`
   - 对数收益率：`LOG(CLOSE) - LOG(SHIFT(CLOSE, 1))`
   - 量价复合因子：`ABS(LOG(CLOSE) - LOG(VWAP)) * VOLUME`
3. 输出前 20 行计算结果到控制台
4. 输出引擎性能指标（耗时、内存峰值、SIMD 级别）

### 16.5 CMake 构建

- 顶层 `CMakeLists.txt` 支持所有第十四节 14.3 的编译开关
- `tests/CMakeLists.txt` 使用 `find_package(GTest)` 自动查找或下载
- `benchmarks/CMakeLists.txt` 使用 `find_package(benchmark)` 自动查找或下载
- 提供 `scripts/build.sh` 和 `scripts/build_debug.sh` 便捷构建脚本

### 16.6 代码注释规范

- 每个 `.h` 文件顶部注明"一期必实现"或"远期预留"
- 每个类的公开接口使用 Doxygen 风格注释
- SIMD 代码块详细注释对应的标量等价逻辑
- 空值处理分支明确标注"空值传播"语义

### 16.7 测试交付

- 每个模块至少 3 个测试用例
- 每个算子包含 SIMD vs Scalar 交叉验证测试
- 提供 2 个集成测试用例（CSV 加载→计算→验证）
- 提供 1 个数值回归测试（Golden File）

---

## 附录 A：与 v1.0 的主要变更对照

| 模块 | v1.0 | v2.0 |
|------|------|------|
| **数据存储** | Column\<double\> 仅 double | Column\<T\> 泛型，支持 int64_t / bool |
| **资产标识** | 无 | MarketData.assetId_ + TimestampIndex |
| **空值处理** | 未设计 | bitmask 惰性分配 + 空值传播语义 |
| **表达式系统** | 链式 API 立即求值 | 表达式模板惰性求值 + 算子融合 |
| **SIMD** | 仅编译器自动向量化 | intrinsics 优先 + 编译器 fallback + 运行时派发 |
| **硬件后端** | 过早抽象层 | 一期无抽象层，直接 CPU SIMD 实现 |
| **BufferPool** | 3 句话描述 | 分级 Slab Allocator + RAII 句柄 + 统计接口 |
| **ffast-math** | 默认启用 | 默认关闭，需用户显式开关 |
| **数据 I/O** | 无 | CSV + 二进制 mmap + Python 绑定预留 |
| **错误处理** | 无 | 五级分类 + 异常体系 + 诊断上下文 |
| **可观测性** | 无 | 分级日志 + EngineMetrics 统计 |
| **测试** | 仅单元+微基准 | 测试金字塔：单元/属性/回归/集成 |
| **多资产** | 无 | MarketDataBundle（远期预留接口） |
| **设计原则** | 存在内部矛盾 | 明确优先级：正确性 > 可复现性 > 性能 |

---

## 附录 B：关键设计决策记录 (ADR)

| ID | 决策 | 理由 | 权衡 |
|----|------|------|------|
| ADR-01 | VOLUME/AMOUNT 使用 int64_t 存储 | 避免 double 精度损失（53-bit 有效位不足） | 增加类型系统复杂度，需类型提升机制 |
| ADR-02 | 表达式模板惰性求值 | 消除链式 API 与零拷贝的矛盾，实现算子融合 | 编译错误信息较差，需额外 SFINAE/Concept 友好设计 |
| ADR-03 | 一期不引入硬件后端抽象层 | 避免"过早抽象"反模式 | 远期 GPU 接入时可能需要重构部分代码 |
| ADR-04 | ffast-math 默认关闭 | 金融计算优先保证数值正确性和可复现性 | 牺牲约 5-15% 的潜在 SIMD 优化空间 |
| ADR-05 | bitmask 惰性分配 | 大部分数据无空值，避免为常见场景付费 | 需要额外 if 分支检查 hasNull_ |
| ADR-06 | 每算子提供标量参考实现 | SIMD 路径交叉验证的基础，也是文档化的"可执行规范" | 维护两套实现的小幅成本 |
