# QuantCore 设计框架复盘

> 评估日期：2026-07-17
>
> 评估范围：当前已实现模块 + 已搭建接口，对照四个核心需求逐项分析。

---

## 一、当前架构总览

```
┌─────────────────────────────────────────────────────────┐
│                     ExecutionEngine                      │  ← 空文件
│  表达式求值 · 算子融合 · SIMD 派发 · BufferPool 管理      │
├─────────────────────────────────────────────────────────┤
│                     表达式系统 (ET)                       │
│  ExprNode → UnaryExpr / BinaryExpr → FusedLoopGenerator │  ← 全空
├──────────────┬────────────────┬─────────────────────────┤
│   UnaryOps   │   BinaryOps    │     数据 I/O 层           │
│  (11个占位)   │  (14个占位)    │  Parquet/HDF5 ✅          │
├──────────────┴────────────────┴─────────────────────────┤
│              Rolling / CrossSection / Reduction          │  ← 接口已搭建
├─────────────────────────────────────────────────────────┤
│                     数据存储层                            │  ← ✅ 已实现
│  Column<T> → ColView<T> → TimestampIndex                 │
│  MarketData → MarketDataView                             │
├─────────────────────────────────────────────────────────┤
│                  基础设施                                 │  ← ✅ 已实现
│  AlignedAllocator · Logger · ErrorHandling · Types       │
└─────────────────────────────────────────────────────────┘

✅ = 已实现   🔧 = 接口已搭建，内容为空   🔴 = 完全缺失
```

**五层算子枚举：**

| 类型 | 数量 | 状态 |
|------|------|------|
| `UnaryOpCode` | 11 (ABS..NOT) | 枚举定义 ✅ / CRTP 基类 🔴 / 实现 🔴 |
| `BinaryOpCode` | 14 (ADD..XOR) | 枚举定义 ✅ / CRTP 基类 🔴 / 实现 🔴 |
| `RollingOpCode` | 7 (SMA..RANK) | 枚举定义 ✅ / CRTP 基类 ✅ / 实现 🔴 |
| `CrossSectionOpCode` | 3 (RANK..QUANTILE) | 枚举定义 ✅ / CRTP 基类 ✅ / 实现 🔴 |
| `ReductionOpCode` | 6 (SUM..PROD) | 枚举定义 ✅ / CRTP 基类 ✅ / 实现 🔴 |

**关键发现：** 五个 CRTP 基类中，**UnaryOperator 和 BinaryOperator（最核心的 25 个算子）的基类仍然是空文件**，只有 Rolling/CrossSection/Reduction 三个远期接口写了详细的协议注释。

---

## 二、需求 1：方便后续扩展新算子 — ✅ 方向正确，⚠️ 存在瓶颈

### 当前流程（新增一个算子）

```
Step 1: Types.h          → 加一行枚举值（kCount 之前）
Step 2: operators/xxx/   → 新建 Xxx.h 结构体文件
Step 3: OperatorRegistry.cpp → kXxxOpNames 数组加一行字符串
Step 4: tests/           → 写单元测试
```

### 优点

1. **模式固定** — 每种算子类型的添加步骤完全一致，新人容易上手
2. **`OperatorRegistry` 双向映射完整** — name→enum（find 方法）和 enum→name（opName 函数）均已实现，日志/诊断/远期字符串解析都能复用
3. **目录结构对称** — 五类算子完全平行的目录布局，找参考实现只需看相邻文件

### 瓶颈

**集中式枚举是单点修改瓶颈。** `Types.h` 中的 5 个 enum 是所有算子的"中央户口本"，每次新增算子都要改这个文件。多人协作时必然冲突。

而且每个算子信息散落在 **4 个地方**：

| 信息 | 位置 |
|------|------|
| 枚举值 | `Types.h` |
| 结构体 | `operators/xxx/Xxx.h` |
| 名称字符串 | `OperatorRegistry.cpp` 的 `kXxxOpNames` 数组 |
| 派发 switch | 未来 `SimdDispatcher` / `FusedLoopGenerator` 中的 switch 分支 |

加一个算子 = 至少改 3 个文件。容易遗漏。

### 改进建议：就地注册宏

```cpp
// operators/unary/Abs.h 底部
REGISTER_UNARY_OP(AbsOp, "abs",
    /* scalar  */ [](double x) { return std::abs(x); },
    /* avx2    */ &abs_avx2_kernel,
    /* avx512  */ &abs_avx512_kernel
);
```

编译期宏展开后自动汇入全局注册表。新增算子只需新建一个 `.h` 文件——不再需要改 `Types.h`、不再需要改 `OperatorRegistry.cpp`、不再需要改派发 switch。

LLVM 的 Pass Registry 和 PyTorch 的 `TORCH_LIBRARY` 都用了这个模式——量化因子库的算子数量会持续膨胀（Rolling 迟早要加 CORR/BETA/COV/RSI），集中式枚举越往后越疼。

**评级：⭐⭐⭐ (3/5) — 规则清晰但存在单点瓶颈**

---

## 三、需求 2：方便以字符串方式写因子 — 🔴 前置依赖全部缺失

### 目标效果

```cpp
// 目标：字符串 → 表达式树 → 计算
auto expr = engine.parse("ABS(LOG(CLOSE)-LOG(VWAP))*VOLUME");
Column<double> result = engine.evaluate(expr);
```

### 需要的组件及状态

```
字符串公式 "ABS(LOG(CLOSE)-LOG(VWAP))*VOLUME"
      │
      ▼
  词法分析器 ──────────────────────── 🔴 不存在
  Token: ABS, (, LOG, (, CLOSE, ), -, ...
      │
      ▼
  语法分析器 ──────────────────────── 🔴 不存在
      │
      ▼
  表达式 AST ──────────────────────── 🔴 全空
  ├── ExprNode.h (AST 基类)           🔴 0 字节
  ├── ColumnRef.h (列引用叶节点)       🔴 0 字节
  ├── Scalar.h (标量叶节点)            🔴 0 字节
  ├── UnaryExpr.h (一元表达式节点)     🔴 0 字节
  ├── BinaryExpr.h (二元表达式节点)    🔴 0 字节
  ├── ExprTraits.h (类型萃取)          🔴 0 字节
  └── FusedLoopGenerator.h (融合)     🔴 0 字节
      │
      ▼
  OperatorRegistry (name→enum 映射) ── ✅ 已实现（唯一就绪的环节）
      │
      ▼
  执行引擎 ────────────────────────── 🔴 空文件
  Column<double> 结果
```

**整个链路 8 个组件，7 个是空的。**

`OperatorRegistry` 的 name→enum 映射是目前唯一可用的部分——它能查到 `"ABS"` 对应 `UnaryOpCode::ABS`。但从 enum 到表达式节点、从节点到执行，中间的一切都不存在。

### 当前硬编码 API vs 字符串公式

```cpp
// ✅ 可以工作（硬编码链式 API）
auto expr = abs(log(close) - log(vwap)) * volume.toDouble();

// 🔴 完全不行（字符串公式）
auto expr = engine.parse("ABS(LOG(CLOSE)-LOG(VWAP))*VOLUME");
```

### 建议实现路径

```
ExprNode → ColumnRef/Scalar → UnaryExpr/BinaryExpr
  → RollingExpr/CrossSectionExpr/ReductionExpr
    → FusedLoopGenerator
      → 公式解析器
```

这是最长的一条链——需要最早启动。每一步都能独立测试（有了 AST 后手写树来验证 evaluate 正确性，再用解析器来自动生成树）。

**评级：⭐ (1/5) — 前置依赖全部缺失**

---

## 四、需求 3：方便高性能因子计算 — ⚠️ 架构正确，实现空白

### 设计文档要求的高性能策略

| 策略 | 原理 | 实现状态 |
|------|------|---------|
| **表达式模板 (CRTP)** | 编译期类型编码，零虚函数开销 | CRTP 基类已定义（Rolling/CrossSection/Reduction），但具体 `evaluateScalar` 未实现 |
| **算子融合** | 嵌套表达式编译为单次循环，零中间临时列 | `FusedLoopGenerator.h` — 空文件 |
| **SIMD 向量化** | 4 级运行时派发（AVX-512/AVX2/SSE4.2/Scalar） | `SimdDispatcher` — 空文件，`simd/avx2/`、`simd/avx512/` 目录只有 `.gitkeep` |
| **BufferPool** | 分级 Slab Allocator 复用中间内存 | `BufferPool.h/.cpp` — 空文件 |
| **零拷贝** | ColView 不持有数据，mmap 直接映射 | ColView ✅ / MarketDataView ✅ |
| **64 字节对齐** | 适配 AVX-512 缓存行 | AlignedAllocator ✅ |

### 核心问题：只有"描述层"，没有"执行层"

当前每个算子文件实质内容：

```cpp
// operators/unary/Abs.h — 这就是全部
struct AbsOp {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::ABS;
    static constexpr const char* name = "abs";
    // evaluateScalar 只在注释里
};
```

只有标识信息（谁是谁），没有计算逻辑（怎么做）。`evaluateScalar()` 只在 CRTP 基类的注释中作为一个"约定的协议"存在——编译期不强制，运行时不可调用。

### 关键设计挑战：五类算子的 evaluate 签名互不兼容

```cpp
// 当前各 CRTP 基类注释中"约定"的签名 — 互不相同：
UnaryOp:   double evaluateScalar(double x)
BinaryOp:  double evaluateScalar(double a, double b)
Rolling:   double evaluateScalar(const double* input, size_t i, size_t window)
CrossSec:  double evaluateScalar(const double* assets, size_t idx, size_t N)
Reduction: double evaluateScalar(const double* data, size_t n, const uint64_t* mask)
```

这五种签名无法塞进同一个 `for(i) result[i] = op.evaluateAt(i)` 循环。

**这意味着 FusedLoopGenerator 不能简单地展开成统一循环**——它需要：
1. 识别每个 AST 节点的算子类别
2. Unary/Binary 算子 → 逐元素融合循环（可以深度融合）
3. Rolling 节点 → 形成融合循环的**边界**（窗口语义不可拆分）
4. CrossSection 节点 → 形成表达式 DAG 的**边界**（需要全部资产在同一时间点的值）
5. Reduction 节点 → 先求标量，再广播回列维度

### 建议执行策略

**分期实现，不要试图一上来就五类融合：**

```
Phase A: Unary + Binary 的完整执行链路
  └── 接口统一，可深度融合，验证性能达标

Phase B: Rolling 算子
  └── 窗口边界，无法与逐元素算子深度融合，独立执行

Phase C: Reduction 算子
  └── 标量归约 + 广播，可实现 CSE 复用

Phase D: CrossSection 算子
  └── 依赖 MarketDataBundle，最后实现
```

**评级：⭐⭐ (2/5) — 路线正确，但离能跑差整个执行层**

---

## 五、需求 4：使用因子遵从统一框架 — ⚠️ 有框架雏形，接口不统一

### 优点

**目录结构完全对称。** 五类算子的组织方式一模一样：

```
operators/
  UnaryOperator.h       → unary/Abs.h, Log.h, Not.h, ...
  BinaryOperator.h      → binary/Add.h, Mul.h, And.h, ...
  RollingOperator.h     → rolling/Sma.h, Ema.h, ...
  CrossSectionOperator.h → cross_section/Rank.h, ...
  ReductionOperator.h   → reduction/Sum.h, Mean.h, ...
```

开发者只需找到对应目录，参照相邻文件的模式即可——这是统一的。

`OperatorRegistry` 也是统一的——五类算子共用同一个 `find`/`list`/`register` 模板方法模式。

### 问题 1：最核心的 CRTP 基类还是空的

`UnaryOperator.h` 和 `BinaryOperator.h` 这两个使用频率最高的基类仍是 0 字节。后三类（Rolling/CrossSection/Reduction）的基类虽然写了详细注释，但"接口协议"只存在于自然语言注释中：

```cpp
// 当前 RollingOperator.h 的"接口协议"——是注释，不是代码
//    static double evaluateScalar(const double* input,
//                                 std::size_t i,
//                                 std::size_t window) noexcept;
```

如果子类没按协议实现，**没有任何编译期检查**。正确的做法是让 CRTP 基类在编译期强制检查：

```cpp
template <typename Derived>
class UnaryOperator {
public:
    // 编译期强制子类提供 evaluateScalar
    double evaluateAt(size_t i) const {
        return static_cast<const Derived*>(this)->evaluateScalar(data_[i]);
    }
    // 如果 Derived 没有 evaluateScalar → 编译报错
};
```

### 问题 2：缺少"因子"这一顶层抽象

当前用户直接操作的是底层 API：

```cpp
// 当前：用户直接操作 Column + 表达式
auto& close = md.column<double>(Field::CLOSE);
auto expr = (close - shift(close, 1)) / shift(close, 1);
auto result = engine.evaluate(expr);
```

缺少一个"因子"概念来封装名称、参数、输入字段：

```cpp
// 理想：统一的 Factor 接口
Factor momentum("momentum")
    .withParam("window", 20)
    .withInput(Field::CLOSE)
    .withExpression([](auto& close) {
        return (close - shift(close, 20)) / shift(close, 20);
    });

// 不管内部是用 Unary/Rolling/CrossSection 哪种算子实现的
Column<double> result = engine.evaluate(momentum, marketData);
```

`Factor` 对象封装了：
- **名称** → 日志/注册/序列化
- **参数** → window=20, min_periods=10
- **输入字段** → Field::CLOSE, Field::VOLUME
- **表达式** → 链式 API 构建的计算 DAG

有了这一层后，批量化管理因子、序列化/反序列化、因子列表展示等功能才有统一的载体。

### 建议

```
Step 1: 先定义 UnaryOperator / BinaryOperator 的 CRTP 基类
        （让 25 个高频算子有编译期接口约束）

Step 2: 设计 Factor 类
        （名称 + 参数 + 输入字段 + 表达式）
```

**评级：⭐⭐ (2/5) — 目录框架统一，缺失编译期接口约束和顶层 Factor 抽象**

---

## 六、修复路径（按优先级排序）

| 优先级 | 任务 | 解决的需求 | 预估工作量 |
|--------|------|-----------|-----------|
| **P0** | 填充 `ExprNode.h` + `ColumnRef.h` + `Scalar.h` + `UnaryExpr.h` + `BinaryExpr.h` | #2 字符串公式前置 + #4 统一框架 | 核心基础设施 |
| **P1** | 定义 `UnaryOperator.h` / `BinaryOperator.h` CRTP 基类（编译期接口约束） | #4 统一框架 + #3 高性能 | 锁定 25 个算子的协议 |
| **P1** | 实现 1-2 个 Unary（如 Abs）+ 1-2 个 Binary（如 Add）的完整执行链路 | #3 高性能 | 打通端到端验证 |
| **P2** | 算子改为就地注册宏（消除 `Types.h` 瓶颈） | #1 扩展性 | 重构现有 41 个算子的注册方式 |
| **P2** | 设计 `Factor` 顶层抽象类 | #4 统一框架 | 封装名称/参数/输入/表达式 |
| **P3** | 实现 `FusedLoopGenerator`（Unary+Binary 深度融合） | #3 高性能 | 算子融合的核心 |
| **P3** | 实现字符串公式解析器（词法 + 语法 → AST） | #2 字符串公式 | 依赖 P0 和 P1 完成 |
| **P3** | 实现 `ExecutionEngine` + `BufferPool` | #3 高性能 | 调度和内存管理层 |

---

## 七、关键文件状态总表

| 文件 | 状态 | 内容 |
|------|------|------|
| `core/Types.h` | ✅ | 5 个算子枚举 + Field + SimdLevel |
| `core/AlignedAllocator.h` | ✅ | 64 字节对齐分配器 |
| `core/ErrorHandling.h` | ✅ | 异常层次 + 断言宏 |
| `core/Logger.{h,cpp}` | ✅ | 分级日志系统 |
| `storage/Column.{h,cpp}` | ✅ | 泛型列式容器 + 空值 bitmask |
| `storage/ColView.h` | ✅ | 零拷贝只读视图 |
| `storage/MarketData.{h,cpp}` | ✅ | 单资产 7 字段数据集 |
| `storage/TimestampIndex.{h,cpp}` | ✅ | 时间轴索引 |
| `storage/MarketDataBundle.h` | 🔧 | 远期存根 |
| `io/ParquetReader.{h,cpp}` | ✅ | Parquet 日线数据读取 |
| `io/HDF5Reader.{h,cpp}` | ✅ | HDF5 日线数据读取 |
| `io/CsvReader.{h,cpp}` | 🔴 | 空文件 |
| `io/BinaryReader.{h,cpp}` | 🔴 | 空文件 |
| `io/BinaryWriter.{h,cpp}` | 🔴 | 空文件 |
| `operators/UnaryOperator.h` | 🔴 | **0 字节 — P0 待定义** |
| `operators/BinaryOperator.h` | 🔴 | **0 字节 — P0 待定义** |
| `operators/RollingOperator.h` | 🔧 | CRTP 基类已定义 + 协议注释 |
| `operators/CrossSectionOperator.h` | 🔧 | CRTP 基类已定义 + 协议注释 |
| `operators/ReductionOperator.h` | 🔧 | CRTP 基类已定义 + 协议注释 |
| `operators/{unary,binary,rolling,cross_section,reduction}/*.h` | 🔧 | 41 个占位结构体（只有 kOpCode + name） |
| `expression/ExprNode.h` | 🔴 | **0 字节 — P0 最高优先级** |
| `expression/UnaryExpr.h` | 🔴 | 0 字节 |
| `expression/BinaryExpr.h` | 🔴 | 0 字节 |
| `expression/ColumnRef.h` | 🔴 | 0 字节 |
| `expression/Scalar.h` | 🔴 | 0 字节 |
| `expression/ExprTraits.h` | 🔴 | 0 字节 |
| `expression/FusedLoopGenerator.h` | 🔴 | 0 字节 |
| `expression/{Rolling,CrossSection,Reduction}Expr.h` | 🔧 | 占位注释 |
| `registry/OperatorRegistry.{h,cpp}` | ✅ | 5 类算子 name↔enum 映射 + find/list |
| `engine/ExecutionEngine.{h,cpp}` | 🔴 | 空文件 |
| `engine/BufferPool.{h,cpp}` | 🔴 | 空文件 |
| `engine/BufferHandle.h` | 🔴 | 空文件 |
| `engine/EngineMetrics.h` | 🔴 | 空文件 |
| `simd/SimdTraits.h` | 🔴 | 空文件 |
| `simd/SimdDispatcher.{h,cpp}` | 🔴 | 空文件 |

**总行数：** 约 3500 行（含 Parquet/HDF5 reader 的 ~780 行），其中算子/表达式/引擎核心链路 **约 600 行（全部为接口注释）**。

---

## 八、结论

当前框架的主要矛盾是：**描述层（枚举+注册）完整，执行层（表达式+融合+SIMD）空白。**

四个需求中，扩展性（#1）是最不令人担心的——规则已经有了，只需迭代优化注册机制。真正卡脖子的是表达式 AST（#2 的前置依赖）和 Unary/Binary CRTP 基类（#3 #4 的前置依赖）。

**下一步最值得投入的：按 P0 → P1 顺序，先打通 Unary/Binary 的完整链路。** 这个链路一旦跑通，Rolling/CrossSection/Reduction 就可以复用相同的模式——它们只是"特殊边界算子"，不是另起炉灶。
