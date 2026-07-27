# QuantCore 添加算子指南

根据对代码库的全面分析，以下是添加新算子的完整步骤。当前项目处于一期开发阶段——存储层和 I/O 层已实现，算子层和表达式层的接口文件已搭建（目前为空），等待填充实现。

---

## 整体架构回顾

```
添加一个算子需要触及 5 层：

  Types.h (枚举注册)
     ↓
  operators/unary/Xxx.h 或 operators/binary/Xxx.h (算子实现)
     ↓
  expression/UnaryExpr.h / BinaryExpr.h (表达式节点，模板参数引用算子)
     ↓
  registry/OperatorRegistry.h/.cpp (名称→枚举映射)
     ↓
  tests/unit/test_unary_ops.cpp / test_binary_ops.cpp (测试)
     ↓
  benchmarks/bench_unary_ops.cpp / bench_binary_ops.cpp (性能基准)
```

---

## 步骤 1：在 `Types.h` 中注册算子枚举

**文件**: `quantcore/core/Types.h:39-66`

**一元算子** — 在 `UnaryOpCode` 枚举中添加新成员：

```cpp
enum class UnaryOpCode : uint8_t {
    ABS,
    LOG,
    LOG10,
    SQRT,
    NEG,
    DIFF,
    SHIFT,
    SIGN,
    SQUARE,
    EXP,
    // ↓ 添加新算子
    CBRT,     // 立方根 (示例)
    kCount,   // ← kCount 必须始终在最后
};
```

**二元算子** — 在 `BinaryOpCode` 枚举中添加：

```cpp
enum class BinaryOpCode : uint8_t {
    ADD,
    SUB,
    MUL,
    DIV,
    POW,
    MAX,
    MIN,
    GT,
    LT,
    EQ,
    NEQ,
    // ↓ 添加新算子
    LOG_ADD,  // log(exp(a) + exp(b)) (示例)
    kCount,
};
```

---

## 步骤 2：实现算子类

### 2a. 一元算子模板

**新建文件**: `quantcore/operators/unary/Cbrt.h`（以立方根为例）

根据设计文档 v2.0 第四节和第五节的规范，算子应该：

```cpp
// quantcore/operators/unary/Cbrt.h
#pragma once

#include <cmath>
#include "quantcore/core/Types.h"

namespace quantcore {

struct CbrtOp {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::CBRT;

    // 标量参考实现 — SIMD 交叉验证的对照基准
    static double evaluateScalar(double x) noexcept {
        return std::cbrt(x);  // 对所有实数有效，无需空值检查（空值传播在调用层处理）
    }

    // SIMD 内核 — 一期可仅声明，用标量 fallback
    // 远期填充 AVX2/AVX-512 intrinsics 实现
    template <SimdLevel L>
    static void evaluateSimd(const double* __restrict__ input,
                             double* __restrict__ output,
                             const uint64_t* __restrict__ nullMask,
                             std::size_t n) noexcept {
        // 一期：标量循环 + 编译器自动向量化
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            output[i] = std::cbrt(input[i]);
        }
    }

    // 算子描述 (用于日志/错误信息)
    static constexpr const char* name = "cbrt";
};

}  // namespace quantcore
```

### 2b. 二元算子模板

**新建文件**: `quantcore/operators/binary/LogAdd.h`

```cpp
// quantcore/operators/binary/LogAdd.h
#pragma once

#include <cmath>
#include "quantcore/core/Types.h"

namespace quantcore {

struct LogAddOp {
    static constexpr BinaryOpCode kOpCode = BinaryOpCode::LOG_ADD;

    // 标量参考实现
    static double evaluateScalar(double a, double b) noexcept {
        // log(exp(a) + exp(b)) 的数值稳定版本
        if (a > b) {
            return a + std::log1p(std::exp(b - a));
        } else {
            return b + std::log1p(std::exp(a - b));
        }
    }

    // SIMD 内核（一期标量 fallback）
    template <SimdLevel L>
    static void evaluateSimd(const double* __restrict__ lhs,
                             const double* __restrict__ rhs,
                             double* __restrict__ output,
                             const uint64_t* __restrict__ nullMask,
                             std::size_t n) noexcept {
        #pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            double a = lhs[i], b = rhs[i];
            output[i] = (a > b) ? (a + std::log1p(std::exp(b - a)))
                                : (b + std::log1p(std::exp(a - b)));
        }
    }

    static constexpr const char* name = "log_add";
};

}  // namespace quantcore
```

### 算子设计原则 (来自规范)

| 原则 | 说明 |
|------|------|
| **纯逻辑无内存** | 算子类自身不持有内存，只描述运算逻辑 |
| **输入为 ColView** | 通过 `ColView<T>` 引用输入数据，不拥有所有权 |
| **输出由引擎注入** | 结果写入引擎提供的目标缓冲区（BufferPool 或最终 Column） |
| **标量参考必须存在** | 每个算子提供一个 `evaluateScalar()` 作为 SIMD 交叉验证基准 |

### 空值传播语义 (来自规范)

| 场景 | 行为 |
|------|------|
| 一元算子输入 null | 输出 = null |
| 二元 Col+Col 任一 null | 输出 = null |
| 二元 Col+Scalar Col 为 null | 输出 = null |
| Diff 首元素 | 输出 = null |
| Shift 越界 | 输出 = null |

---

## 步骤 3：集成到表达式系统

表达式系统使用 **CRTP 静态多态**（零虚函数）。`UnaryExpr` 和 `BinaryExpr` 是模板类，算子类型作为模板参数。

**文件**: `quantcore/expression/UnaryExpr.h`（需填充实现）

```cpp
// 核心模式：
template <typename Op, typename Child>
class UnaryExpr : public ExprNode<UnaryExpr<Op, Child>> {
public:
    UnaryExpr(const Child& child) : child_(child) {}

    double evaluateAt(size_t i) const {
        double val = child_.evaluateAt(i);
        // 空值传播逻辑在 ExprNode 基类或此处统一处理
        return Op::evaluateScalar(val);
    }

    size_t sizeImpl() const { return child_.size(); }

private:
    Child child_;
};
```

**关键点**: 添加新算子**不需要修改** `UnaryExpr`/`BinaryExpr` 模板本身——这些是泛型模板，新算子类型自动适配。只需确保算子类提供 `evaluateScalar()` 静态方法即可。

---

## 步骤 4：注册到 OperatorRegistry

**文件**: `quantcore/registry/OperatorRegistry.cpp`（需填充实现）

```cpp
// OperatorRegistry 维护 算子枚举 ↔ 名称字符串 的双向映射
// 用于：日志输出、性能统计、远期字符串公式解析

#include "quantcore/registry/OperatorRegistry.h"
#include "quantcore/core/Types.h"

namespace quantcore {

static const std::unordered_map<UnaryOpCode, const char*> unaryOpNames = {
    {UnaryOpCode::ABS,    "abs"},
    {UnaryOpCode::LOG,    "log"},
    {UnaryOpCode::LOG10,  "log10"},
    {UnaryOpCode::SQRT,   "sqrt"},
    {UnaryOpCode::NEG,    "neg"},
    {UnaryOpCode::DIFF,   "diff"},
    {UnaryOpCode::SHIFT,  "shift"},
    {UnaryOpCode::SIGN,   "sign"},
    {UnaryOpCode::SQUARE, "square"},
    {UnaryOpCode::EXP,    "exp"},
    {UnaryOpCode::CBRT,   "cbrt"},    // ← 新增
};

static const std::unordered_map<BinaryOpCode, const char*> binaryOpNames = {
    {BinaryOpCode::ADD, "add"},
    {BinaryOpCode::SUB, "sub"},
    {BinaryOpCode::MUL, "mul"},
    {BinaryOpCode::DIV, "div"},
    {BinaryOpCode::POW, "pow"},
    {BinaryOpCode::MAX, "max"},
    {BinaryOpCode::MIN, "min"},
    {BinaryOpCode::GT,  "gt"},
    {BinaryOpCode::LT,  "lt"},
    {BinaryOpCode::EQ,  "eq"},
    {BinaryOpCode::NEQ, "neq"},
    {BinaryOpCode::LOG_ADD, "log_add"},  // ← 新增
};

const char* unaryOpName(UnaryOpCode op) {
    auto it = unaryOpNames.find(op);
    return (it != unaryOpNames.end()) ? it->second : "unknown";
}

const char* binaryOpName(BinaryOpCode op) {
    auto it = binaryOpNames.find(op);
    return (it != binaryOpNames.end()) ? it->second : "unknown";
}

}  // namespace quantcore
```

---

## 步骤 5：更新 CMakeLists.txt

**文件**: `quantcore/CMakeLists.txt`

如果新增了 `.cpp` 文件，需要添加到 `target_sources`。纯头文件的算子不需要修改（当前设计算子均为 header-only 模板）。

---

## 步骤 6：编写测试

**文件**: `tests/unit/test_unary_ops.cpp`（一元算子测试模式）

```cpp
#include <gtest/gtest.h>
#include "quantcore/operators/unary/Cbrt.h"
#include "quantcore/storage/Column.h"

using namespace quantcore;

TEST(UnaryOpCbrt, BasicCorrectness) {
    Column<double> input({8.0, -8.0, 0.0, 27.0, 1.0});
    Column<double> expected({2.0, -2.0, 0.0, 3.0, 1.0});
    Column<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = CbrtOp::evaluateScalar(input[i]);
    }

    for (size_t i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(output[i], expected[i], 1e-12);
    }
}

TEST(UnaryOpCbrt, NullPropagation) {
    Column<double> input({8.0, 27.0, 64.0});
    input.setNull(2);  // 标记第3个元素为空
    Column<double> output(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        if (input.isNull(i)) {
            output.setNull(i);  // 空值传播
        } else {
            output[i] = CbrtOp::evaluateScalar(input[i]);
        }
    }

    EXPECT_FALSE(output.isNull(0));
    EXPECT_FALSE(output.isNull(1));
    EXPECT_TRUE(output.isNull(2));
    EXPECT_NEAR(output[0], 2.0, 1e-12);
    EXPECT_NEAR(output[1], 3.0, 1e-12);
}

TEST(UnaryOpCbrt, EdgeCases) {
    EXPECT_TRUE(std::isinf(CbrtOp::evaluateScalar(
        std::numeric_limits<double>::infinity())));
    EXPECT_TRUE(std::isnan(CbrtOp::evaluateScalar(
        std::numeric_limits<double>::quiet_NaN())));
}
```

---

## 步骤 7：添加性能基准测试

**文件**: `benchmarks/bench_unary_ops.cpp`

```cpp
#include <benchmark/benchmark.h>
#include "quantcore/operators/unary/Cbrt.h"

static void BM_CbrtOp_Scalar(benchmark::State& state) {
    const size_t n = state.range(0);
    std::vector<double> input(n, 8.0), output(n);
    for (auto _ : state) {
        for (size_t i = 0; i < n; ++i)
            output[i] = quantcore::CbrtOp::evaluateScalar(input[i]);
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_CbrtOp_Scalar)->Range(1024, 1<<20);
```

---

## 步骤 8：添加 Rolling 算子（滚动窗口算子）

**文件**: 在 `quantcore/core/Types.h` 的 `RollingOpCode` 枚举中添加成员，然后创建 `operators/rolling/Xxx.h`。

### 8a. Rolling 算子模板

**新建文件**: `quantcore/operators/rolling/Xxx.h`

```cpp
// quantcore/operators/rolling/Sma.h
#pragma once
#include "quantcore/core/Types.h"

namespace quantcore {

struct SmaOp {
    static constexpr RollingOpCode kOpCode = RollingOpCode::SMA;
    static constexpr const char* name = "sma";

    // 标量参考实现
    // @param input  输入数据指针
    // @param i      当前位置 (0-indexed)
    // @param window 窗口大小
    static double evaluateScalar(const double* input, std::size_t i,
                                 std::size_t window) noexcept {
        double sum = 0.0;
        for (std::size_t j = i + 1 - window; j <= i; ++j) {
            sum += input[j];
        }
        return sum / static_cast<double>(window);
    }

    // SIMD 内核（远期实现）
    template <SimdLevel L>
    static void evaluateSimd(const double* __restrict__ input,
                             double* __restrict__ output,
                             std::size_t n, std::size_t window,
                             const uint64_t* __restrict__ nullMask) noexcept;
};

}  // namespace quantcore
```

### 8b. Rolling 算子设计要点

| 要点 | 说明 |
|------|------|
| **窗口参数为运行时参数** | 不用模板参数，避免每个窗口大小都生成一份代码 |
| **前 window-1 个位置** | 标记为 null（数据不足以填满窗口） |
| **min_periods** | 窗口内最少有效观测数，不足则输出 null |
| **空值传播** | 窗口内 null 不计入有效观测，但不传播到输出 |

---

## 步骤 9：添加 CrossSection 算子（横截面算子）

**文件**: 在 `quantcore/core/Types.h` 的 `CrossSectionOpCode` 枚举中添加成员，然后创建 `operators/cross_section/Xxx.h`。

### 9a. CrossSection 算子模板

**新建文件**: `quantcore/operators/cross_section/Rank.h`

```cpp
// quantcore/operators/cross_section/Rank.h
#pragma once
#include "quantcore/core/Types.h"

namespace quantcore {

struct CrossSectionRankOp {
    static constexpr CrossSectionOpCode kOpCode = CrossSectionOpCode::RANK;
    static constexpr const char* name = "cs_rank";

    // 标量参考实现
    // @param assetValues  N 个资产的数值（同一时间点）
    // @param assetIndex   当前资产索引 (0..N-1)
    // @param numAssets    总资产数 N
    static double evaluateScalar(const double* assetValues,
                                 std::size_t assetIndex,
                                 std::size_t numAssets) noexcept;
};

}  // namespace quantcore
```

### 9b. CrossSection 算子设计要点

| 要点 | 说明 |
|------|------|
| **输入为多资产单时间点** | 每次调用传入 N 个资产在时间 t 的值 |
| **输出为 per-asset** | 每个资产得到独立的截面统计值 |
| **依赖 MarketDataBundle** | 远期由 MarketDataBundle 提供对齐的多资产数据 |
| **无法与逐元素算子深度融合** | 截面算子形成表达式 DAG 的边界 |
| **null 处理** | null 资产不参与截面统计计算（非传播） |

---

## 快速检查清单

### 一元/二元算子

- [ ] `Types.h` — `UnaryOpCode` 或 `BinaryOpCode` 枚举中添加（`kCount` 之前）
- [ ] `operators/unary/Xxx.h` 或 `operators/binary/Xxx.h` — 算子结构体实现
- [ ] `evaluateScalar()` — 标量参考实现（必须）
- [ ] `evaluateSimd()` — SIMD 内核（一期可用标量兜底）
- [ ] `registry/OperatorRegistry.cpp` — 名称字符串映射（`kUnaryOpNames` / `kBinaryOpNames` 数组）
- [ ] 单元测试 — 正确性 + 空值传播 + 边界值
- [ ] 性能基准 — Google Benchmark
- [ ] 遵循设计原则：纯逻辑无内存、空值传播语义、`__restrict__` 指针、`#pragma omp simd`

### Rolling 算子（远期）

- [ ] `Types.h` — `RollingOpCode` 枚举中添加（`kCount` 之前）
- [ ] `operators/rolling/Xxx.h` — 算子结构体（含 `window` 参数）
- [ ] `evaluateScalar(input, i, window)` — 标量参考实现
- [ ] `registry/OperatorRegistry.cpp` — `kRollingOpNames` 数组
- [ ] 前 window-1 位置 null 标记处理
- [ ] min_periods 校验

### CrossSection 算子（远期）

- [ ] `Types.h` — `CrossSectionOpCode` 枚举中添加（`kCount` 之前）
- [ ] `operators/cross_section/Xxx.h` — 算子结构体
- [ ] `evaluateScalar(assetValues, assetIndex, numAssets)` — 标量参考实现
- [ ] `registry/OperatorRegistry.cpp` — `kCrossSectionOpNames` 数组
- [ ] null 资产排除（不参与截面统计）

---

## 附：当前代码填充状态

| 文件 | 状态 |
|------|------|
| `core/Types.h` | ✅ 已实现 — Unary/Binary/Rolling/CrossSection 枚举 |
| `operators/UnaryOperator.h` | 🔧 空壳 — 需定义一元算子基类接口 |
| `operators/BinaryOperator.h` | 🔧 空壳 — 需定义二元算子基类接口 |
| `operators/RollingOperator.h` | ✅ 已定义 — CRTP 基类接口（远期） |
| `operators/CrossSectionOperator.h` | ✅ 已定义 — CRTP 基类接口（远期） |
| `operators/unary/*.h` (10个) | 🔧 空壳 — 需逐个实现 |
| `operators/binary/*.h` (8个) | 🔧 空壳 — 需逐个实现 |
| `operators/rolling/*.h` (7个) | ✅ 已搭建 — 远期预留占位 |
| `operators/cross_section/*.h` (3个) | ✅ 已搭建 — 远期预留占位 |
| `expression/*.h` (8个) | 🔧 空壳 — 含 RollingExpr/CrossSectionExpr 占位 |
| `registry/OperatorRegistry.*` | ✅ 已实现 — 含四种算子类型的注册映射 |
| `storage/MarketDataBundle.h` | ✅ 已搭建 — 远期预留存根 |
| `engine/ExecutionEngine.*` | 🔧 空壳 — 需实现融合调度 |
| `tests/unit/test_*_ops.cpp` | 🔧 空壳 — 需编写测试 |

实际已实现的模块：**Types.h**、**Column/ColView**、**MarketData**、**TimestampIndex**、**Logger**、**ErrorHandling**、**AlignedAllocator**、**ParquetReader**、**HDF5Reader**、**OperatorRegistry**。

建议实现顺序：先 `operators/` → 再 `expression/` → 再 `registry/` → 最后 `engine/`，每层写完就写对应测试。
