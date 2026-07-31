# QuantCore 添加算子指南

**更新日期**: 2026-07-31

**状态**: 核心流水线已完整实现。按此指南添加新算子即可直接用于字符串因子表达式。

---

## 整体架构

```
添加一个算子需要触及 5 层：

  Types.h (枚举注册)
     ↓
  operators/unary/Xxx.h 或 operators/binary/Xxx.h (算子实现)
     ↓
  registry/OperatorRegistry.h/.cpp (名称→枚举映射 + dispatch 注册)
     ↓
  tests/op/unary/ 或 tests/op/binary/ (测试)
     ↓
  benchmarks/ (性能基准)
```

**注意**: 表达式层（UnaryExpr/BinaryExpr）**无需修改** — 它们通过 OpCode 分发，自动支持所有已注册的算子。

---

## 步骤 1：在 `Types.h` 中注册算子枚举

**文件**: `quantcore/core/Types.h`

**一元算子** — 在 `UnaryOpCode` 枚举中添加新成员（在 `kCount` 之前）：

```cpp
enum class UnaryOpCode : uint8_t {
    ABS, LOG, LOG10, LOG2, SQRT, NEG, SIGN, SQUARE, EXP, INV, NOT,
    RANK, RANK_PCT, RANK_NORMALIZED,
    kCount,
    CUSTOM_0, CUSTOM_1, ..., CUSTOM_7,  // 用户自定义算子槽位
};
```

**二元算子** — 在 `BinaryOpCode` 枚举中添加：

```cpp
enum class BinaryOpCode : uint8_t {
    ADD, SUB, MUL, DIV, MAX, MIN, GT, LT, EQ, NEQ,
    kCount,
    CUSTOM_0, CUSTOM_1, CUSTOM_2, CUSTOM_3,  // 用户自定义算子槽位
};
```

---

## 步骤 2：实现算子类

### 2a. 一元算子

**新建文件**: `quantcore/operators/unary/MyOp.h`

```cpp
#pragma once

#include <cmath>
#include "quantcore/core/Types.h"
#include "quantcore/operators/UnaryOperator.h"

namespace quantcore {

struct MyOp : public UnaryOperator<MyOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::CUSTOM_0;
    static constexpr const char* name = "my_op";

    // 标量实现 — 必须提供，用于融合循环和 SIMD 交叉验证
    static double evaluateScalar(double x) noexcept {
        return x * 2.0;  // 示例: 将输入翻倍
    }
};

}  // namespace quantcore
```

**关键要求**:
- 继承 `UnaryOperator<MyOp>`（CRTP 模板提供列式求值、空值处理）
- 提供 `static constexpr UnaryOpCode kOpCode`
- 提供 `static double evaluateScalar(double x) noexcept`
- 可选: 覆盖 `evaluateSimd<SimdLevel>()` 提供 SIMD 加速

### 2b. 二元算子

**新建文件**: `quantcore/operators/binary/MyBinaryOp.h`

```cpp
#pragma once

#include <cmath>
#include "quantcore/core/Types.h"
#include "quantcore/operators/BinaryOperator.h"

namespace quantcore {

struct MyBinaryOp : public BinaryOperator<MyBinaryOp> {
    static constexpr BinaryOpCode kOpCode = BinaryOpCode::CUSTOM_0;
    static constexpr const char* name = "my_binary";

    static double evaluateScalar(double a, double b) noexcept {
        return a * a + b * b;  // 示例: 平方和
    }
};

}  // namespace quantcore
```

---

## 步骤 3：注册到 OperatorRegistry

**文件**: `quantcore/registry/OperatorRegistry.cpp`

在对应的注册 block 中添加（在 `OperatorRegistry::OperatorRegistry()` 构造函数内）：

```cpp
// Unary
registry.registerUnary<MyOp>("my_op", UnaryOpCode::CUSTOM_0);

// Binary
registry.registerBinary<MyBinaryOp>("my_binary", BinaryOpCode::CUSTOM_0);
```

**注册即完成** — `registerUnary`/`registerBinary` 同时注册：
1. name → enum 映射（用于 Parser 解析字符串表达式）
2. dispatch lambda（用于 UnaryExpr/BinaryExpr 的列式求值）
3. scalar function pointer（用于 FusedLoopGenerator 的融合循环）

**使用内置枚举值 vs 自定义槽位**:
- 内置枚举值（如 `UnaryOpCode::ABS`）: 直接使用
- 自定义槽位（`CUSTOM_0`..`CUSTOM_7`）: 用于 `registerCustomUnary`/`registerCustomBinary` 公开 API

---

## 步骤 4：编写测试

**文件**: `tests/op/unary/test_my_op.cpp`

```cpp
#include <gtest/gtest.h>

#include "quantcore/core/Types.h"
#include "quantcore/registry/OperatorRegistry.h"
#include "quantcore/storage/Column.h"

using namespace quantcore;

TEST(MyOpTest, BasicCorrectness) {
    auto& reg = OperatorRegistry::instance();

    constexpr std::size_t kN = 5;
    double input[kN]  = {1.0, 2.0, 3.0, 4.0, 5.0};
    double output[kN] = {};
    Operand op(input);

    reg.invokeUnary("my_op", op, output, kN, nullptr);

    EXPECT_DOUBLE_EQ(output[0], 2.0);
    EXPECT_DOUBLE_EQ(output[1], 4.0);
    EXPECT_DOUBLE_EQ(output[2], 6.0);
}

TEST(MyOpTest, ViaExpressionEngine) {
    // 测试通过字符串表达式 + Engine 的完整流水线
    // ...
}
```

然后在 `tests/CMakeLists.txt` 中添加：
```cmake
add_executable(test_op_my_op op/unary/test_my_op.cpp)
```
并将 `test_op_my_op` 添加到 `TEST_TARGETS` 列表。

---

## 步骤 5：添加性能基准测试（可选）

**文件**: `benchmarks/bench_unary_ops.cpp`（或新文件）

```cpp
#include <benchmark/benchmark.h>

static void BM_MyOp(benchmark::State& state) {
    std::size_t n = state.range(0);
    std::vector<double> input(n, 3.0), output(n);
    Operand op(input.data());
    auto& reg = OperatorRegistry::instance();

    for (auto _ : state) {
        reg.invokeUnary("my_op", op, output.data(), n, nullptr);
        benchmark::DoNotOptimize(output.data());
    }
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK(BM_MyOp)->Range(1<<10, 1<<20);
```

---

## 当前项目填充状态（2026-07-31 更新）

| 文件 | 状态 |
|------|------|
| `core/Types.h` | ✅ 已实现 — Unary/Binary/Rolling/Red/Cs 枚举含自定义槽位 |
| `operators/unary/*.h` (12个) | ✅ 全部实现 |
| `operators/binary/*.h` (10个) | ✅ 全部实现 |
| `operators/rolling/*.h` (20个) | ✅ 全部实现 |
| `operators/red/*.h` (10个) | ✅ 全部实现 |
| `operators/cs/*.h` (10个) | ✅ 全部实现 |
| `expression/*.h` (12个) | ✅ 全部实现（ExprNode, UnaryExpr, BinaryExpr, ColumnRef, Scalar, RollingExpr, BinaryRollingExpr, RedExpr, CsExpr, ExprTraits, FusedLoopGenerator, Lexer/Parser/Token） |
| `registry/OperatorRegistry.*` | ✅ 全部实现 — 5 族 dispatch + 自定义算子 API |
| `engine/*` | ✅ 全部实现 — ExecutionEngine, BufferPool, BufferHandle, EngineMetrics, CrossSectionWorkspace |
| `core/FactorCalculator.*` | ✅ 全部实现 — 高层 API |
| `storage/PanelData.*`, `IntermediateColumn.*` | ✅ 全部实现 |
| `io/ParquetReader.*`, `HDF5Reader.*` | ✅ 已实现 |
| `io/CsvReader.*`, `BinaryWriter.*`, `BinaryReader.*` | ⚠️ 空桩 (0 bytes) — 尚未实现 |
| `simd/SimdDispatcher.*`, `SimdTraits.h` | ⚠️ 空桩 — SIMD 调度框架尚未实现 |
| 测试 | ✅ 67 个 ctest 目标, 100% pass |
| 基准测试 | ✅ 5 个 benchmark 文件 |

---

## 快速检查清单

### 一元/二元算子

- [ ] `Types.h` — 在对应枚举中添加（`kCount` 之前）
- [ ] `operators/unary/Xxx.h` 或 `operators/binary/Xxx.h` — 算子结构体，继承 CRTP 基类
- [ ] `evaluateScalar()` — 标量实现（必须）
- [ ] `kOpCode` / `name` — 静态常量（必须）
- [ ] `OperatorRegistry.cpp` — 注册 `registerUnary<Op>()` 或 `registerBinary<Op>()`
- [ ] 单元测试 — `tests/op/` 目录
- [ ] 更新 `tests/CMakeLists.txt` — 添加测试目标
