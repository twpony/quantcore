# 字符串因子表达式解析 — 就绪度评估

**日期**: 2026-07-27 (原始) / **2026-07-31 更新**

**状态**: ✅ **全部就绪** — 完整的 字符串→AST→Engine→Column<double> 流水线已实现并测试通过

---

## 一、目标流水线（已实现）

```
字符串 "ABS(LOG(CLOSE)-LOG(VWAP))*VOLUME"
         │
    [1]  Lexer/Tokenizer      — ✅ 手写 Lexer，支持数字/标识符/运算符/括号/逗号/$引用
         │
    [2]  Parser               — ✅ LL(1) recursive-descent，完整语法（expr→term→factor→primary）
         │
    [3]  Expression AST       — ✅ 8 种节点: ColumnRef, Scalar, UnaryExpr, BinaryExpr,
         │                          RollingExpr, BinaryRollingExpr, RedExpr, CsExpr
         │
    [4]  OperatorRegistry     — ✅ 62 算子全注册 name→enum→dispatch
         │
    [5]  FusedLoopGenerator   — ✅ 3 种融合模式: unary链, binary+chains, resultChain
         │
    [6]  ExecutionEngine      — ✅ 融合优先→fallback 后序求值 + BufferPool 64B对齐
         │
    [7]  FactorCalculator     — ✅ 高层 API: 公式注册/求值/缓存/截面/自定义算子
         │
         ▼
   Column<double> result
```

---

## 二、当前实现状态（已完成 ✅）

| 层次 | 文件 | 说明 |
|------|------|------|
| 存储层 | `Column.h`, `ColView.h`, `MarketData.h`, `TimestampIndex.h`, `PanelData.h`, `IntermediateColumn.h` | 列式存储、面板数据、截面工作区 |
| 表达式层 | `ExprNode.h`, `UnaryExpr.h`, `BinaryExpr.h`, `ColumnRef.h`, `Scalar.h`, `RollingExpr.h`, `BinaryRollingExpr.h`, `RedExpr.h`, `CsExpr.h`, `ExprTraits.h` | 全 AST 节点 + 类型萃取 |
| 融合优化 | `FusedLoopGenerator.h` | 单循环融合求值，纯 Unary 链/Binary+Unary链/resultChain |
| 词法/语法 | `Lexer.h`, `Token.h`, `Parser.h` | 手写 LL(1) + 完整函数调用语法 |
| 一元算子 | `unary/` (12个) | Abs, Log, Log10, Log2, Sqrt, Exp, Neg, Sign, Square, Inv, Not, Rank 系列 |
| 二元算子 | `binary/` (10个) | Add, Sub, Mul, Div, Max, Min, Gt, Lt, Eq, Neq |
| 滚动算子 | `rolling/` (20个) | Sma, Ema, Mean, Var, Std, Max, Min, Median, Sum, Mul, Quantile, ArgMax, ArgMin, Rank, Kurt, Skew, Diff, Shift, Corr, Cov |
| 聚合算子 | `red/` (10个) | Sum, Mean, Std, Var, Min, Max, Mul, Median, ZScore, Quantile |
| 截面算子 | `cs/` (10个) | Rank, Quantile, ZScore, Normalize, NormalizeL1, NormalizeL2, Winsorize, WinsorizeMAD, Clip, Demean |
| 算子注册 | `OperatorRegistry.h/.cpp` | 全 dispatch: unary/binary/rolling/binaryRolling/red/cs + 自定义算子槽位 |
| 引擎 | `ExecutionEngine.h/.cpp`, `BufferPool.h/.cpp`, `BufferHandle.h`, `EngineMetrics.h` | 融合评估 + 池化分配 |
| 高层 API | `FactorCalculator.h/.cpp` | 公式注册/求值/缓存/截面/$ref/自定义算子 |
| 数据 I/O | `ParquetReader`, `HDF5Reader` | Parquet/HDF5 日线数据读取 |
| 因子 | `alpha_0001`-`alpha_0005`, `alpha_1001` | 示例因子 + AOT 编译器 |
| 测试 | 67 个 ctest 目标, 100% pass | 全覆盖：算子、表达式、引擎、解析器、因子计算器、面板数据、黄金值回归 |
| 基准测试 | `bench_*.cpp` (5个) | 一元/二元/融合/缓冲池/SMA20 性能基准 |

---

## 三、高层 API 使用示例

```cpp
#include <quantcore/core/FactorCalculator.h>

FactorCalculator calc;

// 注册因子公式
calc.registerFormula("momentum",   "close / rolling_mean(close, 60) - 1");
calc.registerFormula("volatility", "rolling_std(log(close) - log(rolling_shift(close, 1)), 20)");
calc.registerFormula("log_return", "log(close) - log(rolling_shift(close, 1))");

// 注册自定义算子
calc.registry().registerCustomUnary("triple", UnaryOpCode::CUSTOM_0,
    [](double x) noexcept { return x * 3.0; });

// 一行求值
Column<double> mom = calc.evaluate("momentum", marketData);

// 截面求值（多资产面板数据）
PanelData panel = ...;
auto csResults = calc.evaluateCSExpression(
    "log(volume)", panel, CsOpCode::CS_RANK);
```

---

## 四、四阶段回顾（全部完成）

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | 表达式 AST 节点 (8 种) | ✅ |
| 2 | ExecutionEngine + BufferPool | ✅ |
| 3 | Lexer + Parser | ✅ |
| 4 | FusedLoopGenerator + RollingQuantile dispatch | ✅ |
| 5 | FactorCalculator + 自定义算子 + 截面 | ✅ |

---

## 五、当前限制

1. **CsvReader / BinaryWriter / BinaryReader** — 空桩 (0 bytes)，尚未实现
2. **SimdDispatcher** — 空桩，SIMD 调度框架尚未实现
3. **线程模型** — BufferPool/Engine 显式单线程，并行需每线程一个 Engine
4. **空值处理差异** — rolling 算子用 NaN 标记边界，表达式节点返回 null mask
