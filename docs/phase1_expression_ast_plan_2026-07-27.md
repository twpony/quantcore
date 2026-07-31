# Phase 1 详细实现方案：表达式 AST

**日期**: 2026-07-27

**状态**: ✅ 已完成 — 8 种 AST 节点全部实现，61 算子全注册 dispatch，所有测试通过

---

## 零、现状总结

### 已完成

| 文件 | 状态 | 行数 |
|------|------|------|
| `ExprNode.h` | 完成 — 虚基类 `evaluate()` + `clone()` | 60 |
| `ColumnRef.h` | 完成 — 引用 MarketData 字段的叶节点 | 59 |
| `Scalar.h` | 完成 — double 常量叶节点 | 51 |
| `UnaryExpr.h` | 完成 — 一元算子节点（in-place 求值） | 68 |
| `BinaryExpr.h` | 完成 — 二元算子节点（含 null mask 合并） | 115 |
| `RollingExpr.h` | 完成 — 滚动窗口算子节点 | 90 |
| `OperatorRegistry.h/.cpp` | unary/binary/rolling 全 dispatch 完成；Red/Cs 仅 name 注册 | 535 |
| `test_expression.cpp` | 完成 — 25 个测试用例覆盖全部已实现节点 | 661 |

**总代码量 (已完成): ~1,640 行**

### 待完成

| # | 任务 | 优先级 | 预估行数 |
|---|------|--------|---------|
| 1 | **ExprTraits.h** — 类型萃取辅助 | P0 | ~35 |
| 2 | **RedExpr.h** — 截面归约表达式节点 | P0 | ~75 |
| 3 | **CsExpr.h** — 截面变换表达式节点 | P0 | ~75 |
| 4 | **Red/Cs dispatch 注册** — OperatorRegistry 补充 | P0 | ~130 |
| 5 | **参数化算子支持** — dispatch lambda 中透传 extraParams | P0 | ~40 |
| 6 | **ExprNode::dump()** — AST 树状打印调试工具 | P2 | ~60 |
| 7 | **测试补齐** — RedExpr/CsExpr/参数化算子测试 | P0 | ~200 |
| **合计** | | | **~615** |

---

## 一、核心设计决策：统一 dispatch 签名

### 问题

Red/Cs 算子接口不统一：

| 算子类别 | evaluate 签名 | 例子 |
|----------|--------------|------|
| 无参 Red | `evaluate(ColView, double*)` (CRTP基类) | Sum, Mean, Std, ZScore... |
| 无参 CS | `evaluate(ColView, double*)` (CRTP基类) | Rank, Quantile, Normalize... |
| 1参 CS | `evaluate(ColView, double*, double)` | CsWinsorizeMADOp |
| 2参 CS | `evaluate(ColView, double*, double, double)` | CsWinsorizeOp, CsClipOp |

### 方案

统一使用 `std::vector<double>` 传递额外参数。修改函数指针类型：

```cpp
// 旧（OperatorRegistry.h 当前）
using RedEvalFn = void (*)(ColView<double>, double*);
using CsEvalFn  = void (*)(ColView<double>, double*);

// 新
using RedEvalFn = void (*)(ColView<double>, double*, const std::vector<double>&);
using CsEvalFn  = void (*)(ColView<double>, double*, const std::vector<double>&);
```

无参算子的 lambda 忽略第三个参数即可。此方案避免了维护两套 dispatch 路径的复杂度。

---

## 二、逐文件实施细节

### Step 1: 修改 `OperatorRegistry.h` — 函数指针类型 + 模板方法

**变更位置**: 第 36-38 行，函数指针类型定义

```cpp
// 旧
using RedEvalFn = void (*)(ColView<double>, double*);
using CsEvalFn  = void (*)(ColView<double>, double*);

// 新
using RedEvalFn = void (*)(ColView<double>, double*, const std::vector<double>&);
using CsEvalFn  = void (*)(ColView<double>, double*, const std::vector<double>&);
```

**新增**: `registerRedDispatch` 模板方法（约第 183 行 `invokeRed` 声明之前）

```cpp
// 无参 Red 算子 dispatch 注册
template <typename OpType>
void registerRedDispatch(RedOpCode code) {
    auto idx = static_cast<std::size_t>(code);
    if (idx >= redDispatch_.size()) {
        redDispatch_.resize(idx + 1);
    }
    redDispatch_[idx] = [](ColView<double> input, double* output,
                           const std::vector<double>& /*params*/) {
        OpType op;
        op.evaluate(input, output);
    };
}
```

**新增**: `registerCsDispatch` 模板方法（约第 193 行 `invokeCs` 声明之前）

```cpp
// 无参 CS 算子 dispatch 注册
template <typename OpType>
void registerCsDispatch(CsOpCode code) {
    auto idx = static_cast<std::size_t>(code);
    if (idx >= csDispatch_.size()) {
        csDispatch_.resize(idx + 1);
    }
    csDispatch_[idx] = [](ColView<double> input, double* output,
                          const std::vector<double>& /*params*/) {
        OpType op;
        op.evaluate(input, output);
    };
}
```

**新增**: `registerRedDispatchRaw` / `registerCsDispatchRaw` — 用于参数化算子的手写 lambda 注册

```cpp
// 在 public 区域新增
void registerRedDispatchRaw(RedOpCode code, RedEvalFn fn) {
    auto idx = static_cast<std::size_t>(code);
    if (idx >= redDispatch_.size()) redDispatch_.resize(idx + 1);
    redDispatch_[idx] = fn;
}

void registerCsDispatchRaw(CsOpCode code, CsEvalFn fn) {
    auto idx = static_cast<std::size_t>(code);
    if (idx >= csDispatch_.size()) csDispatch_.resize(idx + 1);
    csDispatch_[idx] = fn;
}
```

**修改**: `invokeRed` / `invokeCs` 声明 — 添加 `const std::vector<double>&` 参数

```cpp
// 旧
void invokeRed(RedOpCode code, ColView<double> values, double* output) const;
void invokeCs(CsOpCode code, ColView<double> values, double* output) const;

// 新
void invokeRed(RedOpCode code, ColView<double> values, double* output,
               const std::vector<double>& extraParams = {}) const;
void invokeCs(CsOpCode code, ColView<double> values, double* output,
              const std::vector<double>& extraParams = {}) const;
```

---

### Step 2: 修改 `OperatorRegistry.cpp` — 注册所有 Red/Cs dispatch

**2.1 修改 `invokeRed` 实现**（第 504-515 行）

```cpp
void OperatorRegistry::invokeRed(RedOpCode code,
                                 ColView<double> values,
                                 double* output,
                                 const std::vector<double>& extraParams) const
{
    auto idx = static_cast<std::size_t>(code);
    if (idx >= redDispatch_.size() || redDispatch_[idx] == nullptr) {
        throw std::runtime_error(
            "Red operator not registered for dispatch: "
            + std::string(redOpName(code)));
    }
    redDispatch_[idx](values, output, extraParams);
}
```

**2.2 修改 `invokeCs` 实现**（第 521-531 行）

```cpp
void OperatorRegistry::invokeCs(CsOpCode code,
                                ColView<double> values,
                                double* output,
                                const std::vector<double>& extraParams) const
{
    auto idx = static_cast<std::size_t>(code);
    if (idx >= csDispatch_.size() || csDispatch_[idx] == nullptr) {
        throw std::runtime_error(
            "CS operator not registered for dispatch: "
            + std::string(csOpName(code)));
    }
    csDispatch_[idx](values, output, extraParams);
}
```

**2.3 在 `instance()` 初始化中注册所有 Red dispatch**（在 `// -- Red operators --` 注释块中）

```cpp
// -- Red operators dispatch --
registry.registerRedDispatch<RedSumOp>     (RedOpCode::RED_SUM);
registry.registerRedDispatch<RedMeanOp>    (RedOpCode::RED_MEAN);
registry.registerRedDispatch<RedStdOp>     (RedOpCode::RED_STD);
registry.registerRedDispatch<RedVarOp>     (RedOpCode::RED_VAR);
registry.registerRedDispatch<RedMinOp>     (RedOpCode::RED_MIN);
registry.registerRedDispatch<RedMaxOp>     (RedOpCode::RED_MAX);
registry.registerRedDispatch<RedMulOp>     (RedOpCode::RED_MUL);
registry.registerRedDispatch<RedMedianOp>  (RedOpCode::RED_MEDIAN);
registry.registerRedDispatch<RedZScoreOp>  (RedOpCode::RED_ZSCORE);
registry.registerRedDispatch<RedQuantileOp>(RedOpCode::RED_QUANTILE);
```

**2.4 在 `instance()` 初始化中注册所有 CS dispatch**（在 `// -- CS operators --` 注释块中）

```cpp
// -- CS operators dispatch (无参) --
registry.registerCsDispatch<CsRankOp>        (CsOpCode::CS_RANK);
registry.registerCsDispatch<CsQuantileOp>    (CsOpCode::CS_QUANTILE);
registry.registerCsDispatch<CsZScoreOp>      (CsOpCode::CS_ZSCORE);
registry.registerCsDispatch<CsNormalizeOp>   (CsOpCode::CS_NORMALIZE);
registry.registerCsDispatch<CsNormalizeL1Op> (CsOpCode::CS_NORMALIZE_L1);
registry.registerCsDispatch<CsNormalizeL2Op> (CsOpCode::CS_NORMALIZE_L2);
registry.registerCsDispatch<CsDemeanOp>      (CsOpCode::CS_DEMEAN);

// -- CS operators dispatch (参数化，手写 lambda) --
// CS_WINSORIZE: params = {lowerPct, upperPct}
registry.registerCsDispatchRaw(CsOpCode::CS_WINSORIZE,
    [](ColView<double> input, double* output, const std::vector<double>& params) {
        CsWinsorizeOp op;
        op.evaluate(input, output, params.at(0), params.at(1));
    });

// CS_WINSORIZE_MAD: params = {n}
registry.registerCsDispatchRaw(CsOpCode::CS_WINSORIZE_MAD,
    [](ColView<double> input, double* output, const std::vector<double>& params) {
        CsWinsorizeMADOp op;
        op.evaluate(input, output, params.at(0));
    });

// CS_CLIP: params = {lower, upper}
registry.registerCsDispatchRaw(CsOpCode::CS_CLIP,
    [](ColView<double> input, double* output, const std::vector<double>& params) {
        CsClipOp op;
        op.evaluate(input, output, params.at(0), params.at(1));
    });
```

---

### Step 3: 新建 `RedExpr.h`

```cpp
// RedExpr.h — expression node for cross-section reduction operators
// Phase: 一期必实现
//
// RedExpr applies a cross-section reduction operator (SUM, MEAN, STD, ZSCORE,
// QUANTILE, ...) to its child expression.  Red operators are "fusion boundaries"
// — their child must be fully materialized before the cross-sectional computation
// can run, because every output element may depend on every input element.
//
// Dispatch is via OperatorRegistry::invokeRed, which uses the same type-erased
// function pointer pattern as invokeUnary / invokeBinary.
//
// extraParams_ stores per-call parameters for operators like RED_QUANTILE(q).
// For stateless operators the vector is empty and ignored.
#pragma once

#include <memory>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/registry/OperatorRegistry.h"
#include "quantcore/storage/ColView.h"

namespace quantcore {

class RedExpr : public ExprNode {
public:
    RedExpr(RedOpCode op,
            std::unique_ptr<ExprNode> child,
            std::vector<double> extraParams = {})
        : op_(op), child_(std::move(child)), extraParams_(std::move(extraParams)) {}

    // ============================================================
    // Accessors
    // ============================================================

    RedOpCode opCode() const noexcept { return op_; }
    const ExprNode* child() const noexcept { return child_.get(); }
    const std::vector<double>& extraParams() const noexcept { return extraParams_; }

    // ============================================================
    // ExprNode interface
    // ============================================================

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<RedExpr>(op_, child_->clone(), extraParams_);
    }

    const uint64_t* evaluate(const MarketData& md,
                             double* output,
                             std::size_t n) const override {
        // 1. Materialize child expression into a temporary buffer.
        //    Red ops need the full input array before computing.
        childBuf_.resize(n);
        const uint64_t* childNull = child_->evaluate(md, childBuf_.data(), n);

        // 2. Build a ColView wrapping the child's result.
        ColView<double> inputView(childBuf_.data(), n, childNull);

        // 3. Dispatch via OperatorRegistry::invokeRed.
        auto& reg = OperatorRegistry::instance();
        reg.invokeRed(op_, inputView, output, extraParams_);

        return childNull;
    }

private:
    RedOpCode op_;
    std::unique_ptr<ExprNode> child_;
    std::vector<double> extraParams_;

    // Mutable buffer for the materialized child result.
    // NOT thread-safe — use clone() for concurrent evaluation.
    mutable std::vector<double> childBuf_;
};

}  // namespace quantcore
```

---

### Step 4: 新建 `CsExpr.h`

结构与 RedExpr 完全对称：

```cpp
// CsExpr.h — expression node for cross-section transform operators
// Phase: 一期必实现
//
// CsExpr applies a cross-section transform operator (RANK, ZSCORE, NORMALIZE,
// WINSORIZE, CLIP, ...) to its child expression.  Cs operators are "fusion
// boundaries" — their child must be fully materialized before the cross-sectional
// computation can run.
//
// Dispatch is via OperatorRegistry::invokeCs.
//
// extraParams_ stores per-call parameters for operators like CS_WINSORIZE(lowerPct,
// upperPct), CS_WINSORIZE_MAD(n), and CS_CLIP(lower, upper).  For stateless
// operators the vector is empty and ignored.
#pragma once

#include <memory>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/registry/OperatorRegistry.h"
#include "quantcore/storage/ColView.h"

namespace quantcore {

class CsExpr : public ExprNode {
public:
    CsExpr(CsOpCode op,
           std::unique_ptr<ExprNode> child,
           std::vector<double> extraParams = {})
        : op_(op), child_(std::move(child)), extraParams_(std::move(extraParams)) {}

    // ============================================================
    // Accessors
    // ============================================================

    CsOpCode opCode() const noexcept { return op_; }
    const ExprNode* child() const noexcept { return child_.get(); }
    const std::vector<double>& extraParams() const noexcept { return extraParams_; }

    // ============================================================
    // ExprNode interface
    // ============================================================

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<CsExpr>(op_, child_->clone(), extraParams_);
    }

    const uint64_t* evaluate(const MarketData& md,
                             double* output,
                             std::size_t n) const override {
        // 1. Materialize child to temp buffer.
        childBuf_.resize(n);
        const uint64_t* childNull = child_->evaluate(md, childBuf_.data(), n);

        // 2. Build ColView.
        ColView<double> inputView(childBuf_.data(), n, childNull);

        // 3. Dispatch.
        auto& reg = OperatorRegistry::instance();
        reg.invokeCs(op_, inputView, output, extraParams_);

        return childNull;
    }

private:
    CsOpCode op_;
    std::unique_ptr<ExprNode> child_;
    std::vector<double> extraParams_;

    // Mutable buffer for the materialized child result.
    // NOT thread-safe — use clone() for concurrent evaluation.
    mutable std::vector<double> childBuf_;
};

}  // namespace quantcore
```

---

### Step 5: 新建 `ExprTraits.h`

```cpp
// ExprTraits.h — compile-time type traits for expression nodes
// Phase: 一期必实现
//
// Provides runtime helper functions for traversing and analyzing expression trees.
// Phase 4 (FusedLoopGenerator) will extend this with compile-time fusion analysis.
#pragma once

#include <cstddef>
#include <ostream>
#include <string>

namespace quantcore {

class ExprNode;

// ============================================================
// Tree analysis helpers
// ============================================================

/// Count the total number of nodes in an expression tree.
std::size_t exprNodeCount(const ExprNode* node);

/// Compute the maximum depth of an expression tree (leaf = depth 0).
std::size_t exprDepth(const ExprNode* node);

/// Pretty-print an expression tree to an ostream.
void exprDump(const ExprNode* node, std::ostream& os, int indent = 0);

/// Return a string representation of the expression tree.
std::string exprToString(const ExprNode* node);

}  // namespace quantcore
```

Phase 1 的 ExprTraits 保持精简。`exprNodeCount()` 和 `exprDepth()` 需要在各节点类中通过虚函数或 visitor 实现，主要在 Phase 2 ExecutionEngine 中使用。

---

### Step 6: 添加 `dump()` 调试方法到 ExprNode

在 `ExprNode.h` 基类中添加 virtual dump：

```cpp
// ExprNode.h 中新增
#include <iosfwd>
#include <string>

virtual void dump(std::ostream& os, int indent = 0) const {
    os << std::string(indent, ' ') << "ExprNode\n";
}
```

各子类 override：
- `ColumnRef::dump()` → `"COLUMN(close)"`
- `Scalar::dump()` → `"SCALAR(3.14)"`
- `UnaryExpr::dump()` → `"LOG"` + child dump
- `BinaryExpr::dump()` → `"ADD"` + lhs dump + rhs dump
- `RollingExpr::dump()` → `"ROLLING_MEAN(5)"` + child dump
- `RedExpr::dump()` → `"RED_SUM"` + child dump
- `CsExpr::dump()` → `"CS_ZSCORE"` + child dump

---

### Step 7: 测试补齐

在 `test_expression.cpp` 中新增测试用例。使用与现有测试相同的 fixture（`ExpressionTest`，kN=20），但 Red/Cs 算子操作的是**单资产的时间序列**（在测试场景下退化为逐元素计算）。

**重要说明**: Red/Cs 算子的设计原意是跨资产截面计算，但在单资产时间序列场景下仍然可以求值——它们对 N 个数据点做归约/变换。Phase 1 测试可以在时间序列维度上验证正确性。

新增约 18 个测试用例：

```cpp
// === RedExpr tests ===
TEST_F(ExpressionTest, RedSumBasic)      { /* RED_SUM(CLOSE) → 所有 CLOSE 值之和广播 */ }
TEST_F(ExpressionTest, RedMeanBasic)     { /* RED_MEAN(CLOSE) → 均值广播到每个位置 */ }
TEST_F(ExpressionTest, RedStdBasic)      { /* RED_STD(CLOSE) */ }
TEST_F(ExpressionTest, RedZScore)        { /* RED_ZSCORE(CLOSE) → (x-mean)/std */ }
TEST_F(ExpressionTest, RedClone)         { /* clone + re-evaluate */ }

// === CsExpr tests ===
TEST_F(ExpressionTest, CsRankBasic)      { /* CS_RANK(CLOSE) */ }
TEST_F(ExpressionTest, CsZScoreBasic)    { /* CS_ZSCORE(CLOSE) */ }
TEST_F(ExpressionTest, CsNormalizeBasic) { /* CS_NORMALIZE(CLOSE) → [0, 1] */ }
TEST_F(ExpressionTest, CsDemeanBasic)    { /* CS_DEMEAN(CLOSE) → x - mean */ }

// === Parameterized CsExpr tests ===
TEST_F(ExpressionTest, CsWinsorize)      { /* CS_WINSORIZE(CLOSE, 0.1, 0.9) */ }
TEST_F(ExpressionTest, CsClip)           { /* CS_CLIP(CLOSE, 12.0, 20.0) */ }

// === Mixed expressions ===
TEST_F(ExpressionTest, CompositeRedWithUnary)  { /* RED_ZSCORE(LOG(CLOSE)) */ }
TEST_F(ExpressionTest, CompositeCsWithBinary)  { /* CS_NORMALIZE(HIGH - LOW) */ }

// === Clone + re-evaluate on different data ===
TEST_F(ExpressionTest, RedExprCloneDifferentData) { /* ... */ }
TEST_F(ExpressionTest, CsExprCloneDifferentData)  { /* ... */ }

// === Edge cases ===
TEST_F(ExpressionTest, RedExprWithNulls)       { /* null propagation */ }
TEST_F(ExpressionTest, CsExprEmptyInput)       { /* zero-size */ }
TEST_F(ExpressionTest, CsExprSingleElement)    { /* single element */ }
```

---

## 三、实施顺序

```
Step 1: OperatorRegistry.h 修改
        ├── 改 RedEvalFn / CsEvalFn 签名
        ├── 加 registerRedDispatch / registerCsDispatch 模板
        ├── 加 registerRedDispatchRaw / registerCsDispatchRaw
        └── 改 invokeRed / invokeCs 声明
                              ↓
Step 2: OperatorRegistry.cpp 修改
        ├── 改 invokeRed / invokeCs 实现
        └── instance() 中注册所有 Red/Cs dispatch
                              ↓
        ┌─────────────────────┤
        ↓                     ↓
Step 3: RedExpr.h       Step 4: CsExpr.h
        ↓                     ↓
Step 5: ExprTraits.h (独立，可与 3/4 并行)
                              ↓
Step 6: ExprNode::dump() (依赖 3/4)
                              ↓
Step 7: test_expression.cpp 新增测试 (依赖 2-6)
```

**预计新增/修改代码量**:

| 操作 | 文件 | 行数 |
|------|------|------|
| 修改 | `OperatorRegistry.h` | +50 |
| 修改 | `OperatorRegistry.cpp` | +80 |
| **新建** | `RedExpr.h` | ~75 |
| **新建** | `CsExpr.h` | ~75 |
| **新建** | `ExprTraits.h` | ~35 |
| 修改 | `ExprNode.h` (加 dump) | +10 |
| 修改 | 各子类 (加 dump override) | +50 |
| 修改 | `test_expression.cpp` | +200 |
| **合计** | | **~575 行** |

---

## 四、风险点

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| `csDispatch_` / `redDispatch_` 是 private 成员，`instance()` 无法直接访问 | 中 | 新增 `registerXxxDispatchRaw` public 方法 |
| CsWinsorizeMADOp::evaluateScalar 是非 static 成员 | 低 | dispatch lambda 中构造算子实例后调用，已验证可行 |
| Red/Cs 在单资产时间序列上的语义可能与跨资产截面不同 | 低 | Phase 1 先在时间序列维度验证正确性；真正的跨资产用法留待 ExecutionEngine 实现 |
| `invokeRed`/`invokeCs` 签名变更可能影响 RollingExpr | 无 | RollingExpr 使用 `invokeRolling`，不受影响 |

---

## 五、与后续 Phase 的关系

| Phase | 对 Phase 1 的依赖 |
|-------|-------------------|
| Phase 2 (ExecutionEngine + BufferPool) | 需要完整的 AST 节点体系，通过 `evaluate(md, output, n)` 统一接口调用 |
| Phase 3 (Lexer + Parser) | 需要 OperatorRegistry 的 name→enum 查找（Red/Cs 的 findRed/findCs 已就绪）|
| Phase 4 (Registry 补全 + 融合优化) | 需要 ExprTraits 的融合边界判断；参数化算子的 dispatch 路径已就绪 |
