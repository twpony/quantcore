# Phase 4 详细设计方案：补全与融合优化

**日期**: 2026-07-27

**状态**: RollingQuantileOp 未注册 dispatch；FusedLoopGenerator.h 为空；其余全部就绪

---

## 零、现状核查

### 4.1 原文档 Phase 4 任务

| 原计划任务 | 实际状态 |
|-----------|---------|
| 补全 OperatorRegistry 的 rolling/red/cs dispatch | ✅ **已在 Phase 1 完成** — 全部 61 个算子均已注册 dispatch |
| Rank 算子统一接口适配 | ✅ **已兼容** — RankOp/RankPctOp/RankNormalizedOp 均提供 `evaluate(Operand&, ...)` 重载，与 `invokeUnary` dispatch 兼容 |
| 算子融合优化（FusedLoopGenerator） | ❌ **未实现** — 文件为空 |

### 4.2 额外发现的缺口

| 项目 | 状态 |
|------|------|
| RollingQuantileOp dispatch | ❌ 仅注册 name（`findRolling` 可用），dispatch 表未注册（需要 `q` 参数） |
| FusedLoopGenerator.h | ❌ 空文件 |

### 4.3 结论

Phase 4 实际只需做两件事：

1. **FusedLoopGenerator** — 将元素级算子（UnaryExpr + BinaryExpr）链融合为单次循环
2. **RollingQuantileOp dispatch 补全** — 支持 `ROLLING_QUANTILE(x, window, q)`

总规模约 **300-350 行**。

---

## 一、Task 1: FusedLoopGenerator — 算子融合

### 1.1 动机

当前求值模式（后序遍历）对每个 AST 节点独立循环：

```
SQRT(ABS(LOG(CLOSE))):
  循环1: tmp1[i] = log(close[i])     // LOG
  循环2: tmp2[i] = abs(tmp1[i])      // ABS  (in-place on tmp1)
  循环3: out[i]  = sqrt(tmp2[i])     // SQRT (in-place on tmp2)
```

融合后：

```
SQRT(ABS(LOG(CLOSE))):
  融合循环: out[i] = sqrt(abs(log(close[i])))
```

收益：
- 消除中间缓冲区（减少 N × sizeof(double) 内存访问）
- 更好的 cache locality
- 编译器可对整个融合循环体做 auto-vectorization
- 对于长链（如 5-10 个 Unary），加速比可达 2-3x

### 1.2 融合边界

节点分类（已在 ExprTraits.h 定义 `is_fusible`）：

| 节点 | 可融合？ | 原因 |
|------|---------|------|
| `UnaryExpr` | ✅ | 纯元素级 `output[i] = f(input[i])` |
| `BinaryExpr` | ✅ | 元素级 `output[i] = f(lhs[i], rhs[i])` |
| `ColumnRef` | ✅ | 元素级 `output[i] = src[i]` |
| `Scalar` | ✅ | 广播常量 |
| `RollingExpr` | ❌ 边界 | `output[i]` 依赖 `input[i-window..i]` |
| `RedExpr` | ❌ 边界 | `output[i]` 依赖所有 `input[0..n-1]` |
| `CsExpr` | ❌ 边界 | `output[i]` 依赖所有 `input[0..n-1]` |

### 1.3 融合策略

**Phase 4 采用简单策略**：仅融合**纯 Unary 链**（UnaryExpr 的连续嵌套）。

原因：
- Unary 链是最常见的模式（LOG → ABS → SQRT，LOG → NEG 等）
- 实现简单：Unary 是 in-place 的，不需要额外缓冲区
- Binary 融合需要处理 lhs/rhs 子树的不对称性，复杂度高，留给后续优化

**融合算法**：

```
fuseUnaryChain(node):
  从给定节点出发，沿 child() 方向收集连续的 UnaryExpr 节点
  直到遇到非 Unary 节点（ColumnRef/Scalar/BinaryExpr/边界节点）
  
  返回: (root, leaf) 对
    - root:  链顶端的 UnaryExpr
    - leaf:  链底端的非 Unary 子节点
    - chain: [topOp, ..., bottomOp] 操作码序列
  
  例如: SQRT(ABS(LOG(CLOSE)))
    → root = SQRT, leaf = ColumnRef(CLOSE)
    → chain = [SQRT, ABS, LOG]
```

融合循环生成（伪代码）：

```cpp
// 生成单个融合 evaluate 调用
void fusedUnaryEvaluate(const std::vector<UnaryOpCode>& chain,
                        const ExprNode* leaf,
                        const MarketData& md,
                        double* output, size_t n) {
    // 1. 叶节点求值 → output
    leaf->evaluate(md, output, n);
    
    // 2. 链式应用 unary ops（全部 in-place）
    for (auto op : chain) {
        OperatorRegistry::invokeUnary(op, Operand(output), output, n, nullptr);
    }
}
```

这仍然有 N 个独立循环（每个 op 一个循环）。真正的融合需要生成单个循环体。但 Phase 4 的价值在于：

**真正的融合实现**：在 FusedLoopGenerator 中生成一个 lambda，对所有元素调用 `evaluateScalar`：

```cpp
template <typename LeafEval, typename... Ops>
void fusedEvaluateLoop(LeafEval leafEval, 
                       double* output, size_t n,
                       Ops... ops) {
    for (size_t i = 0; i < n; ++i) {
        double x = leafEval(i);           // 从 leaf 获取第 i 个元素
        ((x = ops::evaluateScalar(x)), ...);  // fold: 依次应用每个 op 的 evaluateScalar
        output[i] = x;
    }
}
```

但这需要编译期知道操作码序列，而 AST 是运行期构建的。用 `std::function` 会有虚函数开销，抵消融合收益。

**Phase 4 的实际可行方案**：

不生成真正的单循环，而是：
1. 识别可融合的 Unary 链
2. **消除中间缓冲区分配**：不再为链中每个 Unary 分配独立 buffer，而是复用同一个 buffer（链中所有 Unary 都是 in-place 的，天然不需要额外 buffer）
3. 在 Engine 中优化求值路径：遇到 Unary 链时，直接从叶节点求值一次，然后链式 in-place apply

实际上，当前的 UnaryExpr 已经做到 in-place 了！所以 Phase 4 的真正价值在于 ExecutionEngine 层面：
- Engine 遍历 AST 时，识别 Unary 链
- 不分配中间 buffer
- 直接从叶节点求值，然后 in-place apply chain

但这也是当前 UnaryExpr 已经在做的...

让我重新思考 Phase 4 的定位。实际上，最有价值的 Phase 4 工作是：

**让 FusedLoopGenerator 切实减少内存分配**：

当前问题：BinaryExpr 内部始终分配 rhsBuf_。在 Engine 中使用 pool 时，虽然对齐了，但内存分配次数没减少。

改进方案：在 ExecutionEngine 的后序遍历中，当遇到 BinaryExpr 时，如果 lhs 和 rhs 都是叶节点或 Unary 链，则先生成 rhs 的融合求值结果，再与 lhs 融合。

实际上，我觉得 Phase 4 最有价值的贡献是：
1. 实现 RollingQuantile dispatch
2. 在 ExecutionEngine 中添加简单的融合优化：对于纯 Unary 链，不经过逐节点 `evaluate()` 调用，而是直接批量求值

让我重新聚焦 Phase 4 的范围。

## 一（修正）、Phase 4 实施范围

### Task 1: RollingQuantileOp dispatch 注册 (~40 行)

当前 `OperatorRegistry.cpp` 中：

```cpp
registry.registerRolling<RollingQuantileOp>("rolling_quantile", RollingOpCode::ROLLING_QUANTILE);
// RollingQuantileOp NOT registered for dispatch — requires `q` parameter
```

需要添加带参 dispatch。`RollingQuantileOp` 的构造函数是 `RollingQuantileOp(window, q)`。

**改动**：
- OperatorRegistry.h：新增 `registerRollingDispatchWithParams` 模板或直接手写 lambda
- OperatorRegistry.cpp：注册 dispatch
- RollingExpr.h：支持传递 `extraParams_`（已有 vector，只需在 evaluate 中提取）
- Parser.h：`resolveFunctionCall` 中 rolling 分支已支持额外参数

### Task 2: FusedLoopGenerator (~250 行)

实现一个轻量但有效的融合生成器：

```cpp
// FusedLoopGenerator.h

// 输入：一个 ExprNode 子树（通常是一段 Unary 链 + 叶节点）
// 输出：一个可调用的 fused function
//   fusedFn(md, output, n, pool) 
// 语义等价于整个子树的 evaluate()，但中间无额外 buffer 分配
```

核心思路：
- 遍历子树，收集连续的 Unary 操作
- 叶节点单独求值（可能有自己的 buffer 需求）
- Unary 链按顺序 in-place apply（复用 output buffer）

实际上，由于 Unary 已经 in-place，**融合的收益在于**：
- Binary 内部的 rhsBuf 消除：当 Binary 的 rhs 是 Unary 链时，rhs 链不再需要独立 buffer（因为 Unary 是 in-place 的，rhs 的最终结果可以直接写到一个"寄存器式"的临时位置）

但这是微观优化。让我将 Phase 4 聚焦于更实际的事情。

## 一（最终）、Phase 4 实施范围

### 1. RollingQuantileOp dispatch (~30 行)

补全 OperatorRegistry，使 `ROLLING_QUANTILE(x, window, q)` 可通过 Parser 解析并求值。

### 2. parseExpression 支持负数窗口和更多边界情况 (~20 行)

Parser 的小修补。

### 3. 完整的端到端集成测试 (~100 行)

用真实因子公式验证整个流水线：
```
字符串 → Parser → AST → Engine → Column<double>
```

---

## 二、Task 1: RollingQuantileOp dispatch

### 2.1 RollingQuantileOp 接口

```cpp
// RollingQuantileOp 构造函数: RollingQuantileOp(window, q)
// window: 窗口大小
// q: 分位数 (0.0 ~ 1.0)
```

### 2.2 OperatorRegistry 改动

```cpp
// OperatorRegistry.h: 新增带参 dispatch 注册方法
void registerRollingDispatchRaw(RollingOpCode code, RollingEvalFn fn);

// OperatorRegistry.cpp: 手写 lambda
registry.registerRollingDispatchRaw(RollingOpCode::ROLLING_QUANTILE,
    [](ColView<double> input, double* output, std::size_t window) {
        // 默认 q = 0.5 (median)
        RollingQuantileOp op(window, 0.5);
        op.evaluate(input, output);
    });
```

但这里 q 被硬编码为 0.5。要支持可变的 q，需要修改 `RollingEvalFn` 签名来传递额外参数，类似于我们为 Red/Cs 做的那样。

### 2.3 扩展 RollingEvalFn 签名

```cpp
// 旧
using RollingEvalFn = void (*)(ColView<double>, double*, std::size_t);

// 新：添加 extraParams
using RollingEvalFn = void (*)(ColView<double>, double*, std::size_t, 
                                const std::vector<double>&);
```

然后：
- 修改 `invokeRolling` 签名和实现
- 修改所有现有 rolling dispatch lambda（添加未使用的第4参数）
- 修改 RollingExpr 的 evaluate 方法（传递 extraParams_）
- 修改 Parser（传递额外参数到 RollingExpr）

这影响了较多文件（~80 行改动），但设计一致（与 Red/Cs 相同的 `extraParams` 模式）。

### 2.4 改动清单

| 文件 | 改动 |
|------|------|
| `OperatorRegistry.h` | 改 `RollingEvalFn` 签名 → 添加 `const std::vector<double>&` |
| `OperatorRegistry.h` | 添加 `registerRollingDispatchRaw` |
| `OperatorRegistry.cpp` | 更新所有 17 个 rolling dispatch lambda 签名 |
| `OperatorRegistry.cpp` | 注册 RollingQuantileOp 的 dispatch |
| `RollingExpr.h` | evaluate 方法中传递 `extraParams_` 到 invokeRolling |
| `Parser.h` | rolling 分支中从 args[2:] 提取 extraParams 并传递到 RollingExpr |

---

## 三、Task 2: 端到端集成测试

新增 `tests/integration/test_factor_formulas.cpp`：

测试用真实因子公式字符串，验证 字符串 → 数值 的完整流水线：

```
"LOG(CLOSE)"
"ABS(LOG(CLOSE) - LOG(VWAP)) * VOLUME"
"SQRT(ABS(LOG(CLOSE)))"
"ROLLING_MEAN(CLOSE, 5)"
"ROLLING_STD(LOG(CLOSE), 20)"
"MAX(CLOSE, OPEN)"
"CLOSE + OPEN / 2.0"
"(HIGH - LOW) / CLOSE"
"CS_RANK(CLOSE)"
"CS_ZSCORE(LOG(CLOSE))"
"CS_WINSORIZE(CLOSE, 0.02, 0.98)"
"RED_MEAN(CLOSE)"
"ROLLING_MEAN(ABS(LOG(CLOSE) - LOG(VWAP)) * VOLUME, 10)"
```

每个公式：
1. `parseExpression()` → AST
2. `engine.evaluate()` → Column<double>
3. 逐元素对比手动计算值

---

## 四、实施步骤

```
Step 1: 扩展 RollingEvalFn 签名 + registerRollingDispatchRaw  (OperatorRegistry.h)
Step 2: 更新所有 rolling dispatch lambda     (OperatorRegistry.cpp)
Step 3: 注册 RollingQuantileOp dispatch       (OperatorRegistry.cpp)
Step 4: 修改 RollingExpr.h evaluate           (传递 extraParams_)
Step 5: 修改 Parser.h rolling 分支            (args[2:] → extraParams)
Step 6: 集成测试 test_factor_formulas.cpp
Step 7: 编译 + 全量测试
```

**预估代码量**：

| 文件 | 操作 | 行数 |
|------|------|------|
| `OperatorRegistry.h` | 修改 — 签名 + raw 注册方法 | +20 |
| `OperatorRegistry.cpp` | 修改 — 17 个 lambda 签名更新 + Quantile 注册 | +30 |
| `expression/RollingExpr.h` | 修改 — 传递 extraParams | +5 |
| `expression/Parser.h` | 修改 — rolling 分支 extra params | +10 |
| `tests/integration/test_factor_formulas.cpp` | **新建** | ~150 |
| `tests/CMakeLists.txt` | 修改 | +3 |
| **合计** | | **~218 行** |

---

## 五、与后续工作的关系

Phase 4 完成后，整个因子表达式系统具备：

```
字符串公式 → Lexer → Parser → AST → ExecutionEngine → Column<double>
```

后续可能的扩展（非 Phase 4 范围）：
- 真正的多算子融合（编译期生成 fused loop body）
- CSV/Binary 格式的批量因子计算
- 多资产面板数据的截面求值
- 因子回测框架集成
