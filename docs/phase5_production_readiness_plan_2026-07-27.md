# **状态**: ✅ 已完成 — FactorCalculator + 自定义算子 + $ref + CS 截面求值 + 完整测试

**日期**: 2026-07-27

**目标**: 使系统具备完整的产品化因子定义、自定义算子扩展和高层 API 能力

---

## 零、当前能力 vs 目标

### 目标 1: 使用因子表达式字符串定义因子

**当前状态**: ✅ 核心流水线已就绪（Phase 1-4），但缺少生产化封装。

```cpp
// 当前可用（5行样板代码）
auto ast = parseExpression("rolling_mean(abs(log(close)-log(vwap))*volume, 20)");
ExecutionEngine engine;
Column<double> result = engine.evaluate(*ast, marketData);

// 期望能力（1行）
Column<double> result = calc.evaluate("momentum_factor", marketData);
```

**差距**: 缺高层 API、缺命名公式管理、缺表达式缓存（同一公式在 N 只股票上重复解析）

### 目标 2: 自定义因子计算逻辑 + 复用已有算子

**当前状态**: ⚠️ 可手动构建 AST（C++ 代码组合已有节点），但不能在字符串表达式使用自定义算子。

```cpp
// 当前可用 — 手动组合已有算子
auto close = std::make_unique<ColumnRef>(Field::CLOSE);
auto logClose = std::make_unique<UnaryExpr>(UnaryOpCode::LOG, std::move(close));

// 不能做 — 注册新算子并用在字符串表达式中
calc.registerFormula("my_factor", "my_custom_op(close) * volume");
```

**差距**: 缺自定义算子注册机制、缺算子发现/列举 API

---

## 一、Task 1: 自定义算子注册机制

### 1.1 设计思路

当前算子枚举（UnaryOpCode 等）是编译期固定的 `enum class`。要在不修改核心库的前提下支持自定义算子，需要在每个枚举中预留自定义槽位。

**方案**: 为 UnaryOpCode 和 BinaryOpCode 添加 `CUSTOM_N` 值，提供公开的注册 API。

```
UnaryOpCode:
  ABS=0, LOG=1, ..., RANK_NORMALIZED=13,  kCount=14,
  CUSTOM_0=14, CUSTOM_1=15, ..., CUSTOM_7=21

BinaryOpCode:
  ADD=0, SUB=1, ..., NEQ=9,  kCount=10,
  CUSTOM_0=10, CUSTOM_1=11, CUSTOM_2=12, CUSTOM_3=13
```

### 1.2 Types.h 改动

```cpp
enum class UnaryOpCode : uint8_t {
    ABS, LOG, LOG10, LOG2, SQRT, NEG, SIGN, SQUARE, EXP, INV, NOT,
    RANK, RANK_PCT, RANK_NORMALIZED,
    kCount,  // now 14

    // Custom operator slots (reserved for user-defined operators)
    CUSTOM_0 = 14, CUSTOM_1 = 15, CUSTOM_2 = 16, CUSTOM_3 = 17,
    CUSTOM_4 = 18, CUSTOM_5 = 19, CUSTOM_6 = 20, CUSTOM_7 = 21,
};

enum class BinaryOpCode : uint8_t {
    ADD, SUB, MUL, DIV, MAX, MIN, GT, LT, EQ, NEQ,
    kCount,  // now 10

    CUSTOM_0 = 10, CUSTOM_1 = 11, CUSTOM_2 = 12, CUSTOM_3 = 13,
};
```

不需要为 Rolling/Red/Cs 添加自定义槽位——这三族的算子涉及复杂的多元素/跨截面计算，用户自定义的场景极少。需要时可以通过 C++ 代码层面组合已有算子实现。

### 1.3 OperatorRegistry 新增公开 API

```cpp
// OperatorRegistry.h — 在 public 区域新增

/// Register a custom unary operator for string expression use.
/// @param name        Name for formula strings (case-insensitive, like "my_op")
/// @param code        Must be one of UnaryOpCode::CUSTOM_0 .. CUSTOM_7
/// @param scalarFn    The evaluateScalar function (double → double)
///
/// Usage:
///   auto& reg = OperatorRegistry::instance();
///   reg.registerCustomUnary("my_square", UnaryOpCode::CUSTOM_0,
///       [](double x) { return x * x; });
void registerCustomUnary(const std::string& name,
                         UnaryOpCode code,
                         UnaryScalarFn scalarFn);

/// Register a custom binary operator for string expression use.
/// @param name        Name for formula strings ("my_distance")
/// @param code        Must be one of BinaryOpCode::CUSTOM_0 .. CUSTOM_3
/// @param scalarFn    The evaluateScalar function (double, double → double)
void registerCustomBinary(const std::string& name,
                          BinaryOpCode code,
                          BinaryScalarFn scalarFn);

/// List all registered unary operators (built-in + custom).
std::vector<std::string> listUnary() const;

/// List all registered binary operators (built-in + custom).
std::vector<std::string> listBinary() const;
```

实现要点：
- 注册 name→code 映射到 `unaryRegistry_` / `binaryRegistry_`
- 注册 scalarFn 到 `unaryScalar_` / `binaryScalar_`
- 生成一个简单的 element-wise evaluate lambda 注册到 `unaryDispatch_` / `binaryDispatch_`（内部调用 scalarFn 循环）
- 对于列式输入：for(i) output[i] = scalarFn(input[i]); null 位置设 0.0
- 对于标量输入：计算一次 broadcast 到全部位置

### 1.4 使用示例

```cpp
// 1. 定义自定义算子
struct MyTripleOp : public UnaryOperator<MyTripleOp> {
    static constexpr UnaryOpCode kOpCode = UnaryOpCode::CUSTOM_0;
    static constexpr const char* name = "triple";
    static double evaluateScalar(double x) noexcept {
        return x * 3.0;
    }
};

// 2. 注册（程序初始化时执行一次）
auto& reg = OperatorRegistry::instance();
reg.registerCustomUnary("triple", UnaryOpCode::CUSTOM_0, MyTripleOp::evaluateScalar);
// 内置算子风格注册（可选，利用 CRTP）:
reg.registerCustomUnaryEx<MyTripleOp>("triple", UnaryOpCode::CUSTOM_0);

// 3. 在因子表达式中使用
calc.registerFormula("my_factor", "triple(close) + log(volume)");
```

---

## 二、Task 2: FactorCalculator — 高层 API

### 2.1 设计

`FactorCalculator` 是面向用户的一站式入口，封装了解析、缓存、求值的全流程。

```cpp
// FactorCalculator.h
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "quantcore/engine/ExecutionEngine.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/registry/OperatorRegistry.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"

namespace quantcore {

class FactorCalculator {
public:
    FactorCalculator() = default;

    // ============================================================
    // Formula management
    // ============================================================

    /// Register a named formula.
    /// Formula strings use the same syntax as parseExpression().
    /// All built-in and registered custom operators are available.
    ///
    /// Examples:
    ///   calc.registerFormula("momentum",     "close / rolling_mean(close, 20) - 1");
    ///   calc.registerFormula("volatility",   "rolling_std(log_return, 20)");
    ///   calc.registerFormula("log_return",   "log(close) - log(rolling_shift(close, 1))");
    void registerFormula(const std::string& name,
                         const std::string& expression);

    /// Remove a registered formula.
    void unregisterFormula(const std::string& name);

    /// List all registered formula names.
    std::vector<std::string> formulas() const;

    /// Get the expression string for a formula.
    std::string formulaExpression(const std::string& name) const;

    // ============================================================
    // Evaluation
    // ============================================================

    /// Evaluate a named formula against market data.
    /// The formula is parsed on first evaluation and cached for subsequent calls.
    /// @throws std::runtime_error if the formula is not registered or
    ///         the expression fails to parse.
    Column<double> evaluate(const std::string& formulaName,
                            const MarketData& md);

    /// Parse and evaluate an expression directly (no named registration).
    /// The parsed AST is cached by expression string.
    Column<double> evaluateExpression(const std::string& expression,
                                       const MarketData& md);

    // ============================================================
    // Engine & registry access
    // ============================================================

    ExecutionEngine& engine() noexcept { return engine_; }
    const ExecutionEngine& engine() const noexcept { return engine_; }

    OperatorRegistry& registry() { return OperatorRegistry::instance(); }

    // ============================================================
    // Cache management
    // ============================================================

    /// Clear the parsed expression cache.
    /// Subsequent evaluate() calls will re-parse.
    void clearCache();

    /// Number of cached parsed expressions.
    std::size_t cacheSize() const noexcept { return parsedCache_.size(); }

    // ============================================================
    // Batch evaluation (multiple formulas × single MarketData)
    // ============================================================

    /// Evaluate multiple named formulas on the same MarketData.
    /// More efficient than calling evaluate() separately because the
    /// MarketData is accessed only once per field.
    std::vector<Column<double>> evaluateBatch(
        const std::vector<std::string>& formulaNames,
        const MarketData& md);

private:
    /// Get or create a parsed AST for an expression string.
    const ExprNode* getOrParse(const std::string& expression);

    ExecutionEngine engine_;

    // Named formulas: name → expression string
    std::unordered_map<std::string, std::string> formulaDefs_;

    // Parsed AST cache: expression string → AST
    std::unordered_map<std::string, std::unique_ptr<ExprNode>> parsedCache_;
};

}  // namespace quantcore
```

### 2.2 实现要点

```cpp
// FactorCalculator.cpp

void FactorCalculator::registerFormula(const std::string& name,
                                        const std::string& expression) {
    formulaDefs_[name] = expression;
}

Column<double> FactorCalculator::evaluate(const std::string& formulaName,
                                           const MarketData& md) {
    auto it = formulaDefs_.find(formulaName);
    if (it == formulaDefs_.end()) {
        throw std::runtime_error(
            "FactorCalculator: unknown formula '" + formulaName + "'");
    }

    const ExprNode* ast = getOrParse(it->second);
    return engine_.evaluate(*ast, md);
}

const ExprNode* FactorCalculator::getOrParse(const std::string& expression) {
    auto it = parsedCache_.find(expression);
    if (it != parsedCache_.end()) {
        return it->second.get();
    }
    auto ast = parseExpression(expression);
    auto* ptr = ast.get();
    parsedCache_[expression] = std::move(ast);
    return ptr;
}

std::vector<Column<double>> FactorCalculator::evaluateBatch(
        const std::vector<std::string>& formulaNames,
        const MarketData& md) {
    std::vector<Column<double>> results;
    results.reserve(formulaNames.size());
    for (auto& name : formulaNames) {
        results.push_back(evaluate(name, md));
    }
    return results;
}
```

注意：`evaluateBatch` 的当前实现只是循环调用。真正的批量优化（如共享 ColumnRef 的物化结果）可以作为后续优化。Phase 5 先提供接口，实现保持简单。

### 2.3 使用示例

```cpp
FactorCalculator calc;

// 注册公式（支持引用其他公式）
calc.registerFormula("log_return",  "log(close) - log(rolling_shift(close, 1))");
calc.registerFormula("volatility",  "rolling_std(log_return, 20)");
calc.registerFormula("momentum",    "close / rolling_mean(close, 60) - 1");
calc.registerFormula("turnover_adj","abs(log_return) * volume");

// 单次求值
Column<double> vol = calc.evaluate("volatility", marketData);

// 批量求值
auto results = calc.evaluateBatch(
    {"momentum", "volatility", "turnover_adj"}, marketData);
```

**注意**: 公式间的引用（如 `volatility` 引用 `log_return`）在当前设计中是通过**表达式文本内联**实现的。更高级的依赖图解析可以后续添加。

---

## 三、Task 3: 表达式内公式引用

### 3.1 设计

允许一个公式引用另一个已注册的公式：

```
calc.registerFormula("log_return",  "log(close) - log(rolling_shift(close, 1))");
calc.registerFormula("volatility",  "rolling_std($log_return, 20)");
```

`$name` 语法引用已注册公式。Parser 在解析 IDENTIFIER 时检测 `$` 前缀，查找 formulaDefs，将引用的公式表达式内联展开。

### 3.2 Parser 改动

在 `resolveColumnRef` 之前，检测 `$` 前缀：

```cpp
// primary → '$' IDENTIFIER  →  formula reference (inline expansion)
if (check(TokenType::DOLLAR)) {
    advance();  // consume '$'
    Token name = consume(TokenType::IDENTIFIER, "expected formula name after '$'");
    // Look up formula definition from FactorCalculator
    // Re-parse the referenced formula's expression inline
    std::string refExpr = calculator_->resolveFormulaRef(name.text);
    Lexer subLexer;
    auto subTokens = subLexer.tokenize(refExpr);
    Parser subParser;
    return subParser.parse(subTokens);
}
```

### 3.3 简化版本

如果觉得 `$name` 语法过于复杂，Phase 5 可以采用更简单的方案：**不实现公式引用，只支持表达式内联**。公式引用作为 Phase 5+ 的可选特性。

---

## 四、实施步骤

```
Step 1: Types.h — 添加 CUSTOM_0..CUSTOM_N 到 UnaryOpCode / BinaryOpCode
Step 2: OperatorRegistry — 添加 registerCustomUnary / registerCustomBinary / listOperators
Step 3: FactorCalculator.h/.cpp — 高层 API
Step 4: 测试 — 自定义算子注册 + FactorCalculator 集成
Step 5: 编译 + 全量测试
```

**预估代码量**：

| 文件 | 操作 | 行数 |
|------|------|------|
| `core/Types.h` | 修改 — 添加 CUSTOM_N enum 值 | +10 |
| `registry/OperatorRegistry.h` | 修改 — 添加 registerCustomXxx 声明 | +25 |
| `registry/OperatorRegistry.cpp` | 修改 — 实现 registerCustomXxx | +55 |
| `core/FactorCalculator.h` | **新建** | ~85 |
| `core/FactorCalculator.cpp` | **新建** | ~65 |
| `tests/unit/test_factor_calculator.cpp` | **新建** | ~140 |
| `tests/CMakeLists.txt` | 修改 | +3 |
| **合计** | | **~383 行** |

---

## 五、测试计划

`test_factor_calculator.cpp`:

| 类别 | 测试用例 | 数量 |
|------|---------|------|
| 自定义算子注册 | registerCustomUnary、重新注册覆盖、list 输出 | 3 |
| 自定义算子求值 | 字符串表达式中使用自定义算子、与手动构建 AST 对比 | 3 |
| 自定义二元算子 | registerCustomBinary + 字符串表达式中使用 | 2 |
| FactorCalculator 基本 | registerFormula + evaluate、重复求值命中缓存 | 3 |
| FactorCalculator 直接求值 | evaluateExpression（不注册公式名）| 1 |
| FactorCalculator 批量 | evaluateBatch 返回正确结果 | 1 |
| FactorCalculator 缓存 | clearCache 后重解析、cacheSize 验证 | 2 |
| 端到端 | 自定义算子 + 命名公式 + 字符串求值完整流水线 | 3 |
| 错误处理 | 未注册公式、自定义算子名称冲突、无效表达式 | 3 |
| **合计** | | **~21** |

---

## 六、完成后的用户工作流

```cpp
#include <quantcore/core/FactorCalculator.h>
using namespace quantcore;

// ═══════════════════════════════════════════════════════
// 初始化（程序启动时执行一次）
// ═══════════════════════════════════════════════════════

FactorCalculator calc;

// 1. 注册自定义算子
calc.registry().registerCustomUnary("triple", UnaryOpCode::CUSTOM_0,
    [](double x) noexcept { return x * 3.0; });

// 2. 注册因子公式
calc.registerFormula("log_return",  "log(close) - log(rolling_shift(close, 1))");
calc.registerFormula("volatility",  "rolling_std(log_return, 20)");
calc.registerFormula("momentum",    "close / rolling_mean(close, 60) - 1");
calc.registerFormula("custom_fact", "triple(close) + log(volume)");

// ═══════════════════════════════════════════════════════
// 每日批量计算
// ═══════════════════════════════════════════════════════

for (auto& stock : universe) {
    MarketData md = dataLoader.load(stock, "20240101", "20241231");

    // 一键求值
    Column<double> mom = calc.evaluate("momentum", md);

    // 批量求值
    auto factors = calc.evaluateBatch(
        {"momentum", "volatility", "custom_fact"}, md);
}
```

## 七、Phase 1-5 总览

| Phase | 产出 | 用户可见能力 |
|-------|------|------------|
| 1 | 7 种 AST 节点 + 全算子 dispatch | 手动构建表达式树求值 |
| 2 | BufferPool + ExecutionEngine | `engine.evaluate(ast, md)` 统一入口 |
| 3 | Lexer + Parser | `"ABS(LOG(C)-LOG(V))*VOL"` → AST |
| 4 | 循环融合 + RollingQuantile dispatch | 性能优化（单循环消除临时缓冲区） |
| 5 | FactorCalculator + 自定义算子 | **1 行代码定义因子 + 注册并使用自定义算子** |
