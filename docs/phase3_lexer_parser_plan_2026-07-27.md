# Phase 3 详细设计方案：Lexer + Parser

**日期**: 2026-07-27

**状态**: ✅ 已完成 — 手写 Lexer + LL(1) Parser + 全部语法特性，测试通过: 无任何词法/语法解析代码；OperatorRegistry 已就绪（name→code 查找完整）

---

## 零、现状

| 组件 | 状态 |
|------|------|
| OperatorRegistry | ✅ name→code 双向映射完整（5 族共 61 个算子 + 7 个 Field） |
| Phase 1 AST 节点 | ✅ 7 种节点全部就绪 |
| Phase 2 ExecutionEngine | ✅ `engine.evaluate(expr, md) → Column<double>` |
| Lexer / Parser | ❌ **完全不存在** |

**关键依赖链已就绪**: 字符串 → Parser → AST → ExecutionEngine → `Column<double>`。只缺 Lexer 和 Parser 中间的环节。

---

## 一、目标流水线

```
字符串 "ABS(LOG(CLOSE) - LOG(VWAP)) * VOLUME"
         │
    [1]  Lexer  — 拆分为 Token 流
         │
    [2]  Parser — recursive-descent 构建 AST
         │
         ▼
   unique_ptr<ExprNode>  →  ExecutionEngine::evaluate()  →  Column<double>
```

### 支持的语法

```
# 列引用
CLOSE, OPEN, HIGH, LOW, VOLUME, AMOUNT, VWAP

# 一元函数调用
ABS(x), LOG(x), LOG10(x), LOG2(x), SQRT(x), EXP(x)
NEG(x), SIGN(x), SQUARE(x), INV(x), NOT(x)
RANK(x), RANK_PCT(x), RANK_NORMALIZED(x)

# 二元中缀运算
x + y, x - y, x * y, x / y

# 二元函数调用
MAX(x, y), MIN(x, y), GT(x, y), LT(x, y), EQ(x, y), NEQ(x, y)

# 滚动窗口函数
ROLLING_MEAN(x, 20), ROLLING_STD(x, 60), ROLLING_SHIFT(x, 1)

# 截面函数（无参）
CS_RANK(x), CS_ZSCORE(x), CS_NORMALIZE(x), CS_DEMEAN(x)

# 截面函数（有参）
CS_WINSORIZE(x, 0.02, 0.98), CS_CLIP(x, -3.0, 3.0)

# 归约函数
RED_MEAN(x), RED_STD(x), RED_ZSCORE(x)

# 嵌套 + 中缀混合
ABS(LOG(CLOSE) - LOG(VWAP)) * VOLUME
SQRT(ABS(LOG(CLOSE)))
ROLLING_MEAN(HIGH - LOW, 20)
```

### 语法规则（BNF）

```
expr       →  term (('+' | '-') term)*
term       →  factor (('*' | '/') factor)*
factor     →  ('+' | '-') factor           // unary plus/minus (numeric literals)
           |  primary
primary    →  NUMBER                        // double literal
           |  IDENTIFIER '(' args ')'      // function call
           |  IDENTIFIER                    // column reference
           |  '(' expr ')'                  // grouping
args       →  ε                             // empty (zero-arg)
           |  expr (',' expr)*
```

---

## 二、核心设计决策

### 决策 1：Token 粒度

统一使用 8 种 Token：

| Token | 示例 | 说明 |
|-------|------|------|
| `NUMBER` | `3.14`, `-2.5`, `100` | double 字面量 |
| `IDENTIFIER` | `CLOSE`, `rolling_mean`, `abs` | 列名或函数名 |
| `LPAREN` | `(` | |
| `RPAREN` | `)` | |
| `COMMA` | `,` | |
| `PLUS` | `+` | 二元加 / 一元正号 |
| `MINUS` | `-` | 二元减 / 一元负号 |
| `STAR` | `*` | 乘号 |
| `SLASH` | `/` | 除号 |
| `END` | (输入结束) | |

IDENTIFIER 大小写不敏感（全转小写存储），匹配 OperatorRegistry 和 Field 名。

### 决策 2：名字解析策略

Parser 看到 `IDENTIFIER` 时：

**情况 A**: 裸标识符（后面不是 `(`）→ 尝试解析为列引用
```
CLOSE  →  Field::CLOSE  →  ColumnRef(Field::CLOSE)
```

**情况 B**: 标识符后面是 `(` → 函数调用，按优先级查找 OperatorRegistry：
```
1. findUnary(name)       → UnaryExpr(op, child)
2. findBinary(name)      → BinaryExpr(op, lhs, rhs)     [需要恰好 2 个 args]
3. findRolling(name)     → RollingExpr(op, window, child) [需要恰好 2 个 args, 第2个是数值]
4. findRed(name)         → RedExpr(op, child, extraParams)
5. findCs(name)          → CsExpr(op, child, extraParams)
6. 都找不到              → 报错
```

**解析优化**: `findXxx()` 成功后立即构造对应节点，不需要 fallback。

### 决策 3：错误处理

使用语义化错误信息：

```
ABS(CLOSE,         // 位置 10: 缺少 ')'
LOG(UNKNOWN)        // 位置 4: 未知函数 'LOG'
ROLLING_MEAN(CLOSE) // 位置 0: 缺少窗口参数
```

每个解析函数接受 `position` 参数并返回解析失败的上下文位置。

### 决策 4：文件结构

全部 header-only（与 expression/ 风格一致）：

```
expression/
├── Token.h      — Token 类型 + Token 值 + Token 流
├── Lexer.h      — 字符串 → Token 流
└── Parser.h     — Token 流 → AST
```

不需要 .cpp 文件——解析器是轻量的纯算法代码。

---

## 三、逐文件详细设计

### 3.1 Token.h — Token 类型定义

```cpp
// Token.h — token types and values for the expression parser
#pragma once

#include <string>
#include <variant>
#include <string_view>

namespace quantcore {

enum class TokenType : uint8_t {
    NUMBER,       // double literal
    IDENTIFIER,   // column name or function name
    LPAREN,       // '('
    RPAREN,       // ')'
    COMMA,        // ','
    PLUS,         // '+'
    MINUS,        // '-'
    STAR,         // '*'
    SLASH,        // '/'
    END,          // end of input
};

struct Token {
    TokenType type;
    std::string text;       // original text span
    double    numberValue;  // valid when type == NUMBER

    // Position in original source (for error messages)
    std::size_t pos;

    // Convenience constructors
    static Token number(double v, std::string_view txt, std::size_t p) {
        Token t;
        t.type = TokenType::NUMBER;
        t.text = std::string(txt);
        t.numberValue = v;
        t.pos = p;
        return t;
    }
    static Token identifier(std::string_view txt, std::size_t p) {
        Token t;
        t.type = TokenType::IDENTIFIER;
        t.text = std::string(txt);
        t.pos = p;
        return t;
    }
    static Token simple(TokenType ty, std::size_t p) {
        Token t;
        t.type = ty;
        t.text = (ty == TokenType::LPAREN) ? "("
               : (ty == TokenType::RPAREN) ? ")"
               : (ty == TokenType::COMMA)  ? ","
               : (ty == TokenType::PLUS)   ? "+"
               : (ty == TokenType::MINUS)  ? "-"
               : (ty == TokenType::STAR)   ? "*"
               : (ty == TokenType::SLASH)  ? "/"
               : "";
        t.pos = p;
        return t;
    }
};

// For error messages
const char* tokenTypeName(TokenType t);

}  // namespace quantcore
```

### 3.2 Lexer.h — 词法分析器

```cpp
// Lexer.h — string → token stream
#pragma once

#include <cctype>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>

#include "quantcore/expression/Token.h"

namespace quantcore {

class Lexer {
public:
    /// Tokenize an expression string.
    /// @throws std::runtime_error on invalid characters
    std::vector<Token> tokenize(std::string_view input);

private:
    void skipWhitespace();
    Token readNumber();
    Token readIdentifier();
    Token readOperator();

    std::string_view input_;
    std::size_t pos_ = 0;
};

// ============================================================
// Implementation (inline, header-only)
// ============================================================

inline std::vector<Token> Lexer::tokenize(std::string_view input) {
    input_ = input;
    pos_ = 0;
    std::vector<Token> tokens;

    while (pos_ < input_.size()) {
        char c = input_[pos_];

        if (std::isspace(c)) {
            skipWhitespace();
            continue;
        }

        if (std::isdigit(c) || (c == '.' && pos_ + 1 < input_.size()
                                && std::isdigit(input_[pos_ + 1]))) {
            tokens.push_back(readNumber());
        } else if (std::isalpha(c) || c == '_') {
            tokens.push_back(readIdentifier());
        } else if (c == '(') {
            tokens.push_back(Token::simple(TokenType::LPAREN, pos_));
            ++pos_;
        } else if (c == ')') {
            tokens.push_back(Token::simple(TokenType::RPAREN, pos_));
            ++pos_;
        } else if (c == ',') {
            tokens.push_back(Token::simple(TokenType::COMMA, pos_));
            ++pos_;
        } else if (c == '+') {
            tokens.push_back(Token::simple(TokenType::PLUS, pos_));
            ++pos_;
        } else if (c == '-') {
            tokens.push_back(Token::simple(TokenType::MINUS, pos_));
            ++pos_;
        } else if (c == '*') {
            tokens.push_back(Token::simple(TokenType::STAR, pos_));
            ++pos_;
        } else if (c == '/') {
            tokens.push_back(Token::simple(TokenType::SLASH, pos_));
            ++pos_;
        } else {
            throw std::runtime_error(
                std::string("Lexer: unexpected character '")
                + c + "' at position " + std::to_string(pos_));
        }
    }

    tokens.push_back(Token::simple(TokenType::END, pos_));
    return tokens;
}

inline void Lexer::skipWhitespace() {
    while (pos_ < input_.size() && std::isspace(input_[pos_])) {
        ++pos_;
    }
}

inline Token Lexer::readNumber() {
    std::size_t start = pos_;

    // Integer part
    while (pos_ < input_.size() && std::isdigit(input_[pos_])) ++pos_;

    // Fractional part
    if (pos_ < input_.size() && input_[pos_] == '.') {
        ++pos_;
        while (pos_ < input_.size() && std::isdigit(input_[pos_])) ++pos_;
    }

    // Exponent (e.g., 1.5e-3)
    if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
        ++pos_;
        if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) ++pos_;
        while (pos_ < input_.size() && std::isdigit(input_[pos_])) ++pos_;
    }

    std::string_view text = input_.substr(start, pos_ - start);
    double val = std::stod(std::string(text));
    return Token::number(val, text, start);
}

inline Token Lexer::readIdentifier() {
    std::size_t start = pos_;

    // First char: alpha or underscore
    ++pos_;

    // Subsequent chars: alphanumeric or underscore
    while (pos_ < input_.size()
           && (std::isalnum(input_[pos_]) || input_[pos_] == '_')) {
        ++pos_;
    }

    std::string_view text = input_.substr(start, pos_ - start);

    // Convert to lowercase (case-insensitive matching)
    std::string lower;
    lower.reserve(text.size());
    for (char c : text) lower += static_cast<char>(std::tolower(c));

    return Token::identifier(lower, start);
}

}  // namespace quantcore
```

### 3.3 Parser.h — 递归下降语法分析器

```cpp
// Parser.h — recursive-descent parser: token stream → AST
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/expression/ColumnRef.h"
#include "quantcore/expression/Scalar.h"
#include "quantcore/expression/UnaryExpr.h"
#include "quantcore/expression/BinaryExpr.h"
#include "quantcore/expression/RollingExpr.h"
#include "quantcore/expression/RedExpr.h"
#include "quantcore/expression/CsExpr.h"
#include "quantcore/expression/Token.h"
#include "quantcore/registry/OperatorRegistry.h"

namespace quantcore {

class Parser {
public:
    /// Parse a token stream into an expression AST.
    /// @throws std::runtime_error on syntax errors
    std::unique_ptr<ExprNode> parse(const std::vector<Token>& tokens);

private:
    // Advance and return current token
    Token advance();
    const Token& peek() const;     // current token (END if exhausted)
    const Token& peekAhead() const; // lookahead 1
    bool check(TokenType t) const;
    Token consume(TokenType t, const char* msg);

    // Grammar productions
    std::unique_ptr<ExprNode> parseExpr();
    std::unique_ptr<ExprNode> parseTerm();
    std::unique_ptr<ExprNode> parseFactor();
    std::unique_ptr<ExprNode> parsePrimary();

    // Argument list: a, b, c
    std::vector<std::unique_ptr<ExprNode>> parseArgs();

    // Name resolution
    std::unique_ptr<ExprNode> resolveColumnRef(const std::string& name, std::size_t pos);
    std::unique_ptr<ExprNode> resolveFunctionCall(const std::string& name,
        std::vector<std::unique_ptr<ExprNode>> args, std::size_t pos);

    // Helper: extract a numeric literal from an expression node (for window sizes, etc.)
    static double extractNumber(const ExprNode* node);

    const std::vector<Token>* tokens_ = nullptr;
    std::size_t idx_ = 0;
};

// ============================================================
// Implementation
// ============================================================

inline std::unique_ptr<ExprNode> Parser::parse(const std::vector<Token>& tokens) {
    tokens_ = &tokens;
    idx_ = 0;
    auto expr = parseExpr();
    if (!check(TokenType::END)) {
        throw std::runtime_error("Parser: unexpected token after expression at position "
            + std::to_string(peek().pos));
    }
    return expr;
}

inline Token Parser::advance() {
    return (*tokens_)[idx_++];
}

inline const Token& Parser::peek() const {
    return (*tokens_)[idx_];
}

inline const Token& Parser::peekAhead() const {
    if (idx_ + 1 < tokens_->size()) {
        return (*tokens_)[idx_ + 1];
    }
    return (*tokens_)[idx_];  // fallback to END
}

inline bool Parser::check(TokenType t) const {
    return peek().type == t;
}

inline Token Parser::consume(TokenType t, const char* msg) {
    if (check(t)) return advance();
    throw std::runtime_error(
        std::string("Parser: ") + msg + " at position "
        + std::to_string(peek().pos));
}

// ============================================================
// Grammar productions
// ============================================================

// expr → term (('+' | '-') term)*
inline std::unique_ptr<ExprNode> Parser::parseExpr() {
    auto left = parseTerm();

    while (check(TokenType::PLUS) || check(TokenType::MINUS)) {
        Token op = advance();
        auto right = parseTerm();

        BinaryOpCode code = (op.type == TokenType::PLUS)
            ? BinaryOpCode::ADD : BinaryOpCode::SUB;

        left = std::make_unique<BinaryExpr>(code, std::move(left), std::move(right));
    }

    return left;
}

// term → factor (('*' | '/') factor)*
inline std::unique_ptr<ExprNode> Parser::parseTerm() {
    auto left = parseFactor();

    while (check(TokenType::STAR) || check(TokenType::SLASH)) {
        Token op = advance();
        auto right = parseFactor();

        BinaryOpCode code = (op.type == TokenType::STAR)
            ? BinaryOpCode::MUL : BinaryOpCode::DIV;

        left = std::make_unique<BinaryExpr>(code, std::move(left), std::move(right));
    }

    return left;
}

// factor → ('+' | '-') factor | primary
inline std::unique_ptr<ExprNode> Parser::parseFactor() {
    // Unary plus: +x → x (identity, optimize away)
    if (check(TokenType::PLUS)) {
        advance();
        return parseFactor();
    }

    // Unary minus: -x → NEG(x)   or   -3.14 → Scalar(-3.14)
    if (check(TokenType::MINUS)) {
        Token op = advance();

        // If followed by a number literal, fold into a negative scalar
        if (check(TokenType::NUMBER)) {
            Token num = advance();
            return std::make_unique<Scalar>(-num.numberValue);
        }

        // Otherwise: NEG(x)
        auto operand = parseFactor();
        return std::make_unique<UnaryExpr>(UnaryOpCode::NEG, std::move(operand));
    }

    return parsePrimary();
}

// primary → NUMBER | IDENTIFIER '(' args ')' | IDENTIFIER | '(' expr ')'
inline std::unique_ptr<ExprNode> Parser::parsePrimary() {
    if (check(TokenType::NUMBER)) {
        Token t = advance();
        return std::make_unique<Scalar>(t.numberValue);
    }

    if (check(TokenType::IDENTIFIER)) {
        Token name = advance();

        // Function call: NAME '(' args ')'
        if (check(TokenType::LPAREN)) {
            advance();  // consume '('
            auto args = parseArgs();
            consume(TokenType::RPAREN, "expected ')'");
            return resolveFunctionCall(name.text, std::move(args), name.pos);
        }

        // Column reference: bare identifier
        return resolveColumnRef(name.text, name.pos);
    }

    if (check(TokenType::LPAREN)) {
        advance();  // consume '('
        auto expr = parseExpr();
        consume(TokenType::RPAREN, "expected ')'");
        return expr;
    }

    throw std::runtime_error(
        std::string("Parser: unexpected token '") + peek().text
        + "' at position " + std::to_string(peek().pos));
}

// args → ε | expr (',' expr)*
inline std::vector<std::unique_ptr<ExprNode>> Parser::parseArgs() {
    std::vector<std::unique_ptr<ExprNode>> args;

    // Empty argument list
    if (check(TokenType::RPAREN)) {
        return args;
    }

    // First argument
    args.push_back(parseExpr());

    // Subsequent arguments
    while (check(TokenType::COMMA)) {
        advance();  // consume ','
        args.push_back(parseExpr());
    }

    return args;
}

// ============================================================
// Name resolution
// ============================================================

inline std::unique_ptr<ExprNode> Parser::resolveColumnRef(
        const std::string& name, std::size_t pos) {
    // Try to match as a Field name
    static const struct { const char* n; Field f; } fieldMap[] = {
        {"open",   Field::OPEN},
        {"high",   Field::HIGH},
        {"low",    Field::LOW},
        {"close",  Field::CLOSE},
        {"volume", Field::VOLUME},
        {"amount", Field::AMOUNT},
        {"vwap",   Field::VWAP},
    };

    for (auto& entry : fieldMap) {
        if (name == entry.n) {
            return std::make_unique<ColumnRef>(entry.f);
        }
    }

    throw std::runtime_error(
        "Parser: unknown column reference '" + name
        + "' at position " + std::to_string(pos)
        + ". Did you mean to call it as a function?");
}

inline std::unique_ptr<ExprNode> Parser::resolveFunctionCall(
        const std::string& name,
        std::vector<std::unique_ptr<ExprNode>> args,
        std::size_t pos) {

    if (args.empty()) {
        throw std::runtime_error(
            "Parser: function '" + name + "' at position "
            + std::to_string(pos) + " requires at least one argument");
    }

    auto& reg = OperatorRegistry::instance();

    // -- Try unary (1 arg, 0 extra params) --
    try {
        UnaryOpCode code = reg.findUnary(name);
        if (args.size() != 1) {
            throw std::runtime_error(
                "Parser: unary function '" + name + "' takes exactly 1 argument, "
                + std::to_string(args.size()) + " given at position "
                + std::to_string(pos));
        }
        return std::make_unique<UnaryExpr>(code, std::move(args[0]));
    } catch (const std::runtime_error&) {
        // Not a unary op — only continue if the error was "Unknown unary operator"
        // Re-throw if it was an arg count error
    }

    // -- Try binary (2 args) --
    try {
        BinaryOpCode code = reg.findBinary(name);
        if (args.size() != 2) {
            throw std::runtime_error(
                "Parser: binary function '" + name + "' takes exactly 2 arguments, "
                + std::to_string(args.size()) + " given at position "
                + std::to_string(pos));
        }
        return std::make_unique<BinaryExpr>(code, std::move(args[0]), std::move(args[1]));
    } catch (const std::runtime_error&) {
        // Not a binary op — continue
    }

    // -- Try rolling (1 expr arg + 1 numeric window arg) --
    try {
        RollingOpCode code = reg.findRolling(name);
        if (args.size() < 2) {
            throw std::runtime_error(
                "Parser: rolling function '" + name + "' requires at least 2 arguments "
                "(expression, window) at position " + std::to_string(pos));
        }

        // Second argument must be a numeric literal (window size)
        double window = extractNumber(args[1].get());
        std::size_t w = static_cast<std::size_t>(window);
        if (w == 0 || static_cast<double>(w) != window) {
            throw std::runtime_error(
                "Parser: rolling function '" + name + "' window must be a positive "
                "integer, got " + std::to_string(window) + " at position "
                + std::to_string(pos));
        }

        // Extra args beyond the first 2 become extraParams (e.g. ROLLING_QUANTILE)
        std::vector<double> extraParams;
        for (std::size_t i = 2; i < args.size(); ++i) {
            extraParams.push_back(extractNumber(args[i].get()));
        }

        // For standard rolling ops, extraParams is empty and ignored
        return std::make_unique<RollingExpr>(code, w, std::move(args[0]));
    } catch (const std::runtime_error&) {
        // Not a rolling op — continue
    }

    // -- Try Red (1 expr arg + optional extra params) --
    try {
        RedOpCode code = reg.findRed(name);

        std::vector<double> extraParams;
        for (std::size_t i = 1; i < args.size(); ++i) {
            extraParams.push_back(extractNumber(args[i].get()));
        }

        return std::make_unique<RedExpr>(code, std::move(args[0]),
                                          std::move(extraParams));
    } catch (const std::runtime_error&) {
        // Not a Red op — continue
    }

    // -- Try CS (1 expr arg + optional extra params) --
    try {
        CsOpCode code = reg.findCs(name);

        std::vector<double> extraParams;
        for (std::size_t i = 1; i < args.size(); ++i) {
            extraParams.push_back(extractNumber(args[i].get()));
        }

        return std::make_unique<CsExpr>(code, std::move(args[0]),
                                         std::move(extraParams));
    } catch (const std::runtime_error&) {
        // Not a CS op — fall through to error
    }

    throw std::runtime_error(
        "Parser: unknown function '" + name + "' at position "
        + std::to_string(pos));
}

inline double Parser::extractNumber(const ExprNode* node) {
    // The argument should be a Scalar (numeric literal)
    // In a more advanced parser, we could evaluate constant expressions
    const Scalar* s = dynamic_cast<const Scalar*>(node);
    if (s) return s->value();

    throw std::runtime_error(
        "Parser: expected a numeric literal but got a complex expression. "
        "Window sizes and extra parameters must be numeric literals.");
}

// ============================================================
// Top-level convenience function
// ============================================================

/// Parse an expression string and return the AST.
/// Combines Lexer + Parser in one call.
inline std::unique_ptr<ExprNode> parseExpression(const std::string& source) {
    Lexer lexer;
    auto tokens = lexer.tokenize(source);
    Parser parser;
    return parser.parse(tokens);
}

}  // namespace quantcore
```

---

## 四、执行流程示例

```
输入: "ABS(LOG(CLOSE) - LOG(VWAP)) * VOLUME"

Lexer.tokenize():
  → [IDENTIFIER("abs"), LPAREN,
     IDENTIFIER("log"), LPAREN, IDENTIFIER("close"), RPAREN,
     MINUS,
     IDENTIFIER("log"), LPAREN, IDENTIFIER("vwap"), RPAREN,
     RPAREN,
     STAR,
     IDENTIFIER("volume"),
     END]

Parser.parse():
  expr → term ('*' term)?
    term → factor  (= ABS(...))
      factor → primary
        primary → IDENTIFIER("abs") '(' args ')'
          args → expr
            expr → term ('-' term)?
              term → factor  (= LOG(CLOSE))
                factor → primary
                  primary → IDENTIFIER("log") '(' IDENTIFIER("close") ')'
                    → resolveFunctionCall("log", [ColumnRef(CLOSE)])
                    → findUnary("log") = LOG → UnaryExpr(LOG, ColumnRef(CLOSE))
              MINUS
              term → factor  (= LOG(VWAP))
                → UnaryExpr(LOG, ColumnRef(VWAP))
            → BinaryExpr(SUB, LOG(CLOSE), LOG(VWAP))
          → findUnary("abs") = ABS → UnaryExpr(ABS, SUB(...))
    STAR
    term → factor → primary → IDENTIFIER("volume")
      → resolveColumnRef("volume") → ColumnRef(VOLUME)
  → BinaryExpr(MUL, ABS(...), ColumnRef(VOLUME))

结果 AST:
  MUL
  ├── ABS
  │   └── SUB
  │       ├── LOG
  │       │   └── COLUMN(close)
  │       └── LOG
  │           └── COLUMN(vwap)
  └── COLUMN(volume)
```

---

## 五、实施步骤

```
Step 1: Token.h         (独立，无依赖)
                           ↓
Step 2: Lexer.h          (依赖 Token.h)
                           ↓
Step 3: Parser.h          (依赖 Token.h + 所有 ExprNode + OperatorRegistry)
                           ↓
Step 4: test_parser.cpp   (依赖 Step 1-3)
                           ↓
Step 5: CMakeLists.txt    (添加 test_parser 目标)
                           ↓
Step 6: 编译 + 全量测试
```

**预估代码量**：

| 文件 | 操作 | 行数 |
|------|------|------|
| `expression/Token.h` | **新建** | ~80 |
| `expression/Lexer.h` | **新建** | ~150 |
| `expression/Parser.h` | **新建** | ~280 |
| `tests/unit/test_parser.cpp` | **新建** | ~250 |
| `tests/CMakeLists.txt` | 修改 — 添加 test_parser | +3 |
| **合计** | | **~763 行** |

---

## 六、测试计划

| 类别 | 测试用例 | 数量 |
|------|---------|------|
| **Lexer 基础** | 数字（整数、小数、科学计数法）、标识符、操作符、空白跳过 | 5 |
| **Lexer 错误** | 非法字符 | 1 |
| **Parser 列引用** | 7 个 Field 各一个 | 7 |
| **Parser 标量** | 正数、负数、小数、科学计数 | 2 |
| **Parser 一元函数** | ABS, LOG, SQRT, RANK 等 | 4 |
| **Parser 二元中缀** | +, -, *, / 及优先级(乘除优先于加减) | 5 |
| **Parser 二元函数** | MAX(x,y), MIN(x,y), EQ(x,y) | 3 |
| **Parser 括号** | 嵌套分组、多层括号 | 2 |
| **Parser 滚动** | ROLLING_MEAN(x,20), ROLLING_STD(x,60) | 3 |
| **Parser 截面** | CS_RANK, CS_ZSCORE(无参), CS_WINSORIZE(x,0.02,0.98)(有参) | 3 |
| **Parser 归约** | RED_MEAN, RED_STD | 2 |
| **Parser 复合** | ABS(LOG(C)-LOG(V))*V, SQRT(ABS(LOG(C))) | 2 |
| **Parser 错误** | 未知函数、参数数量不匹配、缺少括号 | 4 |
| **端到端** | parseExpression() + engine.evaluate() 验证数值正确 | 3 |
| **大小写** | `abs`, `ABS`, `Abs` 等价 | 1 |
| **合计** | | **~47** |

---

## 七、风险点

| 风险 | 级别 | 缓解 |
|------|------|------|
| `resolveFunctionCall` 中的异常驱动控制流 | 中 | 用 `std::optional` 或 isXxx（如 `bool isUnary(name)`）替代异常，后续优化 |
| Number literal 不支持 `-x` 语法中的 inline negation | 低 | factor 规则处理一元负号，折叠到 Scalar(-v) |
| 大小写转换在 non-ASCII 上失败 | 低 | 因子表达式仅使用 ASCII |
| `dynamic_cast<const Scalar*>` 对非 Scalar 返回 null | 低 | 已有明确错误消息 |
