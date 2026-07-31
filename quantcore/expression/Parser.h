// Parser.h — recursive-descent parser: token stream → expression AST
// Phase: 三期必实现
//
// Grammar (LL(1)):
//   expr    → term (('+' | '-') term)*
//   term    → factor (('*' | '/') factor)*
//   factor  → ('+' | '-') factor
//           | primary
//   primary → NUMBER
//           | IDENTIFIER '(' args ')'        (function call)
//           | IDENTIFIER                      (column reference)
//           | '(' expr ')'                    (grouping)
//   args    → ε | expr (',' expr)*
//
// Name resolution:
//   - Bare identifiers are matched against the 7 Field names (CLOSE, OPEN, ...)
//   - Function calls are resolved via OperatorRegistry in priority order:
//     Unary → Binary → Rolling → Red → Cs
//
// Usage:
//   auto ast = parseExpression("ABS(LOG(CLOSE) - LOG(VWAP)) * VOLUME");
//   ExecutionEngine engine;
//   Column<double> result = engine.evaluate(*ast, marketData);
#pragma once

#include <functional>
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
#include "quantcore/expression/BinaryRollingExpr.h"
#include "quantcore/expression/RedExpr.h"
#include "quantcore/expression/CsExpr.h"
#include "quantcore/expression/Lexer.h"
#include "quantcore/expression/Token.h"
#include "quantcore/registry/OperatorRegistry.h"

namespace quantcore {

// ============================================================
// Parser
// ============================================================

class Parser {
public:
    /// Callback type: formula name → expression string.
    /// Throws std::runtime_error if the formula is unknown.
    using FormulaResolver = std::function<std::string(const std::string&)>;

    /// Set a formula resolver for $name references.
    void setFormulaResolver(FormulaResolver resolver) {
        formulaResolver_ = std::move(resolver);
    }

    /// Parse a token stream into an expression AST.
    /// @throws std::runtime_error on syntax or resolution errors.
    std::unique_ptr<ExprNode> parse(const std::vector<Token>& tokens);

private:
    // Token stream navigation
    Token advance();
    const Token& peek()      const noexcept;
    const Token& peekAhead() const noexcept;
    bool check(TokenType t)  const noexcept;
    Token consume(TokenType t, const char* msg);
    bool isAtEnd()           const noexcept;

    // Grammar productions (recursive descent)
    std::unique_ptr<ExprNode> parseExpr();
    std::unique_ptr<ExprNode> parseTerm();
    std::unique_ptr<ExprNode> parseFactor();
    std::unique_ptr<ExprNode> parsePrimary();

    // Argument list: expr (',' expr)*
    std::vector<std::unique_ptr<ExprNode>> parseArgs();

    // Name resolution
    std::unique_ptr<ExprNode> resolveColumnRef(const std::string& name,
                                                std::size_t pos);
    std::unique_ptr<ExprNode> resolveFunctionCall(
        const std::string& name,
        std::vector<std::unique_ptr<ExprNode>> args,
        std::size_t pos);

    // Extract a numeric literal from an expression (for window sizes etc.)
    static double extractNumber(const ExprNode* node, const std::string& ctx,
                                std::size_t pos);

    const std::vector<Token>* tokens_ = nullptr;
    std::size_t idx_ = 0;
    FormulaResolver formulaResolver_;
};

// ============================================================
// Token stream navigation
// ============================================================

inline std::unique_ptr<ExprNode> Parser::parse(const std::vector<Token>& tokens) {
    tokens_ = &tokens;
    idx_    = 0;
    auto expr = parseExpr();
    if (!check(TokenType::END)) {
        throw std::runtime_error(
            "Parser: unexpected token '" + peek().text
            + "' after expression at position "
            + std::to_string(peek().pos));
    }
    return expr;
}

inline Token Parser::advance() {
    return (*tokens_)[idx_++];
}

inline const Token& Parser::peek() const noexcept {
    return (*tokens_)[idx_];
}

inline const Token& Parser::peekAhead() const noexcept {
    return (idx_ + 1 < tokens_->size())
        ? (*tokens_)[idx_ + 1]
        : (*tokens_)[idx_];  // fallback: return current (END)
}

inline bool Parser::check(TokenType t) const noexcept {
    return peek().type == t;
}

inline Token Parser::consume(TokenType t, const char* msg) {
    if (check(t)) return advance();
    throw std::runtime_error(
        std::string("Parser: ") + msg + " at position "
        + std::to_string(peek().pos) + " (found '" + peek().text + "')");
}

inline bool Parser::isAtEnd() const noexcept {
    return check(TokenType::END);
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

        left = std::make_unique<BinaryExpr>(code,
            std::move(left), std::move(right));
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

        left = std::make_unique<BinaryExpr>(code,
            std::move(left), std::move(right));
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

    // Unary minus
    if (check(TokenType::MINUS)) {
        Token op = advance();

        // -3.14 → Scalar(-3.14)
        if (check(TokenType::NUMBER)) {
            Token num = advance();
            return std::make_unique<Scalar>(-num.numberValue);
        }

        // -x → NEG(x)
        auto operand = parseFactor();
        return std::make_unique<UnaryExpr>(UnaryOpCode::NEG,
            std::move(operand));
    }

    return parsePrimary();
}

// primary → NUMBER | IDENTIFIER '(' args ')' | IDENTIFIER | '(' expr ')'
inline std::unique_ptr<ExprNode> Parser::parsePrimary() {
    if (check(TokenType::NUMBER)) {
        Token t = advance();
        return std::make_unique<Scalar>(t.numberValue);
    }

    // Formula reference: $name
    if (check(TokenType::FORMULA_REF)) {
        Token ref = advance();
        if (!formulaResolver_) {
            throw std::runtime_error(
                "Parser: formula reference '$" + ref.text
                + "' at position " + std::to_string(ref.pos)
                + " — no formula resolver available");
        }
        std::string subExpr = formulaResolver_(ref.text);
        // Re-parse the resolved expression inline
        Lexer subLexer;
        auto subTokens = subLexer.tokenize(subExpr);
        Parser subParser;
        subParser.setFormulaResolver(formulaResolver_);
        return subParser.parse(subTokens);
    }

    if (check(TokenType::IDENTIFIER)) {
        Token name = advance();

        // Function call: NAME '(' ... ')'
        if (check(TokenType::LPAREN)) {
            advance();  // consume '('
            auto args = parseArgs();
            consume(TokenType::RPAREN, "expected ')' after function arguments");
            return resolveFunctionCall(name.text, std::move(args), name.pos);
        }

        // Column reference
        return resolveColumnRef(name.text, name.pos);
    }

    // Grouping: '(' expr ')'
    if (check(TokenType::LPAREN)) {
        advance();  // consume '('
        auto expr = parseExpr();
        consume(TokenType::RPAREN, "expected ')' after expression");
        return expr;
    }

    throw std::runtime_error(
        std::string("Parser: unexpected token '") + peek().text
        + "' at position " + std::to_string(peek().pos)
        + ". Expected a number, identifier, or '('.");
}

// args → ε | expr (',' expr)*
inline std::vector<std::unique_ptr<ExprNode>> Parser::parseArgs() {
    std::vector<std::unique_ptr<ExprNode>> args;

    if (check(TokenType::RPAREN)) return args;

    args.push_back(parseExpr());

    while (check(TokenType::COMMA)) {
        advance();
        args.push_back(parseExpr());
    }

    return args;
}

// ============================================================
// Name resolution
// ============================================================

inline std::unique_ptr<ExprNode> Parser::resolveColumnRef(
        const std::string& name, std::size_t pos) {
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
        "Parser: unknown identifier '" + name
        + "' at position " + std::to_string(pos)
        + ". Valid column names: OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT, VWAP.");
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

    // --- Try unary ---
    try {
        UnaryOpCode code = reg.findUnary(name);
        if (args.size() != 1) {
            throw std::runtime_error(
                "Parser: unary function '" + name + "' takes 1 argument, "
                + std::to_string(args.size()) + " given at position "
                + std::to_string(pos));
        }
        return std::make_unique<UnaryExpr>(code, std::move(args[0]));
    } catch (const std::runtime_error& e) {
        // Re-throw arg-count errors; swallow "Unknown unary operator"
        if (std::string(e.what()).find("takes 1 argument") != std::string::npos) {
            throw;
        }
    }

    // --- Try binary (function-call form: MAX(x, y)) ---
    try {
        BinaryOpCode code = reg.findBinary(name);
        if (args.size() != 2) {
            throw std::runtime_error(
                "Parser: binary function '" + name + "' takes 2 arguments, "
                + std::to_string(args.size()) + " given at position "
                + std::to_string(pos));
        }
        return std::make_unique<BinaryExpr>(code,
            std::move(args[0]), std::move(args[1]));
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find("takes 2 arguments") != std::string::npos) {
            throw;
        }
    }

    // --- Try rolling ---
    try {
        RollingOpCode code = reg.findRolling(name);

        // Binary rolling operators (corr, cov): need 3+ args (expr1, expr2, window, ...)
        if (reg.isBinaryRolling(code)) {
            if (args.size() < 3) {
                throw std::runtime_error(
                    "Parser: binary rolling function '" + name
                    + "' requires at least 3 arguments (expression1, expression2, window_size)"
                    + " at position " + std::to_string(pos));
            }

            double window = extractNumber(args[2].get(), "window size", pos);
            auto w = static_cast<std::size_t>(window);
            if (w == 0 || static_cast<double>(w) != window) {
                throw std::runtime_error(
                    "Parser: window size must be a positive integer, got "
                    + std::to_string(window) + " at position " + std::to_string(pos));
            }

            // Collect extra params (e.g., ddof for rolling_cov)
            std::vector<double> extraParams;
            for (std::size_t i = 3; i < args.size(); ++i) {
                extraParams.push_back(
                    extractNumber(args[i].get(),
                        "extra parameter " + std::to_string(i - 2), pos));
            }

            return std::make_unique<BinaryRollingExpr>(code, w,
                std::move(args[0]), std::move(args[1]), std::move(extraParams));
        }

        // Unary rolling operators: need 2+ args (expr, window, ...)
        if (args.size() < 2) {
            throw std::runtime_error(
                "Parser: rolling function '" + name
                + "' requires at least 2 arguments (expression, window_size)"
                + " at position " + std::to_string(pos));
        }

        double window = extractNumber(args[1].get(), "window size", pos);
        auto w = static_cast<std::size_t>(window);
        if (w == 0 || static_cast<double>(w) != window) {
            throw std::runtime_error(
                "Parser: window size must be a positive integer, got "
                + std::to_string(window) + " at position " + std::to_string(pos));
        }

        // Collect extra params (e.g., q for rolling_quantile)
        std::vector<double> extraParams;
        for (std::size_t i = 2; i < args.size(); ++i) {
            extraParams.push_back(
                extractNumber(args[i].get(),
                    "extra parameter " + std::to_string(i - 1), pos));
        }

        return std::make_unique<RollingExpr>(code, w, std::move(args[0]),
                                              std::move(extraParams));
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find("requires at least 2 arguments") != std::string::npos
            || std::string(e.what()).find("requires at least 3 arguments") != std::string::npos
            || std::string(e.what()).find("window size must be") != std::string::npos
            || std::string(e.what()).find("numeric literal") != std::string::npos) {
            throw;
        }
    }

    // --- Try Red ---
    try {
        RedOpCode code = reg.findRed(name);
        std::vector<double> extraParams;
        for (std::size_t i = 1; i < args.size(); ++i) {
            extraParams.push_back(
                extractNumber(args[i].get(), "extra parameter " + std::to_string(i), pos));
        }
        return std::make_unique<RedExpr>(code, std::move(args[0]),
                                          std::move(extraParams));
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find("numeric literal") != std::string::npos) throw;
    }

    // --- Try CS ---
    try {
        CsOpCode code = reg.findCs(name);
        std::vector<double> extraParams;
        for (std::size_t i = 1; i < args.size(); ++i) {
            extraParams.push_back(
                extractNumber(args[i].get(), "extra parameter " + std::to_string(i), pos));
        }
        return std::make_unique<CsExpr>(code, std::move(args[0]),
                                         std::move(extraParams));
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find("numeric literal") != std::string::npos) throw;
    }

    throw std::runtime_error(
        "Parser: unknown function '" + name + "' at position "
        + std::to_string(pos));
}

inline double Parser::extractNumber(const ExprNode* node,
                                     const std::string& ctx,
                                     std::size_t pos) {
    const Scalar* s = dynamic_cast<const Scalar*>(node);
    if (s) return s->value();

    throw std::runtime_error(
        "Parser: " + ctx + " must be a numeric literal at position "
        + std::to_string(pos));
}

// ============================================================
// Top-level convenience function
// ============================================================

/// Parse an expression string into an AST.  Combines Lexer + Parser.
/// @param resolver Optional callback to resolve $name formula references.
/// @throws std::runtime_error on lexer or parser errors.
inline std::unique_ptr<ExprNode> parseExpression(
        const std::string& source,
        Parser::FormulaResolver resolver = {}) {
    Lexer lexer;
    auto tokens = lexer.tokenize(source);
    Parser parser;
    if (resolver) parser.setFormulaResolver(std::move(resolver));
    return parser.parse(tokens);
}

}  // namespace quantcore
