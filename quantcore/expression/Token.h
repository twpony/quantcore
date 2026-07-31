// Token.h — token types and values for the expression lexer/parser
// Phase: 三期必实现
//
// Defines the Token structure and TokenType enum used by Lexer (string→tokens)
// and Parser (tokens→AST).  All token text is stored as std::string for
// simple lifetime management.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace quantcore {

// ============================================================
// Token type enum
// ============================================================

enum class TokenType : uint8_t {
    NUMBER,       // double literal
    IDENTIFIER,   // column name or function name
    FORMULA_REF,  // $name — reference to a registered formula
    LPAREN,       // '('
    RPAREN,       // ')'
    COMMA,        // ','
    PLUS,         // '+'
    MINUS,        // '-'
    STAR,         // '*'
    SLASH,        // '/'
    END,          // end of input
};

// ============================================================
// Token
// ============================================================

struct Token {
    TokenType   type;
    std::string text;         // original text span (lowercased for identifiers)
    double      numberValue;  // valid when type == NUMBER
    std::size_t pos;          // byte offset in original source (for error messages)

    // Default: END token
    Token() : type(TokenType::END), numberValue(0.0), pos(0) {}

    // Convenience constructors
    static Token number(double v, std::string_view txt, std::size_t p) noexcept {
        Token t;
        t.type        = TokenType::NUMBER;
        t.text        = std::string(txt);
        t.numberValue = v;
        t.pos         = p;
        return t;
    }

    static Token identifier(std::string_view txt, std::size_t p) noexcept {
        Token t;
        t.type = TokenType::IDENTIFIER;
        t.text = std::string(txt);
        t.pos  = p;
        return t;
    }

    static Token simple(TokenType ty, std::size_t p) noexcept {
        Token t;
        t.type = ty;
        t.pos  = p;
        switch (ty) {
            case TokenType::LPAREN: t.text = "(";  break;
            case TokenType::RPAREN: t.text = ")";  break;
            case TokenType::COMMA:  t.text = ",";  break;
            case TokenType::PLUS:   t.text = "+";  break;
            case TokenType::MINUS:  t.text = "-";  break;
            case TokenType::STAR:   t.text = "*";  break;
            case TokenType::SLASH:  t.text = "/";  break;
            case TokenType::END:    t.text = "<end>"; break;
            default: break;
        }
        return t;
    }
};

// ============================================================
// Token type name (for error messages)
// ============================================================

inline const char* tokenTypeName(TokenType t) noexcept {
    switch (t) {
        case TokenType::NUMBER:     return "number";
        case TokenType::IDENTIFIER:  return "identifier";
        case TokenType::FORMULA_REF: return "formula reference";
        case TokenType::LPAREN:     return "'('";
        case TokenType::RPAREN:     return "')'";
        case TokenType::COMMA:      return "','";
        case TokenType::PLUS:       return "'+'";
        case TokenType::MINUS:      return "'-'";
        case TokenType::STAR:       return "'*'";
        case TokenType::SLASH:      return "'/'";
        case TokenType::END:        return "<end>";
        default:                    return "<?>";
    }
}

}  // namespace quantcore
