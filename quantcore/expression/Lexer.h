// Lexer.h — string → token stream
// Phase: 三期必实现
//
// Hand-written lexer that tokenizes an expression string into a sequence
// of Token values.  Whitespace is skipped.  Identifiers are lowercased for
// case-insensitive matching.  Numbers support integer, decimal, and
// scientific notation (1e-3).
//
// Usage:
//   Lexer lexer;
//   std::vector<Token> tokens = lexer.tokenize("ABS(LOG(CLOSE))");
#pragma once

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "quantcore/expression/Token.h"

namespace quantcore {

class Lexer {
public:
    /// Tokenize an expression string.
    /// @throws std::runtime_error on invalid characters.
    std::vector<Token> tokenize(std::string_view input);

private:
    void skipWhitespace() noexcept;
    Token readNumber();
    Token readIdentifier();

    std::string_view input_;
    std::size_t      pos_ = 0;
};

// ============================================================
// Implementation (inline, header-only)
// ============================================================

inline std::vector<Token> Lexer::tokenize(std::string_view input) {
    input_ = input;
    pos_   = 0;
    std::vector<Token> tokens;

    while (pos_ < input_.size()) {
        char c = input_[pos_];

        if (std::isspace(static_cast<unsigned char>(c))) {
            skipWhitespace();
            continue;
        }

        if (std::isdigit(static_cast<unsigned char>(c))
            || (c == '.' && pos_ + 1 < input_.size()
                && std::isdigit(static_cast<unsigned char>(input_[pos_ + 1])))) {
            tokens.push_back(readNumber());
        } else if (c == '$') {
            // Formula reference: $name
            std::size_t start = pos_;
            ++pos_;  // consume '$'
            if (pos_ < input_.size()
                && (std::isalpha(static_cast<unsigned char>(input_[pos_]))
                    || input_[pos_] == '_')) {
                Token t = readIdentifier();
                t.type = TokenType::FORMULA_REF;
                t.pos = start;
                tokens.push_back(t);
            } else {
                throw std::runtime_error(
                    "Lexer: '$' must be followed by an identifier at position "
                    + std::to_string(start));
            }
        } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
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
                std::string("Lexer: unexpected character '") + c
                + "' at position " + std::to_string(pos_));
        }
    }

    tokens.push_back(Token::simple(TokenType::END, pos_));
    return tokens;
}

inline void Lexer::skipWhitespace() noexcept {
    while (pos_ < input_.size()
           && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
        ++pos_;
    }
}

inline Token Lexer::readNumber() {
    std::size_t start = pos_;
    char* end = nullptr;

    // Use strtod for robust number parsing.  It handles all formats:
    // integer, decimal, scientific notation.
    double val = std::strtod(input_.data() + pos_, &end);

    if (end == input_.data() + pos_) {
        // strtod couldn't parse anything — should not happen given our checks
        throw std::runtime_error(
            "Lexer: invalid number at position " + std::to_string(pos_));
    }

    pos_ = static_cast<std::size_t>(end - input_.data());
    std::string_view text = input_.substr(start, pos_ - start);
    return Token::number(val, text, start);
}

inline Token Lexer::readIdentifier() {
    std::size_t start = pos_;

    // First char already checked: alpha or underscore
    ++pos_;

    // Subsequent chars: alphanumeric or underscore
    while (pos_ < input_.size()
           && (std::isalnum(static_cast<unsigned char>(input_[pos_]))
               || input_[pos_] == '_')) {
        ++pos_;
    }

    // Lowercase for case-insensitive matching
    std::string lower;
    std::string_view text = input_.substr(start, pos_ - start);
    lower.reserve(text.size());
    for (char c : text) {
        lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    return Token::identifier(lower, start);
}

}  // namespace quantcore
