// FactorMacros.h — reusable macros for defining expression-based factors
//
// Each factor previously needed ~50 lines across a .h / .cpp pair.  These
// macros collapse boilerplate to 1 line each.
//
// ============================================================
// Usage
// ============================================================
//
//   // === In the header file (.h) ===
//   #include "factors/FactorMacros.h"
//   namespace quantcore { namespace factors {
//   DECLARE_FACTOR(Alpha0003)
//   }}
//
//   // === In the source file (.cpp) ===
//   #include "factors/Alpha0003.h"
//   namespace quantcore { namespace factors {
//   DEFINE_FACTOR(Alpha0003,
//       "alpha_0003",
//       "rolling_corr(close, vwap, 20)")
//   }}
//
// The macros expand to:
//   - kAlpha0003          → "alpha_0003"           (registration key)
//   - kAlpha0003Expr      → "rolling_corr(...)"    (expression string)
//   - RegisterAlpha0003(FactorCalculator&)         (register function)
//   - EvaluateAlpha0003(FactorCalculator&, const MarketData&) → Column<double>
//
// ============================================================
// Factor composition via $ref
// ============================================================
//
// Expressions can reference other factors by their registration key:
//   DEFINE_FACTOR(Alpha0009, "alpha_0009", "$alpha_0003 - $alpha_0001")
//
// ============================================================
// Project-level reuse — three options
// ============================================================
//
// 1. Link libquantcorefactors.so (shared library)
//    → #include "factors/alpha_0001.h", call calc.evaluate("alpha_0001", md)
//
// 2. Copy just the expression string (no linking needed)
//    → calc.registerFormula("my_factor", kAlpha0001Expr)
//
// 3. Reference via $name syntax in other expressions
//    → calc.registerFormula("composite", "$alpha_0001 + $alpha_0002")
#pragma once

#include <string>

#include "quantcore/core/FactorCalculator.h"

// ============================================================
// Token pasting helpers (two-level for macro argument expansion)
// ============================================================

#define QCF_JOIN_IMPL(a, b) a ## b
#define QCF_JOIN(a, b)      QCF_JOIN_IMPL(a, b)

// ============================================================
// DECLARE_FACTOR(cppSuffix)
// ============================================================
//
// Place inside  namespace quantcore { namespace factors { ... } }
// in the factor's .h file.
//
// Generates extern declarations for the registration key constant,
// expression string constant, Register* function, and Evaluate*
// convenience function.

#define DECLARE_FACTOR(cppSuffix)                                               \
    extern const std::string QCF_JOIN(k, cppSuffix);                            \
    extern const std::string QCF_JOIN(k, cppSuffix ## Expr);                    \
    void QCF_JOIN(Register, cppSuffix)(quantcore::FactorCalculator& calc);      \
    quantcore::Column<double> QCF_JOIN(Evaluate, cppSuffix)(                    \
        quantcore::FactorCalculator& calc,                                      \
        const quantcore::MarketData& md);

// ============================================================
// DEFINE_FACTOR(cppSuffix, registrationKey, expression)
// ============================================================
//
// Place inside  namespace quantcore { namespace factors { ... } }
// in the factor's .cpp file.
//
// @param cppSuffix       PascalCase suffix for generated C++ identifiers.
//                        e.g., Alpha0003 → kAlpha0003, RegisterAlpha0003
// @param registrationKey Quoted string — FactorCalculator key.
//                        e.g., "alpha_0003"
// @param expression      Quoted string — the formula expression.
//                        e.g., "close / rolling_shift(close, 20) - 1"

#define DEFINE_FACTOR(cppSuffix, registrationKey, expression)                   \
    const std::string QCF_JOIN(k, cppSuffix){registrationKey};                   \
    const std::string QCF_JOIN(k, cppSuffix ## Expr){expression};                \
    void QCF_JOIN(Register, cppSuffix)(quantcore::FactorCalculator& calc) {      \
        calc.registerFormula(registrationKey, expression);                       \
    }                                                                            \
    quantcore::Column<double> QCF_JOIN(Evaluate, cppSuffix)(                     \
            quantcore::FactorCalculator& calc,                                   \
            const quantcore::MarketData& md) {                                   \
        return calc.evaluate(registrationKey, md);                               \
    }
