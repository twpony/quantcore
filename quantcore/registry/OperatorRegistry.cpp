// OperatorRegistry.cpp — name-to-enum lookup + name-based dispatch
// Phase: 一期基础实现
//
// Provides:
//   1. Enum→name reverse mappings (for logging / metrics)
//   2. Name→enum lookup (findUnary / findBinary / ...)
//   3. Name→evaluate dispatch (invokeUnary / invokeBinary)
//
// Each concrete operator is registered once at startup via
// registerUnary<OpType>(name, code) / registerBinary<OpType>(name, code) /
// registerRolling<OpType>(name, code) / registerRed<OpType>(name, code),
// which stores both the name→code mapping AND a type-erased function
// pointer for invoke(...) dispatch.

#include "quantcore/registry/OperatorRegistry.h"

// -- unary operator headers (for dispatch registration) --
#include "quantcore/operators/unary/Abs.h"
#include "quantcore/operators/unary/Exp.h"
#include "quantcore/operators/unary/Inv.h"
#include "quantcore/operators/unary/Log.h"
#include "quantcore/operators/unary/Log10.h"
#include "quantcore/operators/unary/Log2.h"
#include "quantcore/operators/unary/Neg.h"
#include "quantcore/operators/unary/Not.h"
#include "quantcore/operators/unary/Rank.h"
#include "quantcore/operators/unary/Sign.h"
#include "quantcore/operators/unary/Sqrt.h"
#include "quantcore/operators/unary/Square.h"

// -- rolling operator headers (for name registration) --
#include "quantcore/operators/rolling/RollingArgMax.h"
#include "quantcore/operators/rolling/RollingArgMin.h"
#include "quantcore/operators/rolling/RollingDiff.h"
#include "quantcore/operators/rolling/RollingEma.h"
#include "quantcore/operators/rolling/RollingKurt.h"
#include "quantcore/operators/rolling/RollingMax.h"
#include "quantcore/operators/rolling/RollingMean.h"
#include "quantcore/operators/rolling/RollingMedian.h"
#include "quantcore/operators/rolling/RollingMin.h"
#include "quantcore/operators/rolling/RollingMul.h"
#include "quantcore/operators/rolling/RollingQuantile.h"
#include "quantcore/operators/rolling/RollingRank.h"
#include "quantcore/operators/rolling/RollingShift.h"
#include "quantcore/operators/rolling/RollingSkew.h"
#include "quantcore/operators/rolling/RollingSma.h"
#include "quantcore/operators/rolling/RollingStd.h"
#include "quantcore/operators/rolling/RollingSum.h"
#include "quantcore/operators/rolling/RollingVar.h"

// -- red operator headers (for name registration) --
#include "quantcore/operators/red/RedMax.h"
#include "quantcore/operators/red/RedMean.h"
#include "quantcore/operators/red/RedMedian.h"
#include "quantcore/operators/red/RedMin.h"
#include "quantcore/operators/red/RedMul.h"
#include "quantcore/operators/red/RedQuantile.h"
#include "quantcore/operators/red/RedStd.h"
#include "quantcore/operators/red/RedSum.h"
#include "quantcore/operators/red/RedVar.h"
#include "quantcore/operators/red/RedZScore.h"

// -- cs operator headers (for name registration) --
#include "quantcore/operators/cs/CsClip.h"
#include "quantcore/operators/cs/CsDemean.h"
#include "quantcore/operators/cs/CsNormalize.h"
#include "quantcore/operators/cs/CsNormalizeL1.h"
#include "quantcore/operators/cs/CsNormalizeL2.h"
#include "quantcore/operators/cs/CsQuantile.h"
#include "quantcore/operators/cs/CsRank.h"
#include "quantcore/operators/cs/CsWinsorize.h"
#include "quantcore/operators/cs/CsWinsorizeMAD.h"
#include "quantcore/operators/cs/CsZScore.h"

// -- binary operator headers (for dispatch registration) --
#include "quantcore/operators/binary/Add.h"
#include "quantcore/operators/binary/Div.h"
#include "quantcore/operators/binary/Eq.h"
#include "quantcore/operators/binary/Gt.h"
#include "quantcore/operators/binary/Lt.h"
#include "quantcore/operators/binary/Neq.h"
#include "quantcore/operators/binary/Max.h"
#include "quantcore/operators/binary/Min.h"
#include "quantcore/operators/binary/Mul.h"
#include "quantcore/operators/binary/Sub.h"

#include <stdexcept>

namespace quantcore {

// ============================================================
// Singleton
// ============================================================

OperatorRegistry& OperatorRegistry::instance() {
    static OperatorRegistry registry;
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        // -- Unary operators --
        registry.registerUnary<AbsOp>   ("abs",    UnaryOpCode::ABS);
        registry.registerUnary<LogOp>   ("log",    UnaryOpCode::LOG);
        registry.registerUnary<Log10Op> ("log10",  UnaryOpCode::LOG10);
        registry.registerUnary<Log2Op>  ("log2",   UnaryOpCode::LOG2);
        registry.registerUnary<SqrtOp>  ("sqrt",   UnaryOpCode::SQRT);
        registry.registerUnary<NegOp>   ("neg",    UnaryOpCode::NEG);
        registry.registerUnary<SignOp>  ("sign",   UnaryOpCode::SIGN);
        registry.registerUnary<SquareOp>("square", UnaryOpCode::SQUARE);
        registry.registerUnary<ExpOp>   ("exp",    UnaryOpCode::EXP);
        registry.registerUnary<InvOp>   ("inv",    UnaryOpCode::INV);
        registry.registerUnary<NotOp>   ("not",    UnaryOpCode::NOT);
        registry.registerUnary<RankOp>           ("rank",             UnaryOpCode::RANK);
        registry.registerUnary<RankPctOp>        ("rank_pct",         UnaryOpCode::RANK_PCT);
        registry.registerUnary<RankNormalizedOp> ("rank_normalized",  UnaryOpCode::RANK_NORMALIZED);

        // -- Binary operators --
        registry.registerBinary<AddOp>("add", BinaryOpCode::ADD);
        registry.registerBinary<SubOp>("sub", BinaryOpCode::SUB);
        registry.registerBinary<MulOp>("mul", BinaryOpCode::MUL);
        registry.registerBinary<DivOp>("div", BinaryOpCode::DIV);
        registry.registerBinary<MaxOp>("max", BinaryOpCode::MAX);
        registry.registerBinary<MinOp>("min", BinaryOpCode::MIN);
        registry.registerBinary<GtOp> ("gt",  BinaryOpCode::GT);
        registry.registerBinary<LtOp> ("lt",  BinaryOpCode::LT);
        registry.registerBinary<EqOp> ("eq",  BinaryOpCode::EQ);
        registry.registerBinary<NeqOp>("neq", BinaryOpCode::NEQ);

        // -- Rolling operators --
        registry.registerRolling<RollingSmaOp>     ("rolling_sma",      RollingOpCode::ROLLING_SMA);
        registry.registerRolling<RollingEmaOp>     ("rolling_ema",      RollingOpCode::ROLLING_EMA);
        registry.registerRolling<RollingMaxOp>     ("rolling_max",      RollingOpCode::ROLLING_MAX);
        registry.registerRolling<RollingMinOp>     ("rolling_min",      RollingOpCode::ROLLING_MIN);
        registry.registerRolling<RollingStdOp>     ("rolling_std",      RollingOpCode::ROLLING_STD);
        registry.registerRolling<RollingSumOp>     ("rolling_sum",      RollingOpCode::ROLLING_SUM);
        registry.registerRolling<RollingRankOp>    ("rolling_rank",     RollingOpCode::ROLLING_RANK);
        registry.registerRolling<RollingDiffOp>    ("rolling_diff",     RollingOpCode::ROLLING_DIFF);
        registry.registerRolling<RollingShiftOp>   ("rolling_shift",    RollingOpCode::ROLLING_SHIFT);
        registry.registerRolling<RollingMeanOp>    ("rolling_mean",     RollingOpCode::ROLLING_MEAN);
        registry.registerRolling<RollingVarOp>     ("rolling_var",      RollingOpCode::ROLLING_VAR);
        registry.registerRolling<RollingMedianOp>  ("rolling_median",   RollingOpCode::ROLLING_MEDIAN);
        registry.registerRolling<RollingMulOp>     ("rolling_mul",      RollingOpCode::ROLLING_MUL);
        registry.registerRolling<RollingArgMaxOp>  ("rolling_argmax",   RollingOpCode::ROLLING_ARGMAX);
        registry.registerRolling<RollingArgMinOp>  ("rolling_argmin",   RollingOpCode::ROLLING_ARGMIN);
        registry.registerRolling<RollingKurtOp>    ("rolling_kurt",     RollingOpCode::ROLLING_KURT);
        registry.registerRolling<RollingSkewOp>    ("rolling_skew",     RollingOpCode::ROLLING_SKEW);
        registry.registerRolling<RollingQuantileOp>("rolling_quantile", RollingOpCode::ROLLING_QUANTILE);

        // -- Red operators --
        registry.registerRed<RedSumOp>     ("red_sum",      RedOpCode::RED_SUM);
        registry.registerRed<RedMeanOp>    ("red_mean",     RedOpCode::RED_MEAN);
        registry.registerRed<RedStdOp>     ("red_std",      RedOpCode::RED_STD);
        registry.registerRed<RedVarOp>     ("red_var",      RedOpCode::RED_VAR);
        registry.registerRed<RedMinOp>     ("red_min",      RedOpCode::RED_MIN);
        registry.registerRed<RedMaxOp>     ("red_max",      RedOpCode::RED_MAX);
        registry.registerRed<RedMulOp>     ("red_mul",      RedOpCode::RED_MUL);
        registry.registerRed<RedMedianOp>  ("red_median",   RedOpCode::RED_MEDIAN);
        registry.registerRed<RedZScoreOp>  ("red_zscore",   RedOpCode::RED_ZSCORE);
        registry.registerRed<RedQuantileOp>("red_quantile", RedOpCode::RED_QUANTILE);

        // -- CS operators --
        registry.registerCs<CsRankOp>      ("cs_rank",       CsOpCode::CS_RANK);
        registry.registerCs<CsQuantileOp>  ("cs_quantile",   CsOpCode::CS_QUANTILE);
        registry.registerCs<CsZScoreOp>    ("cs_zscore",     CsOpCode::CS_ZSCORE);
        registry.registerCs<CsNormalizeOp> ("cs_normalize",  CsOpCode::CS_NORMALIZE);
        registry.registerCs<CsNormalizeL1Op> ("cs_normalize_l1",  CsOpCode::CS_NORMALIZE_L1);
        registry.registerCs<CsNormalizeL2Op> ("cs_normalize_l2",  CsOpCode::CS_NORMALIZE_L2);
        registry.registerCs<CsWinsorizeOp>    ("cs_winsorize",     CsOpCode::CS_WINSORIZE);
        registry.registerCs<CsWinsorizeMADOp> ("cs_winsorize_mad", CsOpCode::CS_WINSORIZE_MAD);
        registry.registerCs<CsClipOp>         ("cs_clip",          CsOpCode::CS_CLIP);
        registry.registerCs<CsDemeanOp>    ("cs_demean",     CsOpCode::CS_DEMEAN);
    }
    return registry;
}

// ============================================================
// Enum → name maps (reverse lookup for logging / metrics)
// ============================================================

namespace {

const char* const kUnaryOpNames[] = {
    "abs",     // ABS
    "log",     // LOG
    "log10",   // LOG10
    "log2",    // LOG2
    "sqrt",    // SQRT
    "neg",     // NEG
    "sign",    // SIGN
    "square",  // SQUARE
    "exp",     // EXP
    "inv",     // INV
    "not",     // NOT
    "rank",             // RANK
    "rank_pct",         // RANK_PCT
    "rank_normalized",  // RANK_NORMALIZED
};

const char* const kBinaryOpNames[] = {
    "add",  // ADD
    "sub",  // SUB
    "mul",  // MUL
    "div",  // DIV
    "max",  // MAX
    "min",  // MIN
    "gt",   // GT
    "lt",   // LT
    "eq",   // EQ
    "neq",  // NEQ
};

const char* const kRollingOpNames[] = {
    "rolling_sma",      // ROLLING_SMA
    "rolling_ema",      // ROLLING_EMA
    "rolling_max",      // ROLLING_MAX
    "rolling_min",      // ROLLING_MIN
    "rolling_std",      // ROLLING_STD
    "rolling_sum",      // ROLLING_SUM
    "rolling_rank",     // ROLLING_RANK
    "rolling_diff",     // ROLLING_DIFF
    "rolling_shift",    // ROLLING_SHIFT
    "rolling_mean",     // ROLLING_MEAN
    "rolling_var",      // ROLLING_VAR
    "rolling_median",   // ROLLING_MEDIAN
    "rolling_mul",      // ROLLING_MUL
    "rolling_argmax",   // ROLLING_ARGMAX
    "rolling_argmin",   // ROLLING_ARGMIN
    "rolling_kurt",     // ROLLING_KURT
    "rolling_skew",     // ROLLING_SKEW
    "rolling_quantile", // ROLLING_QUANTILE
};

const char* const kRedOpNames[] = {
    "red_sum",      // RED_SUM
    "red_mean",     // RED_MEAN
    "red_std",      // RED_STD
    "red_var",      // RED_VAR
    "red_min",      // RED_MIN
    "red_max",      // RED_MAX
    "red_mul",      // RED_MUL
    "red_median",   // RED_MEDIAN
    "red_zscore",   // RED_ZSCORE
    "red_quantile", // RED_QUANTILE
};

const char* const kCsOpNames[] = {
    "cs_rank",       // CS_RANK
    "cs_quantile",   // CS_QUANTILE
    "cs_zscore",     // CS_ZSCORE
    "cs_normalize",  // CS_NORMALIZE
    "cs_normalize_l1",  // CS_NORMALIZE_L1
    "cs_normalize_l2",  // CS_NORMALIZE_L2
    "cs_winsorize",     // CS_WINSORIZE
    "cs_winsorize_mad", // CS_WINSORIZE_MAD
    "cs_clip",          // CS_CLIP
    "cs_demean",     // CS_DEMEAN
};

}  // anonymous namespace

// ============================================================
// Name utility functions (declared in Types.h)
// ============================================================

const char* fieldName(Field field) {
    static const char* const names[] = {
        "open", "high", "low", "close", "volume", "amount", "vwap"
    };
    auto idx = static_cast<std::size_t>(field);
    if (idx >= static_cast<std::size_t>(Field::kFieldCount)) return "unknown";
    return names[idx];
}

const char* unaryOpName(UnaryOpCode op) {
    auto idx = static_cast<std::size_t>(op);
    if (idx >= static_cast<std::size_t>(UnaryOpCode::kCount)) return "unknown";
    return kUnaryOpNames[idx];
}

const char* binaryOpName(BinaryOpCode op) {
    auto idx = static_cast<std::size_t>(op);
    if (idx >= static_cast<std::size_t>(BinaryOpCode::kCount)) return "unknown";
    return kBinaryOpNames[idx];
}

const char* rollingOpName(RollingOpCode op) {
    auto idx = static_cast<std::size_t>(op);
    if (idx >= static_cast<std::size_t>(RollingOpCode::kCount)) return "unknown";
    return kRollingOpNames[idx];
}

const char* redOpName(RedOpCode op) {
    auto idx = static_cast<std::size_t>(op);
    if (idx >= static_cast<std::size_t>(RedOpCode::kCount)) return "unknown";
    return kRedOpNames[idx];
}

const char* csOpName(CsOpCode op) {
    auto idx = static_cast<std::size_t>(op);
    if (idx >= static_cast<std::size_t>(CsOpCode::kCount)) return "unknown";
    return kCsOpNames[idx];
}

const char* simdLevelName(SimdLevel level) {
    switch (level) {
        case SimdLevel::SCALAR: return "scalar";
        case SimdLevel::SSE42:  return "sse4.2";
        case SimdLevel::AVX2:   return "avx2";
        case SimdLevel::AVX512: return "avx-512";
        default:                return "unknown";
    }
}

// ============================================================
// OperatorRegistry lookup methods
// ============================================================

UnaryOpCode OperatorRegistry::findUnary(const std::string& name) const {
    auto it = unaryRegistry_.find(name);
    if (it == unaryRegistry_.end()) {
        throw std::runtime_error("Unknown unary operator: " + name);
    }
    return it->second;
}

BinaryOpCode OperatorRegistry::findBinary(const std::string& name) const {
    auto it = binaryRegistry_.find(name);
    if (it == binaryRegistry_.end()) {
        throw std::runtime_error("Unknown binary operator: " + name);
    }
    return it->second;
}

RollingOpCode OperatorRegistry::findRolling(const std::string& name) const {
    auto it = rollingRegistry_.find(name);
    if (it == rollingRegistry_.end()) {
        throw std::runtime_error("Unknown rolling operator: " + name);
    }
    return it->second;
}

RedOpCode OperatorRegistry::findRed(const std::string& name) const {
    auto it = redRegistry_.find(name);
    if (it == redRegistry_.end()) {
        throw std::runtime_error("Unknown Red operator: " + name);
    }
    return it->second;
}

std::vector<std::string> OperatorRegistry::listRed() const {
    std::vector<std::string> names;
    for (const auto& [name, code] : redRegistry_) {
        names.push_back(name);
    }
    return names;
}

CsOpCode OperatorRegistry::findCs(const std::string& name) const {
    auto it = csRegistry_.find(name);
    if (it == csRegistry_.end()) {
        throw std::runtime_error("Unknown CS operator: " + name);
    }
    return it->second;
}

std::vector<std::string> OperatorRegistry::listCs() const {
    std::vector<std::string> names;
    for (const auto& [name, code] : csRegistry_) {
        names.push_back(name);
    }
    return names;
}

std::vector<std::string> OperatorRegistry::listUnary() const {
    std::vector<std::string> names;
    for (const auto& [name, code] : unaryRegistry_) {
        names.push_back(name);
    }
    return names;
}

std::vector<std::string> OperatorRegistry::listBinary() const {
    std::vector<std::string> names;
    for (const auto& [name, code] : binaryRegistry_) {
        names.push_back(name);
    }
    return names;
}

std::vector<std::string> OperatorRegistry::listRolling() const {
    std::vector<std::string> names;
    for (const auto& [name, code] : rollingRegistry_) {
        names.push_back(name);
    }
    return names;
}


// ============================================================
// Name-based dispatch
// ============================================================

void OperatorRegistry::invokeUnary(const std::string& name,
                                    const Operand& input,
                                    double*        output,
                                    std::size_t    n,
                                    const uint64_t* nullMask) const
{
    auto it = unaryRegistry_.find(name);
    if (it == unaryRegistry_.end()) {
        throw std::runtime_error("Unknown unary operator: " + name);
    }
    invokeUnary(it->second, input, output, n, nullMask);
}

void OperatorRegistry::invokeUnary(UnaryOpCode code,
                                    const Operand& input,
                                    double*        output,
                                    std::size_t    n,
                                    const uint64_t* nullMask) const
{
    auto idx = static_cast<std::size_t>(code);
    if (idx >= unaryDispatch_.size() || unaryDispatch_[idx] == nullptr) {
        throw std::runtime_error(
            "Unary operator not registered for dispatch: "
            + std::string(unaryOpName(code)));
    }
    unaryDispatch_[idx](input, output, n, nullMask);
}

void OperatorRegistry::invokeBinary(const std::string& name,
                                     const Operand& lhs,
                                     const Operand& rhs,
                                     double*        output,
                                     std::size_t    n,
                                     const uint64_t* nullMask) const
{
    auto it = binaryRegistry_.find(name);
    if (it == binaryRegistry_.end()) {
        throw std::runtime_error("Unknown binary operator: " + name);
    }
    invokeBinary(it->second, lhs, rhs, output, n, nullMask);
}

void OperatorRegistry::invokeBinary(BinaryOpCode code,
                                     const Operand& lhs,
                                     const Operand& rhs,
                                     double*        output,
                                     std::size_t    n,
                                     const uint64_t* nullMask) const
{
    auto idx = static_cast<std::size_t>(code);
    if (idx >= binaryDispatch_.size() || binaryDispatch_[idx] == nullptr) {
        throw std::runtime_error(
            "Binary operator not registered for dispatch: "
            + std::string(binaryOpName(code)));
    }
    binaryDispatch_[idx](lhs, rhs, output, n, nullMask);
}

}  // namespace quantcore
