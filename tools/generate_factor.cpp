// generate_factor.cpp — AOT code generator for expression-based factors (C++ version)
//
// Uses the project's own Parser + OperatorRegistry to parse expressions,
// then walks the AST and emits .h/.cpp files with direct operator calls.
//
// No duplicate parser — reuses the existing C++ infrastructure.
//
// Build:
//   g++ -std=c++20 -I.. tools/generate_factor.cpp ../quantcore/*.cpp -o generate_factor
//
// Usage:
//   ./generate_factor alpha_0001 "close / rolling_shift(close, 20) - 1"
//   ./generate_factor alpha_0003 "log(close) - log(rolling_shift(close, 1))"

#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/expression/BinaryExpr.h"
#include "quantcore/expression/BinaryRollingExpr.h"
#include "quantcore/expression/ColumnRef.h"
#include "quantcore/expression/CsExpr.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/expression/Parser.h"
#include "quantcore/expression/RedExpr.h"
#include "quantcore/expression/RollingExpr.h"
#include "quantcore/expression/Scalar.h"
#include "quantcore/expression/UnaryExpr.h"
#include "quantcore/registry/OperatorRegistry.h"

using namespace quantcore;

// ============================================================
// Operator → C++ class name + include path
// ============================================================

struct OpInfo {
    std::string className;
    std::string includePath;
};

// Lazy-init map filled once
static std::unordered_map<UnaryOpCode, OpInfo> unaryMap;
static std::unordered_map<BinaryOpCode, OpInfo> binaryMap;
static std::unordered_map<RollingOpCode, OpInfo> rollingMap;
static std::unordered_map<RedOpCode, OpInfo> redMap;
static std::unordered_map<CsOpCode, OpInfo> csMap;

// helper: DivOp → Div, RollingShiftOp → RollingShift
static std::string headerName(std::string className) {
    if (className.size() > 2 && className.substr(className.size() - 2) == "Op")
        className.resize(className.size() - 2);
    return className + ".h";
}

static void initMaps() {
    static bool done = false;
    if (done) return;
    done = true;

    auto initUnary = [](UnaryOpCode c, const char* cls, const char* hdr = nullptr) {
        std::string path = hdr ? hdr :
            "quantcore/operators/unary/" + headerName(cls);
        unaryMap[c] = {cls, path};
    };
    initUnary(UnaryOpCode::ABS,    "AbsOp");
    initUnary(UnaryOpCode::LOG,    "LogOp");
    initUnary(UnaryOpCode::LOG10,  "Log10Op");
    initUnary(UnaryOpCode::LOG2,   "Log2Op");
    initUnary(UnaryOpCode::SQRT,   "SqrtOp");
    initUnary(UnaryOpCode::NEG,    "NegOp");
    initUnary(UnaryOpCode::SIGN,   "SignOp");
    initUnary(UnaryOpCode::SQUARE, "SquareOp");
    initUnary(UnaryOpCode::EXP,    "ExpOp");
    initUnary(UnaryOpCode::INV,    "InvOp");
    initUnary(UnaryOpCode::NOT,    "NotOp");
    initUnary(UnaryOpCode::RANK,             "RankOp", "quantcore/operators/unary/Rank.h");
    initUnary(UnaryOpCode::RANK_PCT,         "RankPctOp", "quantcore/operators/unary/Rank.h");
    initUnary(UnaryOpCode::RANK_NORMALIZED,  "RankNormalizedOp", "quantcore/operators/unary/Rank.h");

    // Binary
    auto initBinary = [](BinaryOpCode c, const char* cls) {
        binaryMap[c] = {cls, "quantcore/operators/binary/" + headerName(cls)};
    };
    initBinary(BinaryOpCode::ADD, "AddOp");
    initBinary(BinaryOpCode::SUB, "SubOp");
    initBinary(BinaryOpCode::MUL, "MulOp");
    initBinary(BinaryOpCode::DIV, "DivOp");
    initBinary(BinaryOpCode::MAX, "MaxOp");
    initBinary(BinaryOpCode::MIN, "MinOp");
    initBinary(BinaryOpCode::GT,  "GtOp");
    initBinary(BinaryOpCode::LT,  "LtOp");
    initBinary(BinaryOpCode::EQ,  "EqOp");
    initBinary(BinaryOpCode::NEQ, "NeqOp");

    // Rolling
    auto initRolling = [](RollingOpCode c, const char* cls) {
        rollingMap[c] = {cls, "quantcore/operators/rolling/" + headerName(cls)};
    };
    initRolling(RollingOpCode::ROLLING_SMA,      "RollingSmaOp");
    initRolling(RollingOpCode::ROLLING_EMA,      "RollingEmaOp");
    initRolling(RollingOpCode::ROLLING_MAX,      "RollingMaxOp");
    initRolling(RollingOpCode::ROLLING_MIN,      "RollingMinOp");
    initRolling(RollingOpCode::ROLLING_STD,      "RollingStdOp");
    initRolling(RollingOpCode::ROLLING_SUM,      "RollingSumOp");
    initRolling(RollingOpCode::ROLLING_RANK,     "RollingRankOp");
    initRolling(RollingOpCode::ROLLING_DIFF,     "RollingDiffOp");
    initRolling(RollingOpCode::ROLLING_SHIFT,    "RollingShiftOp");
    initRolling(RollingOpCode::ROLLING_MEAN,     "RollingMeanOp");
    initRolling(RollingOpCode::ROLLING_VAR,      "RollingVarOp");
    initRolling(RollingOpCode::ROLLING_MEDIAN,   "RollingMedianOp");
    initRolling(RollingOpCode::ROLLING_MUL,      "RollingMulOp");
    initRolling(RollingOpCode::ROLLING_ARGMAX,   "RollingArgMaxOp");
    initRolling(RollingOpCode::ROLLING_ARGMIN,   "RollingArgMinOp");
    initRolling(RollingOpCode::ROLLING_KURT,     "RollingKurtOp");
    initRolling(RollingOpCode::ROLLING_SKEW,     "RollingSkewOp");
    initRolling(RollingOpCode::ROLLING_QUANTILE, "RollingQuantileOp");
    initRolling(RollingOpCode::ROLLING_CORR,     "RollingCorrOp");
    initRolling(RollingOpCode::ROLLING_COV,      "RollingCovOp");

    auto initRed = [](RedOpCode c, const char* cls) {
        redMap[c] = {cls, "quantcore/operators/red/" + headerName(cls)};
    };
    initRed(RedOpCode::RED_SUM,      "RedSumOp");
    initRed(RedOpCode::RED_MEAN,     "RedMeanOp");
    initRed(RedOpCode::RED_STD,      "RedStdOp");
    initRed(RedOpCode::RED_VAR,      "RedVarOp");
    initRed(RedOpCode::RED_MIN,      "RedMinOp");
    initRed(RedOpCode::RED_MAX,      "RedMaxOp");
    initRed(RedOpCode::RED_MUL,      "RedMulOp");
    initRed(RedOpCode::RED_MEDIAN,   "RedMedianOp");
    initRed(RedOpCode::RED_ZSCORE,   "RedZScoreOp");
    initRed(RedOpCode::RED_QUANTILE, "RedQuantileOp");

    auto initCs = [](CsOpCode c, const char* cls) {
        csMap[c] = {cls, "quantcore/operators/cs/" + headerName(cls)};
    };
    initCs(CsOpCode::CS_RANK,       "CsRankOp");
    initCs(CsOpCode::CS_QUANTILE,   "CsQuantileOp");
    initCs(CsOpCode::CS_ZSCORE,     "CsZScoreOp");
    initCs(CsOpCode::CS_NORMALIZE,  "CsNormalizeOp");
    initCs(CsOpCode::CS_NORMALIZE_L1, "CsNormalizeL1Op");
    initCs(CsOpCode::CS_NORMALIZE_L2, "CsNormalizeL2Op");
    initCs(CsOpCode::CS_WINSORIZE,     "CsWinsorizeOp");
    initCs(CsOpCode::CS_WINSORIZE_MAD, "CsWinsorizeMADOp");
    initCs(CsOpCode::CS_CLIP,          "CsClipOp");
    initCs(CsOpCode::CS_DEMEAN,    "CsDemeanOp");
}

// ============================================================
// C++ code emitter
// ============================================================

struct EmitResult {
    std::string operandExpr;   // e.g. "Operand(buf0.data())"
    std::string dataExpr;      // e.g. "buf0.data()"
};

class CodeGenerator {
public:
    CodeGenerator(const std::string& factorName, const std::string& expression)
        : factorName_(factorName), expression_(expression) {}

    void generate(const std::string& hdrPath, const std::string& cppPath) {
        initMaps();

        // Parse expression using existing C++ parser
        auto ast = parseExpression(expression_);

        // Walk AST and emit computation code
        auto result = emitNode(ast.get());

        // Write output files
        writeHeader(hdrPath);
        writeCpp(cppPath, result);
    }

private:
    std::string factorName_;
    std::string expression_;
    int bufCounter_ = 0;
    int colCounter_ = 0;
    int opCounter_ = 0;
    std::vector<std::string> statements_;
    std::set<std::string> includes_;

    std::string newBuf()  { return "buf" + std::to_string(bufCounter_++); }
    std::string newCol()  { return "col" + std::to_string(colCounter_++); }
    std::string newOp()   { return "op" + std::to_string(opCounter_++); }

    std::string pascalCase() const {
        // alpha_0001 → Alpha0001
        std::string s;
        bool cap = true;
        for (char c : factorName_) {
            if (c == '_') { cap = true; continue; }
            s += cap ? static_cast<char>(std::toupper(c)) : c;
            cap = false;
        }
        return s;
    }

    EmitResult emitNode(const ExprNode* node) {
        // Column reference
        if (auto* col = dynamic_cast<const ColumnRef*>(node)) {
            std::string v = newCol();
            std::string fieldName = ::quantcore::fieldName(col->field());
            // Convert to uppercase for Field::XXXX
            for (auto& c : fieldName) c = std::toupper(c);
            statements_.push_back(
                "    const auto& " + v + " = md.column<double>(Field::" + fieldName + ");");
            return { "Operand(" + v + ".data())", v + ".data()" };
        }

        // Scalar
        if (auto* s = dynamic_cast<const Scalar*>(node)) {
            return { "Operand(" + std::to_string(s->value()) + ")",
                     "Operand(" + std::to_string(s->value()) + ")" };
        }

        // Unary expression
        if (auto* u = dynamic_cast<const UnaryExpr*>(node)) {
            auto child = emitNode(u->child());
            auto it = unaryMap.find(u->opCode());
            if (it == unaryMap.end()) throw std::runtime_error("Unknown unary op");
            includes_.insert(it->second.includePath);
            std::string buf = newBuf();
            statements_.push_back(
                "    auto " + buf + " = pool.allocate<double>(n);\n"
                "    " + it->second.className + "{}.evaluate(" +
                child.operandExpr + ", " + buf + ".data(), n, nullptr);");
            return { "Operand(" + buf + ".data())", buf + ".data()" };
        }

        // Binary expression
        if (auto* b = dynamic_cast<const BinaryExpr*>(node)) {
            auto lhs = emitNode(b->lhs());
            auto rhs = emitNode(b->rhs());
            auto it = binaryMap.find(b->opCode());
            if (it == binaryMap.end()) throw std::runtime_error("Unknown binary op");
            includes_.insert(it->second.includePath);
            std::string buf = newBuf();
            statements_.push_back(
                "    auto " + buf + " = pool.allocate<double>(n);\n"
                "    " + it->second.className + "{}.evaluate(" +
                lhs.operandExpr + ", " + rhs.operandExpr + ", " +
                buf + ".data(), n, nullptr);");
            return { "Operand(" + buf + ".data())", buf + ".data()" };
        }

        // Rolling expression
        if (auto* r = dynamic_cast<const RollingExpr*>(node)) {
            auto child = emitNode(r->child());
            auto it = rollingMap.find(r->opCode());
            if (it == rollingMap.end()) throw std::runtime_error("Unknown rolling op");
            includes_.insert(it->second.includePath);
            std::string buf = newBuf();
            std::string opVar = newOp();
            std::string extraArgs;
            for (auto p : r->extraParams())
                extraArgs += ", " + std::to_string(p);
            statements_.push_back(
                "    auto " + buf + " = pool.allocate<double>(n);\n"
                "    " + it->second.className + " " + opVar + "(" +
                std::to_string(r->window()) + extraArgs + ");\n"
                "    " + opVar + ".evaluate(ColView<double>(" +
                child.dataExpr + ", n), " + buf + ".data());");
            return { "Operand(" + buf + ".data())", buf + ".data()" };
        }

        // Binary rolling expression
        if (auto* br = dynamic_cast<const BinaryRollingExpr*>(node)) {
            auto c1 = emitNode(br->child1());
            auto c2 = emitNode(br->child2());
            auto it = rollingMap.find(br->opCode());
            if (it == rollingMap.end()) throw std::runtime_error("Unknown binary rolling op");
            includes_.insert(it->second.includePath);
            std::string buf = newBuf();
            std::string opVar = newOp();
            statements_.push_back(
                "    auto " + buf + " = pool.allocate<double>(n);\n"
                "    " + it->second.className + " " + opVar + "(" +
                std::to_string(br->window()) + ");\n"
                "    " + opVar + ".evaluate(\n"
                "        ColView<double>(" + c1.dataExpr + ", n),\n"
                "        ColView<double>(" + c2.dataExpr + ", n),\n"
                "        " + buf + ".data());");
            return { "Operand(" + buf + ".data())", buf + ".data()" };
        }

        // Red / CS expressions — fallback
        if (auto* red = dynamic_cast<const RedExpr*>(node)) {
            auto child = emitNode(red->child());
            auto it = redMap.find(red->opCode());
            if (it == redMap.end()) throw std::runtime_error("Unknown red op");
            includes_.insert(it->second.includePath);
            std::string buf = newBuf();
            statements_.push_back(
                "    auto " + buf + " = pool.allocate<double>(n);\n"
                "    " + it->second.className + "{}.evaluate(ColView<double>(" +
                child.dataExpr + ", n), " + buf + ".data());");
            return { "Operand(" + buf + ".data())", buf + ".data()" };
        }

        if (auto* cs = dynamic_cast<const CsExpr*>(node)) {
            auto child = emitNode(cs->child());
            auto it = csMap.find(cs->opCode());
            if (it == csMap.end()) throw std::runtime_error("Unknown CS op");
            includes_.insert(it->second.includePath);
            std::string buf = newBuf();
            statements_.push_back(
                "    auto " + buf + " = pool.allocate<double>(n);\n"
                "    " + it->second.className + "{}.evaluate(ColView<double>(" +
                child.dataExpr + ", n), " + buf + ".data());");
            return { "Operand(" + buf + ".data())", buf + ".data()" };
        }

        throw std::runtime_error("Unknown expression node type");
    }

    void writeHeader(const std::string& path) {
        std::ofstream f(path);
        auto pc = pascalCase();

        f << "// " << factorName_ << ".h — AOT-compiled factor (auto-generated)\n"
          << "//\n"
          << "// Expression: " << expression_ << "\n"
          << "//\n"
          << "// Generated by: tools/generate_factor.cpp\n"
          << "// Do not edit by hand — regenerate with:\n"
          << "//   generate_factor " << factorName_ << " \"" << expression_ << "\"\n"
          << "#pragma once\n\n"
          << "#include <string>\n"
          << "#include \"quantcore/core/FactorCalculator.h\"\n"
          << "#include \"quantcore/storage/Column.h\"\n"
          << "#include \"quantcore/storage/MarketData.h\"\n\n"
          << "namespace quantcore {\n"
          << "namespace factors {\n\n"
          << "extern const std::string k" << pc << ";\n"
          << "extern const std::string k" << pc << "Desc;\n"
          << "extern const std::string k" << pc << "Expr;  // backward compat\n\n"
          << "/// Compute " << factorName_ << ": " << expression_ << "\n"
          << "Column<double> compute" << pc << "(const MarketData& md,\n"
          << "                              ExecutionEngine& engine);\n\n"
          << "void register" << pc << "(FactorCalculator& calc);\n\n"
          << "Column<double> evaluate" << pc << "(FactorCalculator& calc,\n"
          << "                                const MarketData& md);\n\n"
          << "}  // namespace factors\n"
          << "}  // namespace quantcore\n";
    }

    void writeCpp(const std::string& path, const EmitResult& result) {
        std::ofstream f(path);
        auto pc = pascalCase();

        f << "// " << factorName_ << ".cpp — AOT-compiled factor (auto-generated)\n"
          << "//\n"
          << "// Expression: " << expression_ << "\n"
          << "//\n"
          << "// Generated by: tools/generate_factor.cpp\n"
          << "// Do not edit by hand.\n\n"
          << "#include \"factors/" << factorName_ << ".h\"\n\n"
          << "#include <cstring>\n"
          << "#include <cmath>\n"
          << "#include <limits>\n"
          << "#include <cstddef>\n\n"
          << "#include \"quantcore/core/Types.h\"\n"
          << "#include \"quantcore/storage/ColView.h\"\n"
          << "#include \"quantcore/storage/Column.h\"\n"
          << "#include \"quantcore/engine/BufferPool.h\"\n"
          << "#include \"quantcore/engine/ExecutionEngine.h\"\n"
          << "#include \"quantcore/operators/RollingOperator.h\"\n"
          << "#include \"quantcore/operators/UnaryOperator.h\"\n"
          << "#include \"quantcore/operators/BinaryOperator.h\"\n";

        for (const auto& inc : includes_)
            f << "#include \"" << inc << "\"\n";

        f << "\nnamespace quantcore {\n"
          << "namespace factors {\n\n"
          << "const std::string k" << pc << " = \"" << factorName_ << "\";\n\n"
          << "const std::string k" << pc << "Desc =\n"
          << "    \"" << expression_ << "\";\n\n"
          << "const std::string k" << pc << "Expr =\n"
          << "    \"" << expression_ << "\";  // backward compat\n\n"
          << "Column<double> compute" << pc << "(const MarketData& md,\n"
          << "                              ExecutionEngine& engine) {\n"
          << "    std::size_t n = md.rowCount();\n"
          << "    auto& pool = engine.pool();\n\n";

        for (const auto& stmt : statements_)
            f << stmt << "\n";

        f << "\n    // Materialize final result\n"
          << "    Column<double> result(n);\n"
          << "    std::memcpy(result.data(), " << result.dataExpr
          << ", n * sizeof(double));\n"
          << "    return result;\n"
          << "}\n\n"
          << "void register" << pc << "(FactorCalculator& calc) {\n"
          << "    calc.registerCustomFactor(\n"
          << "        k" << pc << ",\n"
          << "        [&calc](const MarketData& md) -> Column<double> {\n"
          << "            return compute" << pc << "(md, calc.engine());\n"
          << "        },\n"
          << "        k" << pc << "Desc);\n"
          << "}\n\n"
          << "Column<double> evaluate" << pc << "(FactorCalculator& calc,\n"
          << "                                const MarketData& md) {\n"
          << "    return calc.evaluate(k" << pc << ", md);\n"
          << "}\n\n"
          << "}  // namespace factors\n"
          << "}  // namespace quantcore\n";
    }
};

// ============================================================
// main
// ============================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: generate_factor <name> <expression>\n\n"
                  << "Example:\n"
                  << "  generate_factor alpha_0001 \"close / rolling_shift(close, 20) - 1\"\n"
                  << "  generate_factor alpha_0003 \"rolling_corr(close, vwap, 20)\"\n";
        return 1;
    }

    std::string name = argv[1];
    std::string expr = argv[2];

    CodeGenerator gen(name, expr);
    gen.generate(name + ".h", name + ".cpp");

    std::cout << "Generated: " << name << ".h, " << name << ".cpp\n"
              << "Expression: " << expr << "\n";
    return 0;
}
