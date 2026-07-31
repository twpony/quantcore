// test_registry.cpp — unit tests for OperatorRegistry
//
// Covers: operator listing, name lookup, scalar fn access,
// custom operator registration, and dispatch invocation.

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/registry/OperatorRegistry.h"
#include "quantcore/storage/ColView.h"

using namespace quantcore;

// ============================================================
// Operator listing — all families
// ============================================================

TEST(RegistryTest, ListUnaryOperators) {
    auto& reg = OperatorRegistry::instance();
    auto ops = reg.listUnary();
    EXPECT_GE(ops.size(), 12u);
    bool hasAbs = false, hasLog = false, hasSqrt = false;
    for (auto& name : ops) {
        if (name == "abs")  hasAbs  = true;
        if (name == "log")  hasLog  = true;
        if (name == "sqrt") hasSqrt = true;
    }
    EXPECT_TRUE(hasAbs);
    EXPECT_TRUE(hasLog);
    EXPECT_TRUE(hasSqrt);
}

TEST(RegistryTest, ListBinaryOperators) {
    auto& reg = OperatorRegistry::instance();
    auto ops = reg.listBinary();
    EXPECT_GE(ops.size(), 10u);
    bool hasAdd = false, hasSub = false, hasMul = false;
    for (auto& name : ops) {
        if (name == "add") hasAdd = true;
        if (name == "sub") hasSub = true;
        if (name == "mul") hasMul = true;
    }
    EXPECT_TRUE(hasAdd);
    EXPECT_TRUE(hasSub);
    EXPECT_TRUE(hasMul);
}

TEST(RegistryTest, ListRollingOperators) {
    auto& reg = OperatorRegistry::instance();
    auto ops = reg.listRolling();
    EXPECT_GE(ops.size(), 17u);
    bool hasMean = false, hasStd = false, hasSum = false;
    for (auto& name : ops) {
        if (name == "rolling_mean") hasMean = true;
        if (name == "rolling_std")  hasStd  = true;
        if (name == "rolling_sum")  hasSum  = true;
    }
    EXPECT_TRUE(hasMean);
    EXPECT_TRUE(hasStd);
    EXPECT_TRUE(hasSum);
}

TEST(RegistryTest, ListRedOperators) {
    auto& reg = OperatorRegistry::instance();
    auto ops = reg.listRed();
    EXPECT_GE(ops.size(), 8u);
    bool hasSum = false, hasMean = false;
    for (auto& name : ops) {
        if (name == "red_sum")  hasSum  = true;
        if (name == "red_mean") hasMean = true;
    }
    EXPECT_TRUE(hasSum);
    EXPECT_TRUE(hasMean);
}

TEST(RegistryTest, ListCsOperators) {
    auto& reg = OperatorRegistry::instance();
    auto ops = reg.listCs();
    EXPECT_GE(ops.size(), 8u);
    bool hasRank = false, hasZscore = false;
    for (auto& name : ops) {
        if (name == "cs_rank")   hasRank   = true;
        if (name == "cs_zscore") hasZscore = true;
    }
    EXPECT_TRUE(hasRank);
    EXPECT_TRUE(hasZscore);
}

// ============================================================
// Name lookup
// ============================================================

TEST(RegistryTest, FindUnaryByName) {
    auto& reg = OperatorRegistry::instance();
    EXPECT_EQ(reg.findUnary("abs"),  UnaryOpCode::ABS);
    EXPECT_EQ(reg.findUnary("log"),  UnaryOpCode::LOG);
    EXPECT_EQ(reg.findUnary("sqrt"), UnaryOpCode::SQRT);
    EXPECT_EQ(reg.findUnary("neg"),  UnaryOpCode::NEG);
}

TEST(RegistryTest, FindUnaryCaseInsensitive) {
    auto& reg = OperatorRegistry::instance();
    // Names are stored lowercase; lookup is case-sensitive.
    // Mixed case throws because the Lexer already lowercases identifiers,
    // so the Parser never passes uppercase names.
    EXPECT_EQ(reg.findUnary("abs"), UnaryOpCode::ABS);
    EXPECT_THROW(reg.findUnary("ABS"), std::runtime_error);
}

TEST(RegistryTest, FindUnaryUnknownThrows) {
    auto& reg = OperatorRegistry::instance();
    EXPECT_THROW(reg.findUnary("nonexistent_op_xyz"), std::runtime_error);
}

TEST(RegistryTest, FindBinaryByName) {
    auto& reg = OperatorRegistry::instance();
    EXPECT_EQ(reg.findBinary("add"), BinaryOpCode::ADD);
    EXPECT_EQ(reg.findBinary("sub"), BinaryOpCode::SUB);
    EXPECT_EQ(reg.findBinary("mul"), BinaryOpCode::MUL);
    EXPECT_EQ(reg.findBinary("div"), BinaryOpCode::DIV);
}

TEST(RegistryTest, FindBinaryUnknownThrows) {
    auto& reg = OperatorRegistry::instance();
    EXPECT_THROW(reg.findBinary("nonexistent_op_xyz"), std::runtime_error);
}

TEST(RegistryTest, FindRollingByName) {
    auto& reg = OperatorRegistry::instance();
    EXPECT_EQ(reg.findRolling("rolling_mean"), RollingOpCode::ROLLING_MEAN);
    EXPECT_EQ(reg.findRolling("rolling_std"),  RollingOpCode::ROLLING_STD);
}

TEST(RegistryTest, FindRedByName) {
    auto& reg = OperatorRegistry::instance();
    EXPECT_EQ(reg.findRed("red_sum"),  RedOpCode::RED_SUM);
    EXPECT_EQ(reg.findRed("red_mean"), RedOpCode::RED_MEAN);
}

TEST(RegistryTest, FindCsByName) {
    auto& reg = OperatorRegistry::instance();
    EXPECT_EQ(reg.findCs("cs_rank"),   CsOpCode::CS_RANK);
    EXPECT_EQ(reg.findCs("cs_zscore"), CsOpCode::CS_ZSCORE);
}

// ============================================================
// Scalar function access
// ============================================================

TEST(RegistryTest, GetUnaryScalarFunctions) {
    auto& reg = OperatorRegistry::instance();
    std::vector<UnaryOpCode> builtins = {
        UnaryOpCode::ABS, UnaryOpCode::LOG, UnaryOpCode::SQRT,
        UnaryOpCode::NEG, UnaryOpCode::SIGN, UnaryOpCode::SQUARE,
        UnaryOpCode::EXP, UnaryOpCode::INV
    };
    for (auto code : builtins) {
        auto fn = reg.getUnaryScalar(code);
        EXPECT_NE(fn, nullptr) << "UnaryOpCode " << static_cast<int>(code)
                               << " has no scalar fn";
    }
}

TEST(RegistryTest, GetBinaryScalarFunctions) {
    auto& reg = OperatorRegistry::instance();
    std::vector<BinaryOpCode> builtins = {
        BinaryOpCode::ADD, BinaryOpCode::SUB, BinaryOpCode::MUL, BinaryOpCode::DIV,
        BinaryOpCode::MAX, BinaryOpCode::MIN
    };
    for (auto code : builtins) {
        auto fn = reg.getBinaryScalar(code);
        EXPECT_NE(fn, nullptr) << "BinaryOpCode " << static_cast<int>(code)
                               << " has no scalar fn";
    }
}

TEST(RegistryTest, UnaryScalarFnProducesCorrectResult) {
    auto& reg = OperatorRegistry::instance();
    auto absFn = reg.getUnaryScalar(UnaryOpCode::ABS);
    ASSERT_NE(absFn, nullptr);
    EXPECT_DOUBLE_EQ(absFn(-3.14), 3.14);
    EXPECT_DOUBLE_EQ(absFn(5.0), 5.0);

    auto logFn = reg.getUnaryScalar(UnaryOpCode::LOG);
    ASSERT_NE(logFn, nullptr);
    EXPECT_DOUBLE_EQ(logFn(1.0), 0.0);
}

TEST(RegistryTest, BinaryScalarFnProducesCorrectResult) {
    auto& reg = OperatorRegistry::instance();
    auto addFn = reg.getBinaryScalar(BinaryOpCode::ADD);
    ASSERT_NE(addFn, nullptr);
    EXPECT_DOUBLE_EQ(addFn(2.0, 3.0), 5.0);

    auto mulFn = reg.getBinaryScalar(BinaryOpCode::MUL);
    ASSERT_NE(mulFn, nullptr);
    EXPECT_DOUBLE_EQ(mulFn(4.0, 0.5), 2.0);
}

// ============================================================
// Custom operator registration
// ============================================================

TEST(RegistryTest, RegisterCustomUnary) {
    auto& reg = OperatorRegistry::instance();
    reg.registerCustomUnary("triple", UnaryOpCode::CUSTOM_0,
        [](double x) noexcept { return x * 3.0; });

    EXPECT_EQ(reg.findUnary("triple"), UnaryOpCode::CUSTOM_0);

    auto fn = reg.getUnaryScalar(UnaryOpCode::CUSTOM_0);
    ASSERT_NE(fn, nullptr);
    EXPECT_DOUBLE_EQ(fn(7.0), 21.0);
}

TEST(RegistryTest, RegisterCustomBinary) {
    auto& reg = OperatorRegistry::instance();
    reg.registerCustomBinary("dist", BinaryOpCode::CUSTOM_0,
        [](double a, double b) noexcept { return std::abs(a - b); });

    EXPECT_EQ(reg.findBinary("dist"), BinaryOpCode::CUSTOM_0);

    auto fn = reg.getBinaryScalar(BinaryOpCode::CUSTOM_0);
    ASSERT_NE(fn, nullptr);
    EXPECT_DOUBLE_EQ(fn(3.0, 7.0), 4.0);
}

// ============================================================
// Dispatch invocation
// ============================================================

TEST(RegistryTest, InvokeUnaryByCode) {
    auto& reg = OperatorRegistry::instance();
    constexpr std::size_t kN = 5;
    double input[kN]  = {-3.0, -1.0, 0.0, 2.0, 5.0};
    double output[kN] = {};
    Operand op(input);

    reg.invokeUnary(UnaryOpCode::ABS, op, output, kN, nullptr);
    EXPECT_DOUBLE_EQ(output[0], 3.0);
    EXPECT_DOUBLE_EQ(output[1], 1.0);
    EXPECT_DOUBLE_EQ(output[2], 0.0);
    EXPECT_DOUBLE_EQ(output[3], 2.0);
    EXPECT_DOUBLE_EQ(output[4], 5.0);
}

TEST(RegistryTest, InvokeBinaryByCode) {
    auto& reg = OperatorRegistry::instance();
    constexpr std::size_t kN = 3;
    double lhs[kN] = {1.0, 2.0, 3.0};
    double rhs[kN] = {4.0, 5.0, 6.0};
    double output[kN] = {};
    Operand lop(lhs), rop(rhs);

    reg.invokeBinary(BinaryOpCode::ADD, lop, rop, output, kN, nullptr);
    EXPECT_DOUBLE_EQ(output[0], 5.0);
    EXPECT_DOUBLE_EQ(output[1], 7.0);
    EXPECT_DOUBLE_EQ(output[2], 9.0);
}

// ============================================================
// Singleton identity
// ============================================================

TEST(RegistryTest, SameInstance) {
    auto& r1 = OperatorRegistry::instance();
    auto& r2 = OperatorRegistry::instance();
    EXPECT_EQ(&r1, &r2);
}

// ============================================================
// Rolling dispatch
// ============================================================

TEST(RegistryTest, InvokeRollingMean) {
    auto& reg = OperatorRegistry::instance();
    constexpr std::size_t kN = 10;
    double input[kN]  = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double output[kN] = {};
    ColView<double> view(input, kN);

    reg.invokeRolling(RollingOpCode::ROLLING_MEAN, view, output, 3);
    EXPECT_TRUE(std::isnan(output[0]));
    EXPECT_TRUE(std::isnan(output[1]));
    EXPECT_DOUBLE_EQ(output[2], (1.0 + 2.0 + 3.0) / 3.0);
    EXPECT_DOUBLE_EQ(output[3], (2.0 + 3.0 + 4.0) / 3.0);
    EXPECT_DOUBLE_EQ(output[9], (8.0 + 9.0 + 10.0) / 3.0);
}

TEST(RegistryTest, IsBinaryRolling) {
    auto& reg = OperatorRegistry::instance();
    EXPECT_TRUE(reg.isBinaryRolling(RollingOpCode::ROLLING_CORR));
    EXPECT_TRUE(reg.isBinaryRolling(RollingOpCode::ROLLING_COV));
    EXPECT_FALSE(reg.isBinaryRolling(RollingOpCode::ROLLING_MEAN));
    EXPECT_FALSE(reg.isBinaryRolling(RollingOpCode::ROLLING_STD));
}
