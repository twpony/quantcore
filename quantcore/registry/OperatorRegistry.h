// OperatorRegistry.h — operator name-to-enum bidirectional registry
// Phase: 一期基础实现 + 远期扩展预留
//
// OperatorRegistry maintains the mapping between human-readable operator
// names and their enum codes, AND provides name-based dispatch to the
// actual evaluate functions.  This serves:
//   1. Logging / diagnostics — print operator names in metrics
//   2. Name-based invocation — invokeUnary("abs", ...) calls AbsOp::evaluate
//   3. String-formula parser — "ABS(LOG(CLOSE))" → expression tree
//   4. Genetic programming — enumerate available operators
//
// All five operator families are registered:
//   UnaryOperator       (一期实现, dispatch supported)
//   BinaryOperator      (一期实现, dispatch supported)
//   RollingOperator     (已实现)
//   RedOperator          (red/ — 聚合算子, 合并 cross_section + reduction)
//   CsOperator           (cs/ — 截面变换算子)
#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "quantcore/core/Types.h"

namespace quantcore {

// ============================================================
// Type-erased evaluate function pointers
// ============================================================

using UnaryEvalFn  = void (*)(const Operand&, double*, std::size_t, const uint64_t*);
using BinaryEvalFn = void (*)(const Operand&, const Operand&, double*, std::size_t, const uint64_t*);

class OperatorRegistry {
public:
    static OperatorRegistry& instance();

    // ============================================================
    // Unary operator registration
    // ============================================================

    template <typename OpType>
    void registerUnary(const std::string& name, UnaryOpCode code) {
        unaryRegistry_[name] = code;
        // Capture the stateless operator's evaluate method
        auto idx = static_cast<std::size_t>(code);
        if (idx >= unaryDispatch_.size()) {
            unaryDispatch_.resize(idx + 1);
        }
        unaryDispatch_[idx] = [](const Operand& input, double* output,
                                  std::size_t n, const uint64_t* mask) {
            OpType op;
            op.evaluate(input, output, n, mask);
        };
    }

    UnaryOpCode findUnary(const std::string& name) const;
    std::vector<std::string> listUnary() const;

    /// Invoke a unary operator by name.
    /// @throws std::runtime_error if name is unknown or operator not registered.
    void invokeUnary(const std::string& name,
                     const Operand& input,
                     double*        output,
                     std::size_t    n,
                     const uint64_t* nullMask) const;

    /// Invoke a unary operator by enum code.
    void invokeUnary(UnaryOpCode code,
                     const Operand& input,
                     double*        output,
                     std::size_t    n,
                     const uint64_t* nullMask) const;

    // ============================================================
    // Binary operator registration
    // ============================================================

    template <typename OpType>
    void registerBinary(const std::string& name, BinaryOpCode code) {
        binaryRegistry_[name] = code;
        auto idx = static_cast<std::size_t>(code);
        if (idx >= binaryDispatch_.size()) {
            binaryDispatch_.resize(idx + 1);
        }
        binaryDispatch_[idx] = [](const Operand& lhs, const Operand& rhs,
                                   double* output, std::size_t n,
                                   const uint64_t* mask) {
            OpType op;
            op.evaluate(lhs, rhs, output, n, mask);
        };
    }

    BinaryOpCode findBinary(const std::string& name) const;
    std::vector<std::string> listBinary() const;

    /// Invoke a binary operator by name.
    /// @throws std::runtime_error if name is unknown or operator not registered.
    void invokeBinary(const std::string& name,
                      const Operand& lhs,
                      const Operand& rhs,
                      double*        output,
                      std::size_t    n,
                      const uint64_t* nullMask) const;

    /// Invoke a binary operator by enum code.
    void invokeBinary(BinaryOpCode code,
                      const Operand& lhs,
                      const Operand& rhs,
                      double*        output,
                      std::size_t    n,
                      const uint64_t* nullMask) const;

    // ============================================================
    // Rolling operator registration (远期)
    // ============================================================

    template <typename OpType>
    void registerRolling(const std::string& name, RollingOpCode code) {
        rollingRegistry_[name] = code;
    }

    RollingOpCode findRolling(const std::string& name) const;
    std::vector<std::string> listRolling() const;

    // ============================================================
    // CS operator registration (截面算子, 合并 cross_section + reduction)
    // ============================================================

    template <typename OpType>
    void registerRed(const std::string& name, RedOpCode code) {
        redRegistry_[name] = code;
    }

    RedOpCode findRed(const std::string& name) const;
    std::vector<std::string> listRed() const;

    // ============================================================
    // CS operator registration (截面变换算子)
    // ============================================================

    template <typename OpType>
    void registerCs(const std::string& name, CsOpCode code) {
        csRegistry_[name] = code;
    }

    CsOpCode findCs(const std::string& name) const;
    std::vector<std::string> listCs() const;

private:
    OperatorRegistry() = default;

    std::unordered_map<std::string, UnaryOpCode>   unaryRegistry_;
    std::unordered_map<std::string, BinaryOpCode>  binaryRegistry_;
    std::unordered_map<std::string, RollingOpCode> rollingRegistry_;
    std::unordered_map<std::string, RedOpCode>      redRegistry_;
    std::unordered_map<std::string, CsOpCode>       csRegistry_;

    // Dispatch tables: enum code → evaluate function
    std::vector<UnaryEvalFn>   unaryDispatch_;
    std::vector<BinaryEvalFn>  binaryDispatch_;
};

}  // namespace quantcore
