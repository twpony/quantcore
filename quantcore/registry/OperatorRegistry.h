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
#include "quantcore/storage/ColView.h"

namespace quantcore {

// ============================================================
// Type-erased evaluate function pointers
// ============================================================

using UnaryEvalFn   = void (*)(const Operand&, double*, std::size_t, const uint64_t*);
using BinaryEvalFn  = void (*)(const Operand&, const Operand&, double*, std::size_t, const uint64_t*);
using RollingEvalFn = void (*)(ColView<double>, double*, std::size_t, const std::vector<double>&);

/// Binary rolling evaluation function: (inputX, inputY, output, window, extraParams)
using BinaryRollingEvalFn = void (*)(ColView<double>, ColView<double>, double*,
                                     std::size_t, const std::vector<double>&);

/// Scalar function pointer types for fused-loop evaluation.
/// Indexed by UnaryOpCode / BinaryOpCode; called per-element in a fused loop.
using UnaryScalarFn  = double (*)(double);
using BinaryScalarFn = double (*)(double, double);

/// Red evaluation function.  The third parameter carries per-call extra
/// arguments for parameterized operators (e.g., RED_QUANTILE(q)).
/// Stateless operators ignore it.
using RedEvalFn     = void (*)(ColView<double>, double*, const std::vector<double>&);

/// CS evaluation function.  The third parameter carries per-call extra
/// arguments for parameterized operators (e.g., CS_WINSORIZE(lowerPct, upperPct)).
/// Stateless operators ignore it.
using CsEvalFn      = void (*)(ColView<double>, double*, const std::vector<double>&);

class OperatorRegistry {
public:
    static OperatorRegistry& instance();

    // ============================================================
    // Unary operator registration
    // ============================================================

    template <typename OpType>
    void registerUnary(const std::string& name, UnaryOpCode code) {
        unaryRegistry_[name] = code;
        auto idx = static_cast<std::size_t>(code);
        if (idx >= unaryDispatch_.size()) {
            unaryDispatch_.resize(idx + 1);
        }
        unaryDispatch_[idx] = [](const Operand& input, double* output,
                                  std::size_t n, const uint64_t* mask) {
            OpType op;
            op.evaluate(input, output, n, mask);
        };
        // Register scalar fn for fused loops
        if (idx >= unaryScalar_.size()) {
            unaryScalar_.resize(idx + 1);
        }
        unaryScalar_[idx] = OpType::evaluateScalar;
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
        // Register scalar fn for fused loops
        if (idx >= binaryScalar_.size()) {
            binaryScalar_.resize(idx + 1);
        }
        binaryScalar_[idx] = OpType::evaluateScalar;
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

    // ============================================================
    // Rolling operator dispatch registration
    // ============================================================

    template <typename OpType>
    void registerRollingDispatch(RollingOpCode code) {
        auto idx = static_cast<std::size_t>(code);
        if (idx >= rollingDispatch_.size()) {
            rollingDispatch_.resize(idx + 1);
        }
        rollingDispatch_[idx] = [](ColView<double> input,
                                   double* output,
                                   std::size_t window,
                                   const std::vector<double>& /*params*/) {
            OpType op(window);
            op.evaluate(input, output);
        };
    }

    /// Register a rolling operator with a raw function pointer.
    /// Used for RollingQuantileOp which requires an extra `q` parameter.
    void registerRollingDispatchRaw(RollingOpCode code, RollingEvalFn fn) {
        auto idx = static_cast<std::size_t>(code);
        if (idx >= rollingDispatch_.size()) {
            rollingDispatch_.resize(idx + 1);
        }
        rollingDispatch_[idx] = fn;
    }

    /// Invoke a rolling operator by enum code.
    /// @param extraParams  per-call parameters (e.g., q for RollingQuantileOp).
    void invokeRolling(RollingOpCode code,
                       ColView<double> input,
                       double* output,
                       std::size_t window,
                       const std::vector<double>& extraParams = {}) const;

    // ============================================================
    // Binary rolling operator dispatch registration (rolling_corr, rolling_cov, ...)
    // ============================================================

    /// Register a binary rolling operator for enum-based dispatch.
    /// The operator is constructed with `window` from the dispatch lambda.
    template <typename OpType>
    void registerBinaryRollingDispatch(RollingOpCode code) {
        auto idx = static_cast<std::size_t>(code);
        if (idx >= binaryRollingDispatch_.size()) {
            binaryRollingDispatch_.resize(idx + 1);
        }
        binaryRollingDispatch_[idx] = [](ColView<double> x, ColView<double> y,
                                          double* output, std::size_t window,
                                          const std::vector<double>& /*params*/) {
            OpType op(window);
            op.evaluate(x, y, output);
        };
    }

    /// Register a binary rolling operator with a raw function pointer.
    void registerBinaryRollingDispatchRaw(RollingOpCode code, BinaryRollingEvalFn fn) {
        auto idx = static_cast<std::size_t>(code);
        if (idx >= binaryRollingDispatch_.size()) {
            binaryRollingDispatch_.resize(idx + 1);
        }
        binaryRollingDispatch_[idx] = fn;
    }

    /// Check whether a RollingOpCode is a binary rolling operator.
    bool isBinaryRolling(RollingOpCode code) const noexcept {
        auto idx = static_cast<std::size_t>(code);
        return idx < binaryRollingDispatch_.size()
            && binaryRollingDispatch_[idx] != nullptr;
    }

    /// Invoke a binary rolling operator by enum code.
    /// @param extraParams  per-call parameters (e.g., ddof for rolling_cov).
    void invokeBinaryRolling(RollingOpCode code,
                             ColView<double> x,
                             ColView<double> y,
                             double* output,
                             std::size_t window,
                             const std::vector<double>& extraParams = {}) const;

    // ============================================================
    // Red operator dispatch registration
    // ============================================================

    /// Register a stateless Red operator for enum-based dispatch.
    /// The operator is default-constructed in the dispatch lambda.
    template <typename OpType>
    void registerRedDispatch(RedOpCode code) {
        auto idx = static_cast<std::size_t>(code);
        if (idx >= redDispatch_.size()) {
            redDispatch_.resize(idx + 1);
        }
        redDispatch_[idx] = [](ColView<double> input, double* output,
                               const std::vector<double>& /*params*/) {
            OpType op;
            op.evaluate(input, output);
        };
    }

    /// Register a Red operator with a raw function pointer.
    /// Used for parameterized operators whose dispatch lambda needs
    /// custom parameter extraction from the extraParams vector.
    void registerRedDispatchRaw(RedOpCode code, RedEvalFn fn) {
        auto idx = static_cast<std::size_t>(code);
        if (idx >= redDispatch_.size()) {
            redDispatch_.resize(idx + 1);
        }
        redDispatch_[idx] = fn;
    }

    /// Invoke a Red operator by enum code.
    /// @param extraParams  per-call parameters for parameterized operators
    ///                     (empty vector for stateless operators).
    void invokeRed(RedOpCode code,
                   ColView<double> values,
                   double* output,
                   const std::vector<double>& extraParams = {}) const;

    // ============================================================
    // CS operator dispatch registration
    // ============================================================

    /// Register a stateless CS operator for enum-based dispatch.
    /// The operator is default-constructed in the dispatch lambda.
    /// Dispatches to evaluateSimd for proper batch computation (O(N log N)
    /// for rank, O(N) for zscore/normalize) instead of the O(N²) per-element
    /// evaluateScalar loop in the CRTP base evaluate().
    template <typename OpType>
    void registerCsDispatch(CsOpCode code) {
        auto idx = static_cast<std::size_t>(code);
        if (idx >= csDispatch_.size()) {
            csDispatch_.resize(idx + 1);
        }
        csDispatch_[idx] = [](ColView<double> input, double* output,
                              const std::vector<double>& /*params*/) {
            OpType op;
            op.template evaluateSimd<SimdLevel::SCALAR>(input, output);
        };
    }

    /// Register a CS operator with a raw function pointer.
    /// Used for parameterized operators whose dispatch lambda needs
    /// custom parameter extraction from the extraParams vector.
    void registerCsDispatchRaw(CsOpCode code, CsEvalFn fn) {
        auto idx = static_cast<std::size_t>(code);
        if (idx >= csDispatch_.size()) {
            csDispatch_.resize(idx + 1);
        }
        csDispatch_[idx] = fn;
    }

    /// Invoke a CS operator by enum code.
    /// @param extraParams  per-call parameters for parameterized operators
    ///                     (empty vector for stateless operators).
    void invokeCs(CsOpCode code,
                  ColView<double> values,
                  double* output,
                  const std::vector<double>& extraParams = {}) const;

    // ============================================================
    // Scalar function access (for FusedLoopGenerator)
    // ============================================================

    /// Retrieve the evaluateScalar function pointer for a unary op.
    /// Returns nullptr if not registered.
    UnaryScalarFn getUnaryScalar(UnaryOpCode code) const noexcept {
        auto idx = static_cast<std::size_t>(code);
        return (idx < unaryScalar_.size()) ? unaryScalar_[idx] : nullptr;
    }

    /// Retrieve the evaluateScalar function pointer for a binary op.
    /// Returns nullptr if not registered.
    BinaryScalarFn getBinaryScalar(BinaryOpCode code) const noexcept {
        auto idx = static_cast<std::size_t>(code);
        return (idx < binaryScalar_.size()) ? binaryScalar_[idx] : nullptr;
    }

    // ============================================================
    // Custom operator registration (user-extensible)
    // ============================================================

    /// Register a custom unary operator for use in string expressions.
    /// @param name     Name for formula strings (case-insensitive).
    /// @param code     Must be UnaryOpCode::CUSTOM_0 .. CUSTOM_7.
    /// @param scalarFn The evaluateScalar function (double → double).
    void registerCustomUnary(const std::string& name,
                             UnaryOpCode code,
                             UnaryScalarFn scalarFn);

    /// Register a custom binary operator for use in string expressions.
    /// @param name     Name for formula strings.
    /// @param code     Must be BinaryOpCode::CUSTOM_0 .. CUSTOM_3.
    /// @param scalarFn The evaluateScalar function (double, double → double).
    void registerCustomBinary(const std::string& name,
                              BinaryOpCode code,
                              BinaryScalarFn scalarFn);

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
    std::vector<RollingEvalFn> rollingDispatch_;
    std::vector<BinaryRollingEvalFn> binaryRollingDispatch_;
    std::vector<RedEvalFn>     redDispatch_;
    std::vector<CsEvalFn>      csDispatch_;

    // Scalar fn tables: enum code → evaluateScalar (for fused loops)
    std::vector<UnaryScalarFn>  unaryScalar_;
    std::vector<BinaryScalarFn> binaryScalar_;

    // Custom operator dispatch (capturing lambdas → std::function needed)
    std::vector<std::function<void(const Operand&, double*, std::size_t, const uint64_t*)>> customUnaryDispatch_;
    std::vector<std::function<void(const Operand&, const Operand&, double*, std::size_t, const uint64_t*)>> customBinaryDispatch_;
};

}  // namespace quantcore
