// FusedLoopGenerator.h — operator fusion: AST → single-loop kernels
// Phase: 四期必实现
//
// Detects chains of element-wise operators (UnaryExpr, BinaryExpr) and
// compiles them into a FusedKernel that evaluates the entire subtree in
// a single pass over the data.
//
// Fusion boundaries: RollingExpr, RedExpr, CsExpr — these require the full
// input array before computing, so they cannot participate in element-wise
// fusion.  They are evaluated via standard evaluate() as separate steps.
//
// Supported patterns:
//   1. Unary chain:   SQRT(ABS(LOG(C)))    → 1 loop, 0 temp bufs
//   2. Binary+chains: LOG(C) + LOG(V)      → 1 loop, 0 temp bufs
//   3. Binary+result: ABS(LOG(C)-LOG(V))   → 1 loop, 0 temp bufs
//
// Usage (internal to ExecutionEngine):
//   FusedLoopGenerator gen;
//   auto kernel = gen.tryCompile(exprRoot);
//   if (kernel) {
//       kernel->evaluate(md, output, n, nullMaskOut);
//   } else {
//       exprRoot.evaluate(md, output, n);
//   }
#pragma once

#include <cstring>
#include <memory>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/expression/ColumnRef.h"
#include "quantcore/expression/Scalar.h"
#include "quantcore/expression/UnaryExpr.h"
#include "quantcore/expression/BinaryExpr.h"
#include "quantcore/registry/OperatorRegistry.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"

namespace quantcore {

// ============================================================
// FusedKernel — a compiled single-loop program
// ============================================================

class FusedKernel {
public:
    enum Kind { kUnaryChain, kBinary };

    Kind kind;

    // --- Left leaf (always present) ---
    Field       lhsField;       // valid if lhsIsColumn
    double      lhsScalar;      // valid if !lhsIsColumn
    bool        lhsIsColumn;
    std::vector<UnaryOpCode> lhsChain;  // applied to leaf value

    // --- Right leaf (binary only) ---
    Field       rhsField;
    double      rhsScalar;
    bool        rhsIsColumn;
    std::vector<UnaryOpCode> rhsChain;

    // --- Binary operator (binary only) ---
    BinaryOpCode binaryOp;

    // --- Result chain (applied after binary, or on top of unary leaf) ---
    std::vector<UnaryOpCode> resultChain;

    // ============================================================
    // Evaluate the kernel in a single fused loop
    // ============================================================

    void evaluate(const MarketData& md,
                  double* output,
                  std::size_t n,
                  const uint64_t*& outNullMask) const {
        auto& reg = OperatorRegistry::instance();

        if (kind == kUnaryChain) {
            evaluateUnary(reg, md, output, n, outNullMask);
        } else {
            evaluateBinary(reg, md, output, n, outNullMask);
        }
    }

private:
    // --- Unary chain: result = resultChain(lhsChain(leaf)) ---
    void evaluateUnary(OperatorRegistry& reg,
                       const MarketData& md,
                       double* output, std::size_t n,
                       const uint64_t*& outNullMask) const {
        // Get leaf data
        const double* src;
        const uint64_t* nullMask;
        double scalarVal;

        if (lhsIsColumn) {
            const auto& col = md.column<double>(lhsField);
            src = col.data();
            nullMask = col.nullMaskData();
        } else {
            scalarVal = lhsScalar;
            src = nullptr;
            nullMask = nullptr;  // scalars never null
        }

        // Build merged chain: lhsChain + resultChain
        std::size_t totalOps = lhsChain.size() + resultChain.size();
        // Small optimization: if chain is empty, just copy
        if (totalOps == 0) {
            if (lhsIsColumn) {
                std::memcpy(output, src, n * sizeof(double));
            } else {
                std::fill_n(output, n, lhsScalar);
            }
            outNullMask = nullMask;
            return;
        }

        // Collect all scalar function pointers
        std::vector<UnaryScalarFn> fns;
        fns.reserve(totalOps);
        for (auto op : lhsChain)   fns.push_back(reg.getUnaryScalar(op));
        for (auto op : resultChain) fns.push_back(reg.getUnaryScalar(op));

        if (nullMask == nullptr) {
            // Fast path: no nulls
            for (std::size_t i = 0; i < n; ++i) {
                double x = lhsIsColumn ? src[i] : scalarVal;
                for (auto fn : fns) x = fn(x);
                output[i] = x;
            }
            outNullMask = nullptr;
        } else {
            // Slow path: skip null positions
            for (std::size_t i = 0; i < n; ++i) {
                if ((nullMask[i / 64] >> (i % 64)) & uint64_t{1}) {
                    output[i] = 0.0;
                } else {
                    double x = lhsIsColumn ? src[i] : scalarVal;
                    for (auto fn : fns) x = fn(x);
                    output[i] = x;
                }
            }
            outNullMask = nullMask;
        }
    }

    // --- Binary: result = resultChain(binaryOp(lhsChain(l), rhsChain(r))) ---
    void evaluateBinary(OperatorRegistry& reg,
                        const MarketData& md,
                        double* output, std::size_t n,
                        const uint64_t*& outNullMask) const {
        BinaryScalarFn binFn = reg.getBinaryScalar(binaryOp);

        // LHS
        const double* lhsSrc = nullptr;
        const uint64_t* lhsNull = nullptr;
        double lhsScalarVal = 0.0;

        if (lhsIsColumn) {
            const auto& col = md.column<double>(lhsField);
            lhsSrc = col.data();
            lhsNull = col.nullMaskData();
        } else {
            lhsScalarVal = lhsScalar;
        }

        // RHS
        const double* rhsSrc = nullptr;
        const uint64_t* rhsNull = nullptr;
        double rhsScalarVal = 0.0;

        if (rhsIsColumn) {
            const auto& col = md.column<double>(rhsField);
            rhsSrc = col.data();
            rhsNull = col.nullMaskData();
        } else {
            rhsScalarVal = rhsScalar;
        }

        // Combined null: null if EITHER side is null
        const uint64_t* combinedNull = nullptr;
        if (lhsNull == nullptr && rhsNull == nullptr) {
            combinedNull = nullptr;
        } else if (lhsNull == nullptr) {
            combinedNull = rhsNull;
        } else if (rhsNull == nullptr) {
            combinedNull = lhsNull;
        } else {
            // Both have nulls — we handle this inline in the loop
            combinedNull = nullptr;  // signal "check both"
        }

        // Collect scalar fns
        std::vector<UnaryScalarFn> lhsFns, rhsFns, resFns;
        for (auto op : lhsChain)   lhsFns.push_back(reg.getUnaryScalar(op));
        for (auto op : rhsChain)   rhsFns.push_back(reg.getUnaryScalar(op));
        for (auto op : resultChain) resFns.push_back(reg.getUnaryScalar(op));

        if (lhsNull == nullptr && rhsNull == nullptr) {
            // Fast path: no nulls at all
            for (std::size_t i = 0; i < n; ++i) {
                double l = lhsIsColumn ? lhsSrc[i] : lhsScalarVal;
                for (auto fn : lhsFns) l = fn(l);
                double r = rhsIsColumn ? rhsSrc[i] : rhsScalarVal;
                for (auto fn : rhsFns) r = fn(r);
                double v = binFn(l, r);
                for (auto fn : resFns) v = fn(v);
                output[i] = v;
            }
            outNullMask = nullptr;
        } else if (combinedNull == nullptr) {
            // Both sides have nulls — check each
            for (std::size_t i = 0; i < n; ++i) {
                bool lhsNull_i = lhsNull && ((lhsNull[i/64] >> (i%64)) & 1);
                bool rhsNull_i = rhsNull && ((rhsNull[i/64] >> (i%64)) & 1);
                if (lhsNull_i || rhsNull_i) {
                    output[i] = 0.0;
                } else {
                    double l = lhsIsColumn ? lhsSrc[i] : lhsScalarVal;
                    for (auto fn : lhsFns) l = fn(l);
                    double r = rhsIsColumn ? rhsSrc[i] : rhsScalarVal;
                    for (auto fn : rhsFns) r = fn(r);
                    double v = binFn(l, r);
                    for (auto fn : resFns) v = fn(v);
                    output[i] = v;
                }
            }
            outNullMask = (lhsNull && rhsNull) ? nullptr : (lhsNull ? lhsNull : rhsNull);
        } else {
            // Only one side has nulls
            for (std::size_t i = 0; i < n; ++i) {
                if ((combinedNull[i/64] >> (i%64)) & 1) {
                    output[i] = 0.0;
                } else {
                    double l = lhsIsColumn ? lhsSrc[i] : lhsScalarVal;
                    for (auto fn : lhsFns) l = fn(l);
                    double r = rhsIsColumn ? rhsSrc[i] : rhsScalarVal;
                    for (auto fn : rhsFns) r = fn(r);
                    double v = binFn(l, r);
                    for (auto fn : resFns) v = fn(v);
                    output[i] = v;
                }
            }
            outNullMask = combinedNull;
        }
    }
};

// ============================================================
// FusedLoopGenerator — AST → FusedKernel compiler
// ============================================================

class FusedLoopGenerator {
public:
    /// Try to compile an expression subtree into a FusedKernel.
    /// Returns nullptr if the subtree cannot be fused (contains
    /// RollingExpr, RedExpr, CsExpr, or unsupported patterns).
    std::unique_ptr<FusedKernel> tryCompile(const ExprNode* root);

private:
    // Recursive pattern matchers.  Return true if the subtree was
    // successfully compiled into the kernel being built.
    bool matchLeaf(const ExprNode* node,
                   Field& field, double& scalar, bool& isColumn);
    bool matchChain(const ExprNode* node,
                    std::vector<UnaryOpCode>& chain,
                    Field& leafField, double& leafScalar, bool& leafIsColumn);
};

// ============================================================
// Implementation
// ============================================================

inline bool FusedLoopGenerator::matchLeaf(const ExprNode* node,
        Field& field, double& scalar, bool& isColumn) {
    if (auto* cr = dynamic_cast<const ColumnRef*>(node)) {
        field    = cr->field();
        isColumn = true;
        return true;
    }
    if (auto* s = dynamic_cast<const Scalar*>(node)) {
        scalar   = s->value();
        isColumn = false;
        return true;
    }
    return false;
}

// ============================================================
// Fusibility check — some UnaryOpCodes are NOT element-wise
// and should not participate in fused loops.
// ============================================================

inline bool isFusibleUnary(UnaryOpCode op) noexcept {
    switch (op) {
        // RANK operators require the full column for sorting — they
        // cannot be evaluated per-element in a fused loop.
        case UnaryOpCode::RANK:
        case UnaryOpCode::RANK_PCT:
        case UnaryOpCode::RANK_NORMALIZED:
            return false;
        default:
            return true;
    }
}

inline bool FusedLoopGenerator::matchChain(const ExprNode* node,
        std::vector<UnaryOpCode>& chain,
        Field& leafField, double& leafScalar, bool& leafIsColumn) {
    if (auto* u = dynamic_cast<const UnaryExpr*>(node)) {
        // Reject non-fusible operators — they need the full column
        // (e.g., RANK requires sorting) and cannot be part of a
        // fused element-wise loop.
        if (!isFusibleUnary(u->opCode())) {
            return false;
        }
        // Recurse: collect ops from bottom to top
        if (matchChain(u->child(), chain, leafField, leafScalar, leafIsColumn)) {
            chain.push_back(u->opCode());
            return true;
        }
        return false;
    }
    // Base case: leaf node
    return matchLeaf(node, leafField, leafScalar, leafIsColumn);
}

inline std::unique_ptr<FusedKernel> FusedLoopGenerator::tryCompile(
        const ExprNode* root) {
    if (!root) return nullptr;

    // Pattern 1 & 3: Unary chain (possibly with binary at base)
    if (auto* u = dynamic_cast<const UnaryExpr*>(root)) {
        // Check if the child is a BinaryExpr (Pattern 3: result chain on binary)
        if (auto* bin = dynamic_cast<const BinaryExpr*>(u->child())) {
            auto kernel = std::make_unique<FusedKernel>();
            kernel->kind = FusedKernel::kBinary;

            // Match lhs chain
            if (!matchChain(bin->lhs(), kernel->lhsChain,
                            kernel->lhsField, kernel->lhsScalar,
                            kernel->lhsIsColumn)) {
                return nullptr;
            }
            // Match rhs chain
            if (!matchChain(bin->rhs(), kernel->rhsChain,
                            kernel->rhsField, kernel->rhsScalar,
                            kernel->rhsIsColumn)) {
                return nullptr;
            }

            kernel->binaryOp = bin->opCode();

            // Result chain: the UnaryExpr chain above the binary
            // Collect from the UnaryExpr chain wrapping the binary
            const ExprNode* cur = root;
            while (cur != bin) {
                if (auto* u2 = dynamic_cast<const UnaryExpr*>(cur)) {
                    kernel->resultChain.push_back(u2->opCode());
                    cur = u2->child();
                } else {
                    break;
                }
            }
            // resultChain was built top-down; reverse to get bottom-up order
            // Actually, we want apply order: binary result → resultChain[0] → [1] → ...
            // The chain was built from outer Unary to inner; the actual eval order
            // is inner-to-outer after binary.
            // So we need to reverse.
            std::reverse(kernel->resultChain.begin(), kernel->resultChain.end());

            return kernel;
        }

        // Pattern 1: Pure unary chain on leaf
        auto kernel = std::make_unique<FusedKernel>();
        kernel->kind = FusedKernel::kUnaryChain;

        // resultChain will hold the top-level unary ops;
        // lhsChain will hold the chain from leaf upward.
        // Actually, for kUnaryChain, we apply lhsChain then resultChain.
        // Let's just put everything in lhsChain and leave resultChain empty.
        if (!matchChain(root, kernel->lhsChain,
                        kernel->lhsField, kernel->lhsScalar,
                        kernel->lhsIsColumn)) {
            return nullptr;
        }

        return kernel;
    }

    // Pattern 2: Binary at root with chains on leaves
    if (auto* bin = dynamic_cast<const BinaryExpr*>(root)) {
        auto kernel = std::make_unique<FusedKernel>();
        kernel->kind = FusedKernel::kBinary;

        if (!matchChain(bin->lhs(), kernel->lhsChain,
                        kernel->lhsField, kernel->lhsScalar,
                        kernel->lhsIsColumn)) {
            return nullptr;
        }
        if (!matchChain(bin->rhs(), kernel->rhsChain,
                        kernel->rhsField, kernel->rhsScalar,
                        kernel->rhsIsColumn)) {
            return nullptr;
        }

        kernel->binaryOp = bin->opCode();
        // No result chain
        return kernel;
    }

    // Pattern: Single leaf (just copy)
    auto leafKernel = std::make_unique<FusedKernel>();
    leafKernel->kind = FusedKernel::kUnaryChain;
    if (matchLeaf(root, leafKernel->lhsField, leafKernel->lhsScalar,
                  leafKernel->lhsIsColumn)) {
        return leafKernel;
    }

    return nullptr;
}

}  // namespace quantcore
