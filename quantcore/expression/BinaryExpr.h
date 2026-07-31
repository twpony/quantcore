// BinaryExpr.h — expression node for binary operators
// Phase: 一期必实现
//
// BinaryExpr applies a binary operator (+, -, *, /, MAX, MIN, GT, ...)
// to its left and right child expressions.  Dispatch is done at runtime
// via OperatorRegistry::invokeBinary.
//
// Unlike UnaryExpr, BinaryExpr cannot evaluate fully in-place — it
// needs at least one temporary buffer to hold one child's result while
// the other child writes to the output buffer.
//
// Phase 1 uses heap allocation (std::make_unique<double[]>) for the
// temporary buffer.  Phase 2 will replace this with BufferPool-backed
// allocation; the interface remains unchanged.
//
// Null masking:
//   Binary operators require a combined null mask — an element is null
//   if EITHER lhs[i] OR rhs[i] is null.  The combined mask is built by
//   OR-ing the masks from the two children.
//
//   If both children return nullptr (no nulls), the combined mask is
//   nullptr and the operator's fast path (no null-checking) is taken.
//
// Thread safety: evaluate() is NOT thread-safe.  Use clone() to create
// independent copies for concurrent evaluation.
#pragma once

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/engine/BufferPool.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/registry/OperatorRegistry.h"

namespace quantcore {

class BinaryExpr : public ExprNode {
public:
    BinaryExpr(BinaryOpCode op,
               std::unique_ptr<ExprNode> lhs,
               std::unique_ptr<ExprNode> rhs)
        : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

    // ============================================================
    // Accessors
    // ============================================================

    BinaryOpCode opCode() const noexcept { return op_; }
    const ExprNode* lhs() const noexcept { return lhs_.get(); }
    const ExprNode* rhs() const noexcept { return rhs_.get(); }

    // ============================================================
    // ExprNode interface
    // ============================================================

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<BinaryExpr>(op_, lhs_->clone(), rhs_->clone());
    }

    const uint64_t* evaluate(const MarketData& md,
                             double* output,
                             std::size_t n) const override {
        // 1. Evaluate RHS into a temporary buffer
        rhsBuf_.resize(n);
        const uint64_t* rhsNull = rhs_->evaluate(md, rhsBuf_.data(), n);

        // 2. Evaluate LHS into the output buffer
        const uint64_t* lhsNull = lhs_->evaluate(md, output, n);

        // 3. Build combined null mask (null if lhs[i] OR rhs[i] is null)
        const uint64_t* combinedNull = nullptr;

        if (lhsNull == nullptr && rhsNull == nullptr) {
            // Fast path: neither child has nulls
            combinedNull = nullptr;
        } else if (lhsNull == nullptr) {
            // Only rhs has nulls → use rhs mask directly
            combinedNull = rhsNull;
        } else if (rhsNull == nullptr) {
            // Only lhs has nulls → use lhs mask directly
            combinedNull = lhsNull;
        } else {
            // Both have nulls → build OR of the two masks
            std::size_t words = (n + 63) / 64;
            combinedMask_.resize(words);
            for (std::size_t w = 0; w < words; ++w) {
                combinedMask_[w] = lhsNull[w] | rhsNull[w];
            }
            combinedNull = combinedMask_.data();
        }

        // 4. Apply binary operator via OperatorRegistry dispatch
        auto& reg = OperatorRegistry::instance();
        Operand lhsOp(output);           // output already holds LHS result
        Operand rhsOp(rhsBuf_.data());   // RHS result in temp buffer
        reg.invokeBinary(op_, lhsOp, rhsOp, output, n, combinedNull);

        return combinedNull;
    }

    // Pool-aware evaluate: allocates temp buffers from BufferPool for
    // 64-byte alignment.  Falls back to std::vector when pool==nullptr.
    const uint64_t* evaluate(const MarketData& md,
                             double* output,
                             std::size_t n,
                             BufferPool* pool) const override {
        if (!pool) return evaluate(md, output, n);

        // 1. Evaluate RHS into a pool-allocated buffer
        auto rhsHandle = pool->allocate<double>(n);
        const uint64_t* rhsNull = rhs_->evaluate(md, rhsHandle.data(), n, pool);

        // 2. Evaluate LHS into the output buffer
        const uint64_t* lhsNull = lhs_->evaluate(md, output, n, pool);

        // 3. Build combined null mask
        const uint64_t* combinedNull = nullptr;
        BufferHandle<uint64_t> maskHandle;

        if (lhsNull == nullptr && rhsNull == nullptr) {
            combinedNull = nullptr;
        } else if (lhsNull == nullptr) {
            combinedNull = rhsNull;
        } else if (rhsNull == nullptr) {
            combinedNull = lhsNull;
        } else {
            std::size_t words = (n + 63) / 64;
            maskHandle = pool->allocate<uint64_t>(words);
            for (std::size_t w = 0; w < words; ++w) {
                maskHandle[w] = lhsNull[w] | rhsNull[w];
            }
            combinedNull = maskHandle.data();
        }

        // 4. Apply binary operator
        auto& reg = OperatorRegistry::instance();
        reg.invokeBinary(op_, Operand(output), Operand(rhsHandle.data()),
                         output, n, combinedNull);

        // rhsHandle + maskHandle released on scope exit
        return combinedNull;
    }

    void dump(std::ostream& os, int indent = 0) const override {
        os << std::string(indent, ' ') << binaryOpName(op_) << "\n";
        lhs_->dump(os, indent + 4);
        rhs_->dump(os, indent + 4);
    }

    std::size_t nodeCount() const override {
        return 1 + lhs_->nodeCount() + rhs_->nodeCount();
    }
    std::size_t maxDepth() const override {
        return 1 + std::max(lhs_->maxDepth(), rhs_->maxDepth());
    }

private:
    BinaryOpCode op_;
    std::unique_ptr<ExprNode> lhs_;
    std::unique_ptr<ExprNode> rhs_;

    // Mutable buffers for temporary storage during evaluate().
    // Stored as members so returned pointers remain valid after
    // evaluate() returns (the Engine may read them briefly).
    // NOT thread-safe — use clone() for concurrent evaluation.
    mutable std::vector<double>   rhsBuf_;
    mutable std::vector<uint64_t> combinedMask_;
};

}  // namespace quantcore
