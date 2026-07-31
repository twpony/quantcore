// RedExpr.h — expression node for cross-section reduction operators
// Phase: 一期必实现
//
// RedExpr applies a cross-section reduction operator (SUM, MEAN, STD, ZSCORE,
// QUANTILE, ...) to its child expression.  Red operators are "fusion boundaries"
// — their child must be fully materialized before the cross-sectional computation
// can run, because every output element may depend on every input element.
//
// Dispatch is via OperatorRegistry::invokeRed, which uses the same type-erased
// function pointer pattern as invokeUnary / invokeBinary.
//
// extraParams_ stores per-call parameters for operators like RED_QUANTILE(q).
// For stateless operators the vector is empty and ignored.
//
// Thread safety: evaluate() is NOT thread-safe.  Use clone() to create
// independent copies for concurrent evaluation.
#pragma once

#include <memory>
#include <vector>

#include "quantcore/core/Types.h"
#include "quantcore/engine/BufferPool.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/registry/OperatorRegistry.h"
#include "quantcore/storage/ColView.h"

namespace quantcore {

class RedExpr : public ExprNode {
public:
    RedExpr(RedOpCode op,
            std::unique_ptr<ExprNode> child,
            std::vector<double> extraParams = {})
        : op_(op), child_(std::move(child)), extraParams_(std::move(extraParams)) {}

    // ============================================================
    // Accessors
    // ============================================================

    RedOpCode opCode() const noexcept { return op_; }
    const ExprNode* child() const noexcept { return child_.get(); }
    const std::vector<double>& extraParams() const noexcept { return extraParams_; }

    // ============================================================
    // ExprNode interface
    // ============================================================

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<RedExpr>(op_, child_->clone(), extraParams_);
    }

    const uint64_t* evaluate(const MarketData& md,
                             double* output,
                             std::size_t n) const override {
        // 1. Materialize child expression into a temporary buffer.
        //    Red ops need the full input array before computing.
        childBuf_.resize(n);
        const uint64_t* childNull = child_->evaluate(md, childBuf_.data(), n);

        // 2. Build a ColView wrapping the child's result.
        ColView<double> inputView(childBuf_.data(), n, childNull);

        // 3. Dispatch via OperatorRegistry::invokeRed.
        auto& reg = OperatorRegistry::instance();
        reg.invokeRed(op_, inputView, output, extraParams_);

        return childNull;
    }

    const uint64_t* evaluate(const MarketData& md,
                             double* output,
                             std::size_t n,
                             BufferPool* pool) const override {
        if (!pool) return evaluate(md, output, n);

        auto childHandle = pool->allocate<double>(n);
        const uint64_t* childNull = child_->evaluate(md, childHandle.data(), n, pool);
        ColView<double> inputView(childHandle.data(), n, childNull);

        auto& reg = OperatorRegistry::instance();
        reg.invokeRed(op_, inputView, output, extraParams_);

        return childNull;
    }

    // ============================================================
    // Debugging
    // ============================================================

    void dump(std::ostream& os, int indent = 0) const override {
        os << std::string(indent, ' ')
           << "RED_" << redOpName(op_);
        if (!extraParams_.empty()) {
            os << "(";
            for (std::size_t i = 0; i < extraParams_.size(); ++i) {
                if (i > 0) os << ", ";
                os << extraParams_[i];
            }
            os << ")";
        }
        os << "\n";
        child_->dump(os, indent + 4);
    }

    std::size_t nodeCount() const override { return 1 + child_->nodeCount(); }
    std::size_t maxDepth()  const override { return 1 + child_->maxDepth(); }

private:
    RedOpCode op_;
    std::unique_ptr<ExprNode> child_;
    std::vector<double> extraParams_;

    // Mutable buffer for the materialized child result.
    // NOT thread-safe — use clone() for concurrent evaluation.
    mutable std::vector<double> childBuf_;
};

}  // namespace quantcore
