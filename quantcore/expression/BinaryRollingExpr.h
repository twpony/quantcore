// BinaryRollingExpr.h — expression node for binary rolling-window operators
// Phase: 远期扩展
//
// BinaryRollingExpr applies a binary rolling-window operator (CORR, COV)
// to two child expressions.  The window size is stored as a runtime value.
//
// Binary rolling operators need TWO input series, unlike standard rolling
// operators which only need one.  Both children are materialized before
// the rolling computation.
//
// BinaryRollingExpr is a "fusion boundary" — its children must be fully
// materialized before the rolling window can be computed.
//
// Dispatch is via OperatorRegistry::invokeBinaryRolling.
//
// Boundary convention:
//   - Window operators: first (window-1) elements are NaN.
#pragma once

#include <memory>
#include <stdexcept>

#include "quantcore/core/Types.h"
#include "quantcore/engine/BufferPool.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/registry/OperatorRegistry.h"
#include "quantcore/storage/ColView.h"

namespace quantcore {

class BinaryRollingExpr : public ExprNode {
public:
    BinaryRollingExpr(RollingOpCode op, std::size_t window,
                      std::unique_ptr<ExprNode> child1,
                      std::unique_ptr<ExprNode> child2,
                      std::vector<double> extraParams = {})
        : op_(op), window_(window),
          child1_(std::move(child1)), child2_(std::move(child2)),
          extraParams_(std::move(extraParams)) {}

    // ============================================================
    // Accessors
    // ============================================================

    RollingOpCode opCode() const noexcept { return op_; }
    std::size_t window() const noexcept { return window_; }
    const ExprNode* child1() const noexcept { return child1_.get(); }
    const ExprNode* child2() const noexcept { return child2_.get(); }
    const std::vector<double>& extraParams() const noexcept { return extraParams_; }

    // ============================================================
    // ExprNode interface
    // ============================================================

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<BinaryRollingExpr>(op_, window_,
            child1_->clone(), child2_->clone(), extraParams_);
    }

    const uint64_t* evaluate(const MarketData& md,
                             double* output,
                             std::size_t n) const override {
        childBuf1_.resize(n);
        childBuf2_.resize(n);
        const uint64_t* null1 = child1_->evaluate(md, childBuf1_.data(), n);
        const uint64_t* null2 = child2_->evaluate(md, childBuf2_.data(), n);

        // Merge null masks: if either input is null, mark position as null.
        // For simplicity, we pass the first mask if present, otherwise the second.
        // The operator treats NaN in the input naturally.
        ColView<double> view1(childBuf1_.data(), n, null1);
        ColView<double> view2(childBuf2_.data(), n, null2);

        auto& reg = OperatorRegistry::instance();
        reg.invokeBinaryRolling(op_, view1, view2, output, window_, extraParams_);

        return nullptr;
    }

    const uint64_t* evaluate(const MarketData& md,
                             double* output,
                             std::size_t n,
                             BufferPool* pool) const override {
        if (!pool) return evaluate(md, output, n);

        auto handle1 = pool->allocate<double>(n);
        auto handle2 = pool->allocate<double>(n);
        const uint64_t* null1 = child1_->evaluate(md, handle1.data(), n, pool);
        const uint64_t* null2 = child2_->evaluate(md, handle2.data(), n, pool);

        ColView<double> view1(handle1.data(), n, null1);
        ColView<double> view2(handle2.data(), n, null2);

        auto& reg = OperatorRegistry::instance();
        reg.invokeBinaryRolling(op_, view1, view2, output, window_, extraParams_);

        return nullptr;
    }

    void dump(std::ostream& os, int indent = 0) const override {
        os << std::string(indent, ' ')
           << rollingOpName(op_) << "(" << window_;
        if (!extraParams_.empty()) {
            for (auto p : extraParams_) os << ", " << p;
        }
        os << ")\n";
        child1_->dump(os, indent + 4);
        child2_->dump(os, indent + 4);
    }

    std::size_t nodeCount() const override {
        return 1 + child1_->nodeCount() + child2_->nodeCount();
    }
    std::size_t maxDepth()  const override {
        return 1 + std::max(child1_->maxDepth(), child2_->maxDepth());
    }

private:
    RollingOpCode op_;
    std::size_t window_;
    std::unique_ptr<ExprNode> child1_;
    std::unique_ptr<ExprNode> child2_;
    std::vector<double> extraParams_;

    mutable std::vector<double> childBuf1_;
    mutable std::vector<double> childBuf2_;
};

}  // namespace quantcore
