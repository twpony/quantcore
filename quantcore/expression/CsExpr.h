// CsExpr.h — expression node for cross-section transform operators
// Phase: 一期必实现
//
// CsExpr applies a cross-section transform operator (RANK, ZSCORE, NORMALIZE,
// WINSORIZE, CLIP, ...) to its child expression.  Cs operators are "fusion
// boundaries" — their child must be fully materialized before the cross-sectional
// computation can run.
//
// Dispatch is via OperatorRegistry::invokeCs.
//
// extraParams_ stores per-call parameters for operators like CS_WINSORIZE(lowerPct,
// upperPct), CS_WINSORIZE_MAD(n), and CS_CLIP(lower, upper).  For stateless
// operators the vector is empty and ignored.
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

class CsExpr : public ExprNode {
public:
    CsExpr(CsOpCode op,
           std::unique_ptr<ExprNode> child,
           std::vector<double> extraParams = {})
        : op_(op), child_(std::move(child)), extraParams_(std::move(extraParams)) {}

    // ============================================================
    // Accessors
    // ============================================================

    CsOpCode opCode() const noexcept { return op_; }
    const ExprNode* child() const noexcept { return child_.get(); }
    const std::vector<double>& extraParams() const noexcept { return extraParams_; }

    // ============================================================
    // ExprNode interface
    // ============================================================

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<CsExpr>(op_, child_->clone(), extraParams_);
    }

    // ============================================================
    // IMPORTANT: CsExpr CANNOT be evaluated on a single MarketData.
    //
    // Cross-sectional operators (CS_RANK, CS_ZSCORE, ...) operate on
    // the cross-section of ALL assets at a single date.  Evaluating
    // them on a single asset's time-series (MarketData) would apply
    // the operator within that one asset's own history — which is
    // semantically wrong for cross-sectional factors.
    //
    // The correct evaluation path is through FactorCalculator:
    //
    //   // Expression-based:
    //   calc.evaluateCSExpression("log(volume)", panel, CS_RANK);
    //
    //   // Or with a pre-built AST:
    //   calc.evaluateCS(CS_RANK, innerExpr, panel);
    //
    // These methods evaluate the child expression per-asset, then
    // apply the CS operator to the cross-sectional vector at each
    // date via PanelData.
    // ============================================================

    const uint64_t* evaluate(const MarketData& /*md*/,
                             double* /*output*/,
                             std::size_t /*n*/) const override {
        throw std::logic_error(
            std::string("CsExpr::evaluate(MarketData): cross-sectional "
                        "operator '") + csOpName(op_) + "' cannot be "
                        "evaluated on a single asset. "
                        "Use FactorCalculator::evaluateCSExpression() "
                        "or FactorCalculator::evaluateCS() with a "
                        "PanelData instead.");
    }

    const uint64_t* evaluate(const MarketData& md,
                             double* output,
                             std::size_t n,
                             BufferPool* /*pool*/) const override {
        return evaluate(md, output, n);  // routes to the error above
    }

    // ============================================================
    // Debugging
    // ============================================================

    void dump(std::ostream& os, int indent = 0) const override {
        os << std::string(indent, ' ')
           << "CS_" << csOpName(op_);
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
    CsOpCode op_;
    std::unique_ptr<ExprNode> child_;
    std::vector<double> extraParams_;
};

}  // namespace quantcore
