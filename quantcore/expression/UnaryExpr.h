// UnaryExpr.h — expression node for unary operators
// Phase: 一期必实现
//
// UnaryExpr applies a unary operator (ABS, LOG, SQRT, NEG, ...) to
// its child expression.  Dispatch is done at runtime via the
// OperatorRegistry's invokeUnary, which uses the existing dispatch
// table — no template parameter needed on the AST node.
//
// evaluate() uses in-place computation: the child writes to the output
// buffer, then the unary operator reads from and writes back to the
// same buffer.  This works because unary operators are element-wise
// (output[i] depends only on input[i]).
//
// null propagation: the child's null mask is passed through to the
// operator.  The UnaryOperator CRTP base class already handles null
// checking before calling evaluateScalar().
#pragma once

#include <memory>
#include <stdexcept>

#include "quantcore/core/Types.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/registry/OperatorRegistry.h"

namespace quantcore {

class UnaryExpr : public ExprNode {
public:
    UnaryExpr(UnaryOpCode op, std::unique_ptr<ExprNode> child)
        : op_(op), child_(std::move(child)) {}

    // ============================================================
    // Accessors
    // ============================================================

    UnaryOpCode opCode() const noexcept { return op_; }
    const ExprNode* child() const noexcept { return child_.get(); }

    // ============================================================
    // ExprNode interface
    // ============================================================

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<UnaryExpr>(op_, child_->clone());
    }

    const uint64_t* evaluate(const MarketData& md,
                             double* output,
                             std::size_t n) const override {
        // 1. Evaluate child into output (in-place)
        const uint64_t* childNull = child_->evaluate(md, output, n);

        // 2. Apply unary operator via OperatorRegistry dispatch.
        //    Operand wraps the output pointer as a column operand.
        auto& reg = OperatorRegistry::instance();
        Operand input(output);
        reg.invokeUnary(op_, input, output, n, childNull);

        return childNull;
    }

    void dump(std::ostream& os, int indent = 0) const override {
        os << std::string(indent, ' ') << unaryOpName(op_) << "\n";
        child_->dump(os, indent + 4);
    }

    std::size_t nodeCount() const override { return 1 + child_->nodeCount(); }
    std::size_t maxDepth()  const override { return 1 + child_->maxDepth(); }

private:
    UnaryOpCode op_;
    std::unique_ptr<ExprNode> child_;
};

}  // namespace quantcore
