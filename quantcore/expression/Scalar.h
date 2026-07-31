// Scalar.h — leaf expression node holding a constant double value
// Phase: 一期必实现
//
// Scalar represents a compile-time constant in an expression tree.
// When evaluated, it fills the entire output buffer with the constant
// value.  Scalars never produce nulls.
//
// Typical usage:
//   auto three = std::make_unique<Scalar>(3.0);
//   auto expr  = std::make_unique<BinaryExpr>(BinaryOpCode::MUL,
//                      std::move(closeRef), std::move(three));
//   // → CLOSE * 3.0
#pragma once

#include <algorithm>
#include <memory>

#include "quantcore/expression/ExprNode.h"

namespace quantcore {

class Scalar : public ExprNode {
public:
    explicit Scalar(double value) : value_(value) {}

    // ============================================================
    // Accessors
    // ============================================================

    double value() const noexcept { return value_; }

    // ============================================================
    // ExprNode interface
    // ============================================================

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<Scalar>(value_);
    }

    const uint64_t* evaluate(const MarketData& /*md*/,
                             double* output,
                             std::size_t n) const override {
        std::fill_n(output, n, value_);
        return nullptr;  // scalars never have nulls
    }

    void dump(std::ostream& os, int indent = 0) const override {
        os << std::string(indent, ' ') << "SCALAR(" << value_ << ")\n";
    }

private:
    double value_;
};

}  // namespace quantcore
