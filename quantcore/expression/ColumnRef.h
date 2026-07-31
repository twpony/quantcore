// ColumnRef.h — leaf expression node referencing a MarketData field
// Phase: 一期必实现
//
// ColumnRef is the simplest expression node — it refers to one of the
// 7 standard fields (OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT, VWAP)
// in a MarketData instance.  When evaluated, it copies the underlying
// column data into the output buffer.
//
// This copy is intentional: it decouples the expression tree from the
// storage layout.  When FusedLoopGenerator (Phase 4) processes a chain
// like ABS(CLOSE), the copy is eliminated — ColumnRef is inlined into
// the fused loop body.
#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>

#include "quantcore/core/Types.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/storage/MarketData.h"

namespace quantcore {

class ColumnRef : public ExprNode {
public:
    explicit ColumnRef(Field field) : field_(field) {}

    // ============================================================
    // Accessors
    // ============================================================

    Field field() const noexcept { return field_; }

    // ============================================================
    // ExprNode interface
    // ============================================================

    std::unique_ptr<ExprNode> clone() const override {
        return std::make_unique<ColumnRef>(field_);
    }

    const uint64_t* evaluate(const MarketData& md,
                             double* output,
                             std::size_t n) const override {
        const auto& col = md.column<double>(field_);
        const double* src = col.data();
        std::memcpy(output, src, n * sizeof(double));

        // Zero-copy: return the source column's null mask.
        // The caller must not modify it.
        return col.nullMaskData();  // nullptr if the column has no nulls
    }

    void dump(std::ostream& os, int indent = 0) const override {
        os << std::string(indent, ' ') << "COLUMN(" << fieldName(field_) << ")\n";
    }

private:
    Field field_;
};

}  // namespace quantcore
