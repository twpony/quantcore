// ExprNode.h — abstract base class for expression AST nodes
// Phase: 一期必实现
//
// All expression tree nodes derive from this class.  The design uses
// dynamic polymorphism (virtual functions + unique_ptr) rather than
// CRTP or variant for simplicity in the prototype phase.  Migration
// to a static-dispatch approach is possible if profiling shows the
// virtual-call overhead matters.
//
// The expression tree is NOT bound to a specific MarketData instance
// at construction time.  Instead, evaluate() receives the MarketData
// as a parameter, enabling the same tree to be cloned and evaluated
// on different datasets (critical for factor reuse across stocks).
//
// evaluate() writes into a caller-allocated output buffer, keeping
// memory management in the Engine/BufferPool layer rather than in
// individual nodes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>

#include "quantcore/storage/MarketData.h"

namespace quantcore {

class BufferPool;  // forward (engine/BufferPool.h)

class ExprNode {
public:
    virtual ~ExprNode() = default;

    // ============================================================
    // Deep clone — returns an independent copy of the entire subtree
    // ============================================================

    virtual std::unique_ptr<ExprNode> clone() const = 0;

    // ============================================================
    // Evaluate this subtree
    // ============================================================
    //
    // @param md       MarketData context for resolving ColumnRef leaves.
    // @param output   Pre-allocated output buffer (length = n).
    //                 Must be writable and 64-byte aligned for SIMD.
    // @param n        Number of elements to compute.
    // @return         Pointer to the result's null bitmask, or nullptr
    //                 if the result has no nulls.  The returned pointer
    //                 is valid only until the next evaluate() call on
    //                 this tree (it may point into an internal buffer).
    //
    // Thread safety: evaluate() is not thread-safe on the same node.
    // Use clone() to create independent copies for concurrent use.

    virtual const uint64_t* evaluate(const MarketData& md,
                                     double* output,
                                     std::size_t n) const = 0;

    // ============================================================
    // Evaluate with BufferPool
    // ============================================================
    //
    // Nodes that need temporary buffers (BinaryExpr, RollingExpr,
    // RedExpr, CsExpr) should override this to allocate from the pool
    // instead of using internal std::vector members.  The default
    // implementation falls back to the no-pool evaluate().
    //
    // @param pool  Optional BufferPool for aligned temporary allocations.
    //              If nullptr, internal std::vector buffers are used (as
    //              in the no-pool overload).  The pool must outlive this
    //              evaluate() call — returned pointers (null masks) are
    //              only valid until the pool deallocation or the next
    //              pool allocation.

    virtual const uint64_t* evaluate(const MarketData& md,
                                     double* output,
                                     std::size_t n,
                                     BufferPool* /*pool*/) const {
        return evaluate(md, output, n);
    }

    // ============================================================
    // Debugging — pretty-print the AST
    // ============================================================

    /// Write a human-readable tree representation to an ostream.
    /// Each node prints itself on one line; children are indented by
    /// `indent` spaces.  The default implementation prints "ExprNode".
    virtual void dump(std::ostream& os, int indent = 0) const {
        os << std::string(indent, ' ') << "ExprNode\n";
    }

    // ============================================================
    // Tree analysis
    // ============================================================

    /// Total number of nodes in this subtree (including this node).
    virtual std::size_t nodeCount() const { return 1; }

    /// Maximum depth of this subtree (leaf = depth 0).
    virtual std::size_t maxDepth() const { return 0; }
};

}  // namespace quantcore
