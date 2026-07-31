// ExprTraits.h — compile-time type traits and runtime helpers for expression trees
// Phase: 一期必实现
//
// Provides:
//   1. Compile-time node category traits (leaf vs internal vs fusion-boundary)
//   2. Runtime tree analysis (wrappers around virtual methods on ExprNode)
//   3. exprToString() convenience for debugging
//
// Phase 4 (FusedLoopGenerator) will extend this with fusion eligibility analysis.
#pragma once

#include <cstddef>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>

#include "quantcore/expression/ExprNode.h"

namespace quantcore {

// ============================================================
// Compile-time node category traits
// ============================================================

/// Primary template: not a leaf.
template <typename T>
struct is_leaf_node : std::false_type {};

// Forward declarations for specialization.
class ColumnRef;
class Scalar;

template <> struct is_leaf_node<ColumnRef> : std::true_type {};
template <> struct is_leaf_node<Scalar>    : std::true_type {};

/// Primary template: not fusible by default.
template <typename T>
struct is_fusible : std::false_type {};

// UnaryExpr and BinaryExpr are element-wise and can be fused into a single loop.
class UnaryExpr;
class BinaryExpr;

template <> struct is_fusible<UnaryExpr>  : std::true_type {};
template <> struct is_fusible<BinaryExpr> : std::true_type {};

// ============================================================
// Runtime tree analysis (delegates to virtual methods on ExprNode)
// ============================================================

/// Total number of nodes in the subtree rooted at `node`.
inline std::size_t exprNodeCount(const ExprNode* node) {
    return node ? node->nodeCount() : 0;
}

/// Maximum depth of the subtree rooted at `node` (leaf = depth 0).
inline std::size_t exprDepth(const ExprNode* node) {
    return node ? node->maxDepth() : 0;
}

/// Pretty-print the expression tree to an ostream.
inline void exprDump(const ExprNode* node, std::ostream& os, int indent = 0) {
    if (node) {
        node->dump(os, indent);
    } else {
        os << std::string(indent, ' ') << "(null)\n";
    }
}

/// Return a string representation of the expression tree.
inline std::string exprToString(const ExprNode* node) {
    if (!node) return "(null)";
    std::ostringstream oss;
    node->dump(oss, 0);
    return oss.str();
}

}  // namespace quantcore
