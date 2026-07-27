// RollingExpr.h — rolling-window expression node (CRTP)
// Phase: 远期预留接口 — placeholder, implementation deferred
//
// RollingExpr wraps a child expression with a RollingOpCode and a
// window-size parameter.  It is the expression-system counterpart
// to RollingOperator.
//
// Template parameters:
//   Op    — concrete rolling operator type (e.g. RollingSmaOp)
//   Child — sub-expression type (typically ColumnRef or another ExprNode)
//
// evaluateAt(i) = Op::evaluateScalar(child data, i, window)
#pragma once
