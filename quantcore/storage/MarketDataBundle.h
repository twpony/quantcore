// MarketDataBundle.h — backward-compatibility alias for PanelData
// Phase: 五期实现
//
// MarketDataBundle was the original placeholder name for multi-asset panel
// data.  The implementation now lives in PanelData.  This header includes
// PanelData and provides a type alias so that existing code using the name
// MarketDataBundle continues to compile.
//
// New code should use PanelData directly.

#pragma once

#include "quantcore/storage/PanelData.h"

namespace quantcore {

/// MarketDataBundle is a backward-compatible alias for PanelData.
/// Prefer PanelData in new code.
using MarketDataBundle = PanelData;

}  // namespace quantcore
