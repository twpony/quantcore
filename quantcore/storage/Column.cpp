// Column.cpp — explicit template instantiations for Column<T>
// Phase: 一期必实现
//
// Most Column<T> logic lives in the header (template).  This translation
// unit provides explicit instantiations for the concrete types used by
// the engine, reducing compile-time bloat in consuming TUs.
#include "Column.h"

namespace quantcore {

// Explicit instantiations for all storage types used in the engine.
// NOTE: Column<bool> is intentionally NOT supported because
// std::vector<bool> is a packed bitset with no data() method and
// non-standard iterators.  Use Column<uint8_t> for boolean/signal
// storage instead (per design doc: "信号/标记 bool 或 int8_t").
template class Column<double>;
template class Column<int64_t>;
template class Column<uint8_t>;

// Comparison operators
template bool operator==(const Column<double>&, const Column<double>&);
template bool operator!=(const Column<double>&, const Column<double>&);
template bool operator==(const Column<int64_t>&, const Column<int64_t>&);
template bool operator!=(const Column<int64_t>&, const Column<int64_t>&);
template bool operator==(const Column<uint8_t>&, const Column<uint8_t>&);
template bool operator!=(const Column<uint8_t>&, const Column<uint8_t>&);

}  // namespace quantcore
