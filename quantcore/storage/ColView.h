// ColView.h — zero-copy read-only view into a Column<T> or raw buffer
// Phase: 一期必实现
//
// ColView<T> is a lightweight, non-owning reference to a contiguous region
// of T values with optional null-mask support.  It is the primary currency
// for passing data into operators and the execution engine.
//
// Lifecycle: the underlying data must outlive the ColView.  This is
// analogous to std::string_view or std::span — the view itself performs
// no lifetime management.
#pragma once

#include <cstddef>
#include <cstdint>

#include "quantcore/storage/Column.h"

namespace quantcore {

template <typename T>
class ColView {
public:
    // ============================================================
    // Construction
    // ============================================================

    /// Empty view (size = 0).
    ColView() = default;

    /// View a raw pointer range [start, end).  No null mask.
    ColView(const T* data, std::size_t start, std::size_t end)
        : data_(data + start)
        , nullMask_(nullptr)
        , size_(end > start ? end - start : 0)
        , nullMaskOffset_(0)
    {}

    /// View an entire Column.
    explicit ColView(const Column<T>& col)
        : data_(col.data())
        , nullMask_(col.hasNullMask() ? col.nullMaskData() : nullptr)
        , size_(col.size())
        , nullMaskOffset_(0)
    {}

    /// View a sub-range [start, end) of a Column.
    /// The null-mask bit for view position i corresponds to original
    /// Column position (start + i).
    ColView(const Column<T>& col, std::size_t start, std::size_t end)
        : data_(col.data() + start)
        , nullMask_(col.hasNullMask() ? col.nullMaskData() : nullptr)
        , size_(end > start ? end - start : 0)
        , nullMaskOffset_(start)
    {}

    /// Construct from raw pointer + size + optional null mask.
    /// The null mask is indexed from 0 (aligned with data[0]).
    ColView(const T* data, std::size_t size, const uint64_t* nullMask = nullptr)
        : data_(data)
        , nullMask_(nullMask)
        , size_(size)
        , nullMaskOffset_(0)
    {}

    // Default copy / move
    ColView(const ColView&) = default;
    ColView& operator=(const ColView&) = default;
    ColView(ColView&&) noexcept = default;
    ColView& operator=(ColView&&) noexcept = default;

    // ============================================================
    // Element access
    // ============================================================

    const T& operator[](std::size_t i) const {
        return data_[i];
    }

    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    // ============================================================
    // Null-mask queries (forwarded from underlying Column)
    // ============================================================

    bool hasNullMask() const noexcept { return nullMask_ != nullptr; }

    bool isNull(std::size_t i) const {
        if (!nullMask_) return false;
        std::size_t pos = nullMaskOffset_ + i;
        return (nullMask_[pos / 64] >> (pos % 64)) & uint64_t{1};
    }

    // ============================================================
    // Sub-view slicing
    // ============================================================

    /// Create a sub-view of [start, end) within the current view.
    /// start and end are relative to this view, not the original column.
    /// The null-mask offset propagates so that isNull(i) in the sub-view
    /// checks the correct bit in the original Column's null mask.
    ColView subView(std::size_t start, std::size_t end) const {
        ColView result;
        result.data_           = data_ + start;
        result.nullMask_       = nullMask_;
        result.size_           = (end > start ? end - start : 0);
        result.nullMaskOffset_ = nullMaskOffset_ + start;
        return result;
    }

    /// First `count` elements.
    ColView head(std::size_t count) const {
        return subView(0, count < size_ ? count : size_);
    }

    /// Last `count` elements.
    ColView tail(std::size_t count) const {
        std::size_t offset = count < size_ ? size_ - count : 0;
        return subView(offset, size_);
    }

    // ============================================================
    // Iteration support (range-based for)
    // ============================================================

    const T* begin() const noexcept { return data_; }
    const T* end()   const noexcept { return data_ + size_; }

private:
    const T*        data_           = nullptr;
    const uint64_t* nullMask_       = nullptr;
    std::size_t     size_           = 0;
    std::size_t     nullMaskOffset_ = 0;  // bit offset into nullMask_ (for sub-views)
};

// ============================================================
// Factory helpers
// ============================================================

template <typename T>
ColView<T> makeColView(const Column<T>& col) {
    return ColView<T>(col);
}

template <typename T>
ColView<T> makeColView(const Column<T>& col, std::size_t start, std::size_t end) {
    return ColView<T>(col, start, end);
}

template <typename T>
ColView<T> makeColView(const T* data, std::size_t size) {
    return ColView<T>(data, size);
}

}  // namespace quantcore
