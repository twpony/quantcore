// Column.h — generic columnar container with null-mask support
// Phase: 一期必实现
//
// Column<T> is the fundamental storage primitive of QuantCore.  It owns a
// contiguous, 64-byte-aligned array of T, plus an optional lazily-allocated
// bitmask that tracks null / missing values independently of NaN.
//
// Key properties:
//   - Memory is always 64-byte aligned (AVX-512 cache line)
//   - Null mask is only allocated when at least one null exists
//   - Support for zero-copy mmap construction (read-only wrap)
//   - Move-aware; copies are deep
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#include "quantcore/core/AlignedAllocator.h"
#include "quantcore/core/Types.h"

namespace quantcore {

// ============================================================
// Null mask helpers
// ============================================================

namespace detail {

// Number of uint64_t words needed for `n` bits
inline constexpr std::size_t nullMaskWords(std::size_t n) noexcept {
    return (n + 63) / 64;
}

// Set bit i in mask
inline void setNullBit(uint64_t* mask, std::size_t i) noexcept {
    mask[i / 64] |= (uint64_t{1} << (i % 64));
}

// Clear bit i in mask
inline void clearNullBit(uint64_t* mask, std::size_t i) noexcept {
    mask[i / 64] &= ~(uint64_t{1} << (i % 64));
}

// Test bit i in mask (true = null)
inline bool isNullBit(const uint64_t* mask, std::size_t i) noexcept {
    return (mask[i / 64] >> (i % 64)) & uint64_t{1};
}

// Count non-null entries in a mask (for statistics)
inline std::size_t countNulls(const uint64_t* mask, std::size_t size) noexcept {
    std::size_t count = 0;
    std::size_t words = nullMaskWords(size);
    for (std::size_t w = 0; w < words; ++w) {
        count += __builtin_popcountll(mask[w]);
    }
    return count;
}

}  // namespace detail

// ============================================================
// Column<T>
// ============================================================

template <typename T>
class Column {
public:
    using value_type      = T;
    using allocator_type  = AlignedAllocator<T, kSimdAlignment>;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;

    // ============================================================
    // Construction
    // ============================================================

    /// Default: empty column (size=0, no null mask).
    Column() = default;

    /// Allocate `size` default-initialized elements.
    explicit Column(std::size_t size)
        : data_(size)
        , hasNull_(false)
    {}

    /// Allocate `size` elements, each initialized to `value`.
    Column(std::size_t size, const T& value)
        : data_(size, value)
        , hasNull_(false)
    {}

    /// Copy from a raw pointer + size.
    Column(const T* src, std::size_t size)
        : data_(src, src + size)
        , hasNull_(false)
    {}

    /// From initializer_list (convenience for tests).
    Column(std::initializer_list<T> il)
        : data_(il)
        , hasNull_(false)
    {}

    /// Construct from mmap'd region.  The Column does NOT own the memory;
    /// the caller is responsible for keeping the mmap alive and for
    /// calling `releaseMmap()` before destruction.
    ///
    /// @param ptr   Pointer to the start of a T array in an mmap'd region.
    /// @param size  Number of valid T elements.
    /// @param nullMask  Optional null-mask pointer (may be nullptr).
    /// @return A Column that wraps the external memory.
    static Column fromMmap(const T* ptr,
                           std::size_t size,
                           const uint64_t* nullMask = nullptr) {
        Column col;
        col.mmapData_   = ptr;
        col.mmapSize_   = size;
        col.mmapNullMask_ = nullMask;
        col.hasNull_    = (nullMask != nullptr);
        return col;
    }

    /// Release mmap backing. After this call the column is empty.
    /// Safe to call on non-mmap columns (no-op).
    void releaseMmap() {
        mmapData_     = nullptr;
        mmapSize_     = 0;
        mmapNullMask_ = nullptr;
    }

    /// True when this column wraps an external mmap region.
    bool isMmap() const noexcept { return mmapData_ != nullptr; }

    // ============================================================
    // Copy & Move
    // ============================================================

    Column(const Column& other)
        : data_(other.data_)
        , nullMask_(other.nullMask_)
        , hasNull_(other.hasNull_)
        , mmapData_(nullptr)
        , mmapSize_(0)
        , mmapNullMask_(nullptr)
    {}

    Column& operator=(const Column& other) {
        if (this != &other) {
            data_     = other.data_;
            nullMask_ = other.nullMask_;
            hasNull_  = other.hasNull_;
        }
        return *this;
    }

    Column(Column&& other) noexcept
        : data_(std::move(other.data_))
        , nullMask_(std::move(other.nullMask_))
        , hasNull_(other.hasNull_)
        , mmapData_(other.mmapData_)
        , mmapSize_(other.mmapSize_)
        , mmapNullMask_(other.mmapNullMask_)
    {
        other.hasNull_ = false;
        other.mmapData_ = nullptr;
        other.mmapSize_ = 0;
        other.mmapNullMask_ = nullptr;
    }

    Column& operator=(Column&& other) noexcept {
        if (this != &other) {
            data_       = std::move(other.data_);
            nullMask_   = std::move(other.nullMask_);
            hasNull_    = other.hasNull_;
            mmapData_   = other.mmapData_;
            mmapSize_   = other.mmapSize_;
            mmapNullMask_ = other.mmapNullMask_;
            other.hasNull_ = false;
            other.mmapData_ = nullptr;
            other.mmapSize_ = 0;
            other.mmapNullMask_ = nullptr;
        }
        return *this;
    }

    // ============================================================
    // Element access
    // ============================================================

    T& operator[](std::size_t i) {
        return const_cast<T&>(static_cast<const Column*>(this)->operator[](i));
    }

    const T& operator[](std::size_t i) const {
        if (mmapData_) {
            return mmapData_[i];
        }
        return data_[i];
    }

    T* data() {
        return const_cast<T*>(static_cast<const Column*>(this)->data());
    }

    const T* data() const {
        if (mmapData_) return mmapData_;
        return data_.data();
    }

    std::size_t size() const noexcept {
        if (mmapData_) return mmapSize_;
        return data_.size();
    }

    bool empty() const noexcept { return size() == 0; }

    std::size_t capacity() const noexcept {
        if (mmapData_) return mmapSize_;
        return data_.capacity();
    }

    // ============================================================
    // Null-mask management
    // ============================================================

    /// True if any element has been explicitly marked null.
    bool hasNullMask() const noexcept {
        if (mmapData_) return mmapNullMask_ != nullptr;
        return hasNull_;
    }

    /// Mark element `i` as null.  Allocates the bitmask on first use.
    void setNull(std::size_t i) {
        ensureNullMask();
        detail::setNullBit(nullMask_.data(), i);
    }

    /// Check if element `i` is marked null.
    bool isNull(std::size_t i) const {
        if (mmapNullMask_) {
            return detail::isNullBit(mmapNullMask_, i);
        }
        if (!hasNull_) return false;
        return detail::isNullBit(nullMask_.data(), i);
    }

    /// Clear the null mark at position `i`.
    void clearNull(std::size_t i) {
        if (hasNull_ && !isMmap()) {
            detail::clearNullBit(nullMask_.data(), i);
        }
    }

    /// Raw pointer to the null bitmask (nullptr if no null mask exists).
    const uint64_t* nullMaskData() const noexcept {
        if (mmapNullMask_) return mmapNullMask_;
        if (!hasNull_) return nullptr;
        return nullMask_.data();
    }

    /// Number of null-marked elements.
    std::size_t nullCount() const {
        if (mmapNullMask_) {
            return detail::countNulls(mmapNullMask_, mmapSize_);
        }
        if (!hasNull_) return 0;
        return detail::countNulls(nullMask_.data(), data_.size());
    }

    /// Mark all elements valid and release the null mask.
    void clearAllNulls() {
        if (!isMmap()) {
            nullMask_.clear();
            nullMask_.shrink_to_fit();
            hasNull_ = false;
        }
    }

    // ============================================================
    // Memory properties
    // ============================================================

    /// True if the data pointer meets the 64-byte alignment requirement.
    bool isAligned() const noexcept {
        return reinterpret_cast<std::uintptr_t>(data()) % kSimdAlignment == 0;
    }

    /// Byte size of the data region.
    std::size_t memoryBytes() const noexcept {
        return size() * sizeof(T);
    }

    // ============================================================
    // Mutation
    // ============================================================

    /// Reserve capacity (no-op for mmap columns).
    void reserve(std::size_t cap) {
        if (!isMmap()) data_.reserve(cap);
    }

    /// Resize the column. New elements are default-initialized.
    /// Null mask is resized accordingly.
    void resize(std::size_t newSize) {
        if (isMmap()) return;
        data_.resize(newSize);
        if (hasNull_) {
            nullMask_.resize(detail::nullMaskWords(newSize), 0);
        }
    }

    /// Resize and fill with `value`.
    void resize(std::size_t newSize, const T& value) {
        if (isMmap()) return;
        data_.resize(newSize, value);
        if (hasNull_) {
            nullMask_.resize(detail::nullMaskWords(newSize), 0);
        }
    }

    /// Clear all data.
    void clear() {
        if (!isMmap()) {
            data_.clear();
            nullMask_.clear();
            hasNull_ = false;
        }
    }

    // ============================================================
    // Iterators
    // ============================================================

    auto begin()        { return data_.begin(); }
    auto begin()  const { return data_.begin(); }
    auto end()          { return data_.end(); }
    auto end()    const { return data_.end(); }
    auto cbegin() const { return data_.cbegin(); }
    auto cend()   const { return data_.cend(); }

private:
    // Allocate the null mask if not already present.
    void ensureNullMask() {
        if (hasNull_) return;
        nullMask_.resize(detail::nullMaskWords(size()), 0);
        hasNull_ = true;
    }

    // Owned storage
    std::vector<T, AlignedAllocator<T, kSimdAlignment>> data_;
    std::vector<uint64_t> nullMask_;
    bool hasNull_ = false;

    // mmap external storage (zero-copy)
    const T*        mmapData_     = nullptr;
    std::size_t     mmapSize_     = 0;
    const uint64_t* mmapNullMask_ = nullptr;
};

// ============================================================
// Comparison helpers (useful for tests)
// ============================================================

template <typename T>
bool operator==(const Column<T>& a, const Column<T>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a.isNull(i) != b.isNull(i)) return false;
        if (!a.isNull(i) && a[i] != b[i]) return false;
    }
    return true;
}

template <typename T>
bool operator!=(const Column<T>& a, const Column<T>& b) {
    return !(a == b);
}

}  // namespace quantcore
