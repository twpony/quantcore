// BufferHandle.h — RAII handle for pool-allocated or externally-owned memory
// Phase: 二期必实现
//
// BufferHandle<T> owns a contiguous array of T allocated from a BufferPool
// (or externally wrapped via wrap()).  On destruction (or explicit release()),
// pool-allocated memory is returned to the pool.  Externally-wrapped memory
// is simply detached.
//
// Move-only semantics — copying would create ambiguity about which handle
// owns the underlying memory.
//
// Usage:
//   auto handle = pool.allocate<double>(1000);
//   handle[0] = 3.14;
//   double* raw = handle.data();
//   // handle released automatically at scope exit
//
//   double* external = getSomeBuffer();
//   auto wrap = BufferHandle<double>::wrap(external, 500);
//   // wrap does NOT free external on destruction
#pragma once

#include <cstddef>
#include <utility>

namespace quantcore {

class BufferPool;  // forward

template <typename T>
class BufferHandle {
public:
    // ============================================================
    // Construction
    // ============================================================

    /// Empty handle (data_ == nullptr).
    BufferHandle() = default;

    /// Construct from raw parts.  Called by BufferPool::allocate().
    /// @param data  Aligned pointer from the pool.
    /// @param size  Element count.
    /// @param pool  Owning pool (may be nullptr for externally-owned memory).
    BufferHandle(T* data, std::size_t size, BufferPool* pool) noexcept
        : data_(data), size_(size), pool_(pool) {}

    /// Wrap externally-owned memory.  The handle does NOT free this memory
    /// on destruction — it simply detaches.
    static BufferHandle wrap(T* data, std::size_t size) noexcept {
        return BufferHandle(data, size, nullptr);
    }

    // ============================================================
    // Move semantics (only)
    // ============================================================

    BufferHandle(BufferHandle&& other) noexcept
        : data_(other.data_), size_(other.size_), pool_(other.pool_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.pool_ = nullptr;
    }

    BufferHandle& operator=(BufferHandle&& other) noexcept {
        if (this != &other) {
            release();
            data_ = other.data_;
            size_ = other.size_;
            pool_ = other.pool_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.pool_ = nullptr;
        }
        return *this;
    }

    BufferHandle(const BufferHandle&) = delete;
    BufferHandle& operator=(const BufferHandle&) = delete;

    // ============================================================
    // Destruction
    // ============================================================

    ~BufferHandle() { release(); }

    // ============================================================
    // Accessors
    // ============================================================

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    T& operator[](std::size_t i) { return data_[i]; }
    const T& operator[](std::size_t i) const { return data_[i]; }

    T* begin() noexcept { return data_; }
    T* end()   noexcept { return data_ + size_; }
    const T* begin() const noexcept { return data_; }
    const T* end()   const noexcept { return data_ + size_; }

    // ============================================================
    // Lifecycle management
    // ============================================================

    /// Return pool-allocated memory to the pool.  Idempotent — safe to
    /// call multiple times.  After release(), the handle is empty.
    void release() noexcept;

    /// Detach ownership and return the (pointer, size) pair.  The caller
    /// is now responsible for lifecycle management.  The handle is empty
    /// after detach().
    std::pair<T*, std::size_t> detach() noexcept {
        auto p = std::make_pair(data_, size_);
        data_ = nullptr;
        size_ = 0;
        pool_ = nullptr;
        return p;
    }

private:
    T*          data_ = nullptr;
    std::size_t size_ = 0;
    BufferPool* pool_ = nullptr;
};

// BufferHandle<T>::release() is defined in BufferPool.h after the full
// BufferPool definition is visible (avoids circular include dependency).

}  // namespace quantcore
