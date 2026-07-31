# Phase 2 详细设计方案：ExecutionEngine + BufferPool

**日期**: 2026-07-27

**状态**: ✅ 已完成 — ExecutionEngine + BufferPool + BufferHandle + EngineMetrics 全部实现并测试通过

---

## 零、现状

| 文件 | 状态 |
|------|------|
| `engine/ExecutionEngine.h` | **空文件 (0 bytes)** |
| `engine/ExecutionEngine.cpp` | **空文件 (0 bytes)** |
| `engine/BufferPool.h` | **空文件 (0 bytes)** |
| `engine/BufferPool.cpp` | **空文件 (0 bytes)** |
| `engine/BufferHandle.h` | **空文件 (0 bytes)** |
| `engine/EngineMetrics.h` | **空文件 (0 bytes)** |
| `core/AlignedAllocator.h` | ✅ 完成 — 64 字节对齐分配器 (`std::aligned_alloc`) |
| `core/ErrorHandling.h` | ✅ 完成 — `ResourceError` 已预留 "BufferPool exhaustion" 注释 |
| `tests/unit/test_buffer_pool.cpp` | **空文件 (0 bytes)** |
| CMakeLists.txt | ✅ 已引用 `engine/ExecutionEngine.cpp` 和 `engine/BufferPool.cpp` |

**关键观察**:
- 当前表达式节点内部使用 `std::vector<double>` 做临时缓冲区 — 使用 `std::allocator`，**无 64 字节对齐保证**
- `BinaryExpr` 注释明确写道："Phase 2 will replace this with BufferPool-backed allocation"
- `Column<T>` 已使用 `AlignedAllocator<T>`，但表达式节点未使用
- 求值模式为调用者提供 output buffer；测试代码手动创建 `std::vector<double> output(kN)`

---

## 一、架构总览

```
用户
 │ engine.evaluate(expr, md)
 ▼
ExecutionEngine
 ├── BufferPool      — 分级 Slab 分配器，提供对齐临时内存
 ├── BufferHandle<T> — RAII 句柄，析构自动归还
 ├── EngineMetrics   — 性能统计（分配次数、求值耗时）
 └── evaluate()      — 后序遍历 AST，管理临时列生命周期，返回 Column<double>
```

### 数据流

```
engine.evaluate(expr, md):
  │
  ├─ 1. 获取行数 n = md.rowCount()
  │
  ├─ 2. 后序遍历 AST，自底向上求值:
  │      ColumnRef(CLOSE) → memcpy → output (caller-allocated)
  │      ColumnRef(VWAP)  → memcpy → tmp (pool-allocated)
  │      SUB(output, tmp)          → output (in-place, tmp freed)
  │      ABS(output)               → output (in-place)
  │      MUL(output, COLUMN(VOLUME)) → result (pool-allocated)
  │
  └─ 3. 从 pool 分配结果缓冲区 → 复制结果 → 返回 Column<double>
```

---

## 二、核心设计决策

### 决策 1：Engine 是否接管节点内部缓冲区？

**选型 A**：Engine 只分配顶层输出，节点内部继续用 `std::vector<double>`
- 优点：改动最小，与 Phase 1 完全兼容
- 缺点：内部缓冲区未对齐，未受益于 BufferPool

**选型 B**：Engine 分配所有临时缓冲区，通过 `evaluate()` 的 `BufferPool*` 参数传入
- 优点：全部 64 字节对齐，BufferPool 统一管理，性能更好
- 缺点：需要修改所有表达节点的内部实现

**→ 选择 B**，理由：
- `BinaryExpr` 注释明确要求 Phase 2 替换内部缓冲区
- 64 字节对齐对 SIMD 性能至关重要
- 改动模式一致：每个内部节点新增 `evaluate(md, output, n, pool)` 重载，保留旧接口向后兼容

### 决策 2：BufferPool 的粒度

**选型 A**：全局单例
- 优点：简单
- 缺点：多线程竞争

**选型 B**：每个 ExecutionEngine 持有一个 BufferPool 实例
- 优点：线程局部，无锁，Engine 销毁时统一回收
- 缺点：跨 Engine 无法共享

**→ 选择 B**。多线程场景下每个线程创建独立 Engine。BufferPool 内部不做线程同步。

### 决策 3：Slab 大小分级

参考 jemalloc 思路，按典型因子计算场景分级：

| 级别 | 块大小 | 每个 64KB Slab 含块数 | 用途 |
|------|--------|----------------------|------|
| Tiny | 256 B | 256 | 超小窗口（~32 个 double） |
| Small | 4 KB | 16 | 日线数据（~500 行） |
| Medium | 16 KB | 4 | 中长期数据（~2K 行） |
| Large | 64 KB | 1 | 分钟线（~8K 行） |
| Huge | 256 KB | 0 (独立分配) | tick 数据（~32K 行） |

超过 256 KB 的分配直接走 `std::aligned_alloc` / `std::free`，不入池。

### 决策 4：BufferHandle 是否参数化？

使用 `template <typename T>` 的 `BufferHandle<T>`：
- `BufferHandle<double>` 用于数据缓冲区
- `BufferHandle<uint64_t>` 用于 null mask 缓冲区
- 类型安全，且支持 `operator[]` 和 range-based for

---

## 三、逐文件详细设计

### 3.1 BufferHandle.h — RAII 内存句柄

```cpp
// BufferHandle.h — RAII handle for pool-allocated or externally-owned memory
// Phase: 二期必实现
#pragma once

#include <cstddef>
#include <utility>

namespace quantcore {

class BufferPool;  // fwd

template <typename T>
class BufferHandle {
public:
    BufferHandle() = default;

    // 从 BufferPool 分配
    BufferHandle(T* data, std::size_t size, BufferPool* pool) noexcept
        : data_(data), size_(size), pool_(pool) {}

    // 包装外部内存（不归还池，析构时仅置零）
    static BufferHandle wrap(T* data, std::size_t size) noexcept {
        return BufferHandle(data, size, nullptr);
    }

    // 移动 only
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

    ~BufferHandle() { release(); }

    // 访问器
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

    // 提前归还池（幂等）
    void release() noexcept;

    // 释放所有权 — 调用者负责管理生命周期
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

// 模板实现（在头文件中）
template <typename T>
void BufferHandle<T>::release() noexcept {
    if (data_ && pool_) {
        pool_->deallocate(data_, size_);
    }
    data_ = nullptr;
    size_ = 0;
    pool_ = nullptr;
}

}  // namespace quantcore
```

### 3.2 BufferPool.h/.cpp — 分级 Slab 分配器

```cpp
// BufferPool.h — slab-based allocator for aligned temporary buffers
// Phase: 二期必实现
//
// Allocates 64-byte aligned memory in size-class slabs.  Memory is NOT
// zeroed.  Deallocated blocks are reused via per-size-class free lists.
//
// Each slab is 64 KB (one cache-line-aligned page).  Blocks within a
// slab are tracked by a bitmap (1 = free, 0 = allocated).
//
// Overflow (>256 KB) uses raw aligned_alloc/free and is not pooled.
//
// Thread safety: BufferPool is NOT thread-safe.  Create one per thread.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "quantcore/core/Types.h"

namespace quantcore {

template <typename T> class BufferHandle;

class BufferPool {
public:
    // 大小分级
    enum class SizeClass : uint8_t {
        kTiny   = 0,   // 256 B    (32 doubles)
        kSmall  = 1,   // 4 KB     (512 doubles)
        kMedium = 2,   // 16 KB    (2048 doubles)
        kLarge  = 3,   // 64 KB    (8192 doubles)
        kHuge   = 4,   // 256 KB   (32768 doubles)
        kCount  = 5,
    };

    static constexpr std::size_t kSlabSize = 64 * 1024;  // 64 KB
    static constexpr std::size_t kOverflowThreshold = 256 * 1024;  // 256 KB

    BufferPool() = default;
    ~BufferPool();

    BufferPool(const BufferPool&) = delete;
    BufferPool& operator=(const BufferPool&) = delete;
    BufferPool(BufferPool&&) = delete;
    BufferPool& operator=(BufferPool&&) = delete;

    // ============================================================
    // Public typed interface
    // ============================================================

    /// Allocate `n` aligned elements of type T.
    template <typename T>
    BufferHandle<T> allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        void* ptr = allocateRaw(bytes);
        return BufferHandle<T>(static_cast<T*>(ptr), n, this);
    }

    /// Deallocate a buffer returned by allocate<T>().
    template <typename T>
    void deallocate(T* ptr, std::size_t n) noexcept {
        deallocateRaw(static_cast<void*>(ptr), n * sizeof(T));
    }

    // ============================================================
    // Statistics
    // ============================================================

    std::size_t totalAllocated()   const noexcept { return totalAllocated_;   }
    std::size_t totalDeallocated() const noexcept { return totalDeallocated_; }
    std::size_t slabCount()        const noexcept { return slabs_.size();     }
    std::size_t freeListSize(SizeClass sc) const noexcept;

private:
    // Size classification
    static SizeClass classify(std::size_t bytes) noexcept;
    static std::size_t blockSize(SizeClass sc) noexcept;

    // Raw (untyped) allocation / deallocation
    void* allocateRaw(std::size_t bytes);
    void  deallocateRaw(void* ptr, std::size_t bytes) noexcept;

    // Allocate a new slab for the given size class
    void  allocateSlab(SizeClass sc);

    struct Slab {
        void*               memory;      // 64-KB aligned chunk
        std::size_t         blockSize;   // per-block bytes
        std::size_t         numBlocks;
        std::vector<uint8_t> bitmap;     // 1 = free, 0 = used
        std::size_t         freeCount;
    };

    std::vector<Slab>    slabs_;
    std::vector<void*>   freeLists_[static_cast<int>(SizeClass::kCount)];

    std::size_t totalAllocated_   = 0;
    std::size_t totalDeallocated_ = 0;
};

}  // namespace quantcore
```

**BufferPool 核心逻辑** (`BufferPool.cpp`):

```
allocateRaw(bytes):
  1. bytes == 0 → return nullptr
  2. 如果 bytes > kOverflowThreshold:
        ptr = std::aligned_alloc(64, bytes)
        totalAllocated_ += bytes
        return ptr
  3. SizeClass sc = classify(bytes)
  4. 如果 freeLists_[sc] 非空 → pop 返回最后一个元素
  5. allocateSlab(sc)
  6. 从新 slab 的 freeList 中 pop 返回

deallocateRaw(ptr, bytes):
  1. ptr == nullptr → return
  2. 如果 bytes > kOverflowThreshold:
        std::free(ptr)
        totalDeallocated_ += bytes
        return
  3. SizeClass sc = classify(bytes)
  4. 通过地址范围找到 ptr 所属的 Slab
  5. 标记对应的 bitmap 位为 free
  6. 将 ptr 推入 freeLists_[sc]

classify(bytes):
  if      bytes <= 256       → kTiny
  else if bytes <= 4*1024    → kSmall
  else if bytes <= 16*1024   → kMedium
  else if bytes <= 64*1024   → kLarge
  else if bytes <= 256*1024  → kHuge
  else                       → 走 overflow 路径

blockSize(sc):
  switch(sc):
    kTiny   → 256
    kSmall  → 4*1024
    kMedium → 16*1024
    kLarge  → 64*1024
    kHuge   → 256*1024

allocateSlab(sc):
  1. 计算 numBlocks = kSlabSize / blockSize(sc)
  2. ptr = std::aligned_alloc(64, kSlabSize)
  3. 将 slab 切分为 numBlocks 个 block
  4. 将所有 block 地址推入 freeLists_[sc]
  5. 记录 Slab 元信息 {ptr, blockSize, numBlocks, bitmap, freeCount}
```

### 3.3 ExecutionEngine.h/.cpp — 求值引擎

```cpp
// ExecutionEngine.h — orchestrates expression evaluation with buffer management
// Phase: 二期必实现
//
// Provides:
//   1. evaluate(expr, md) → Column<double>  unified entry point
//   2. BufferPool lifecycle management
//   3. EngineMetrics collection
//
// The engine uses post-order traversal to evaluate the expression tree.
// Temporary buffers are allocated from the internal BufferPool.
// Phase 4 will add FusedLoopGenerator for operator fusion.
#pragma once

#include <cstddef>
#include <memory>

#include "quantcore/core/Types.h"
#include "quantcore/engine/BufferPool.h"
#include "quantcore/engine/EngineMetrics.h"
#include "quantcore/expression/ExprNode.h"
#include "quantcore/storage/Column.h"
#include "quantcore/storage/MarketData.h"

namespace quantcore {

class ExecutionEngine {
public:
    ExecutionEngine() = default;

    // ============================================================
    // Main entry point
    // ============================================================

    /// Evaluate an expression tree against MarketData.
    /// The expression tree is NOT modified — the engine traverses it
    /// read-only.  For concurrent evaluation of the same tree on
    /// different data, create multiple Engine instances.
    ///
    /// @return A Column<double> containing the result.  The output
    ///         memory is 64-byte-aligned.
    Column<double> evaluate(const ExprNode& expr, const MarketData& md);

    // ============================================================
    // Metrics
    // ============================================================

    const EngineMetrics& metrics() const noexcept { return metrics_; }
    void resetMetrics() { metrics_.reset(); }

    // ============================================================
    // Buffer pool access (for testing / inspection)
    // ============================================================

    BufferPool& pool() noexcept { return pool_; }
    const BufferPool& pool() const noexcept { return pool_; }

private:
    BufferPool    pool_;
    EngineMetrics metrics_;
};

}  // namespace quantcore
```

**ExecutionEngine 核心逻辑** (`ExecutionEngine.cpp`):

```cpp
Column<double> ExecutionEngine::evaluate(const ExprNode& expr,
                                          const MarketData& md) {
    std::size_t n = md.rowCount();
    if (n == 0) return Column<double>();

    auto t0 = std::chrono::steady_clock::now();

    // 1. Allocate result buffer from pool
    auto resultHandle = pool_.allocate<double>(n);

    // 2. Post-order evaluation — each node writes to its output buffer.
    //    Internal nodes that need temp buffers allocate from pool_
    //    via the overload evaluate(md, output, n, &pool_).
    const uint64_t* nullMask = expr.evaluate(md, resultHandle.data(), n, &pool_);

    // 3. Build Column<double> from result
    Column<double> result(n);
    std::memcpy(result.data(), resultHandle.data(), n * sizeof(double));

    // 4. Propagate null mask
    if (nullMask) {
        for (std::size_t i = 0; i < n; ++i) {
            if (detail::isNullBit(nullMask, i)) {
                result.setNull(i);
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    metrics_.recordEvaluation(static_cast<std::size_t>(usec), n, expr.nodeCount());

    return result;
}
```

### 3.4 EngineMetrics.h — 性能统计

```cpp
// EngineMetrics.h — lightweight performance counters for ExecutionEngine
// Phase: 二期必实现
#pragma once

#include <cstddef>

namespace quantcore {

class EngineMetrics {
public:
    void reset() noexcept {
        evalCount_      = 0;
        totalUsec_      = 0;
        totalRows_      = 0;
        totalNodeCount_ = 0;
        allocCount_     = 0;
        deallocCount_   = 0;
        currentBytes_   = 0;
        peakBytes_      = 0;
        totalAllocBytes_= 0;
    }

    /// Record one completed expression evaluation.
    void recordEvaluation(std::size_t usec, std::size_t rows,
                          std::size_t nodeCount) noexcept {
        ++evalCount_;
        totalUsec_ += usec;
        totalRows_ += rows;
        totalNodeCount_ += nodeCount;
    }

    /// Record a buffer allocation (called from BufferPool).
    void recordAllocation(std::size_t bytes) noexcept {
        ++allocCount_;
        currentBytes_ += bytes;
        totalAllocBytes_ += bytes;
        if (currentBytes_ > peakBytes_) peakBytes_ = currentBytes_;
    }

    /// Record a buffer deallocation (called from BufferPool).
    void recordDeallocation(std::size_t bytes) noexcept {
        ++deallocCount_;
        currentBytes_ -= bytes;
    }

    // Query interface
    std::size_t evaluationCount()  const noexcept { return evalCount_;      }
    std::size_t totalUsec()        const noexcept { return totalUsec_;      }
    std::size_t totalRows()        const noexcept { return totalRows_;      }
    std::size_t allocationCount()  const noexcept { return allocCount_;     }
    std::size_t peakBytes()        const noexcept { return peakBytes_;      }
    std::size_t avgUsecPerEval()   const noexcept {
        return evalCount_ ? totalUsec_ / evalCount_ : 0;
    }

private:
    std::size_t evalCount_       = 0;
    std::size_t totalUsec_       = 0;
    std::size_t totalRows_       = 0;
    std::size_t totalNodeCount_  = 0;
    std::size_t allocCount_      = 0;
    std::size_t deallocCount_    = 0;
    std::size_t currentBytes_    = 0;
    std::size_t peakBytes_       = 0;
    std::size_t totalAllocBytes_ = 0;
};

}  // namespace quantcore
```

---

## 四、表达式节点适配（决策 1 的实现）

### 4.1 ExprNode 接口扩展

```cpp
// ExprNode.h 中新增重载
// 原接口保持不变（向后兼容）
virtual const uint64_t* evaluate(const MarketData& md,
                                 double* output,
                                 std::size_t n) const = 0;

// 新增：接受 BufferPool*，内部临时缓冲区从池中分配
// 默认实现回退到无 pool 版本（向后兼容）
virtual const uint64_t* evaluate(const MarketData& md,
                                 double* output,
                                 std::size_t n,
                                 BufferPool* /*pool*/) const {
    return evaluate(md, output, n);
}
```

### 4.2 各节点改动细节

**BinaryExpr** — 改动最大（有 `rhsBuf_` 和 `combinedMask_`）：

```cpp
const uint64_t* evaluate(const MarketData& md, double* output,
                         std::size_t n, BufferPool* pool) const override {
    if (!pool) return evaluate(md, output, n);  // fallback

    // 1. 从 pool 分配 RHS 缓冲区
    auto rhsHandle = pool->allocate<double>(n);
    const uint64_t* rhsNull = rhs_->evaluate(md, rhsHandle.data(), n, pool);

    // 2. LHS 写入 output
    const uint64_t* lhsNull = lhs_->evaluate(md, output, n, pool);

    // 3. 合并 null mask（从 pool 分配临时 mask）
    const uint64_t* combinedNull = nullptr;
    BufferHandle<uint64_t> maskHandle;
    if (lhsNull && rhsNull) {
        std::size_t words = (n + 63) / 64;
        maskHandle = pool->allocate<uint64_t>(words);
        for (std::size_t w = 0; w < words; ++w) {
            maskHandle[w] = lhsNull[w] | rhsNull[w];
        }
        combinedNull = maskHandle.data();
    } else if (lhsNull) {
        combinedNull = lhsNull;
    } else if (rhsNull) {
        combinedNull = rhsNull;
    }

    // 4. Dispatch
    auto& reg = OperatorRegistry::instance();
    reg.invokeBinary(op_, Operand(output), Operand(rhsHandle.data()),
                     output, n, combinedNull);

    // maskHandle + rhsHandle 析构自动归还池
    // 注意：如果 combinedNull 指向 maskHandle，调用者需要在返回后立即
    // 使用结果——默认实现就是这样（结果立即被父节点消费）。
    // 如果需要跨 evaluate() 调用保持 mask，调用者应自行复制。
    return combinedNull;
}
```

**RollingExpr** / **RedExpr** / **CsExpr** — 改动模式相同：

```cpp
// 以 RollingExpr 为例
const uint64_t* evaluate(const MarketData& md, double* output,
                         std::size_t n, BufferPool* pool) const override {
    if (!pool) return evaluate(md, output, n);

    // 1. 从 pool 分配 child 缓冲区
    auto childHandle = pool->allocate<double>(n);
    const uint64_t* childNull = child_->evaluate(md, childHandle.data(), n, pool);

    // 2. 构建 ColView
    ColView<double> inputView(childHandle.data(), n, childNull);

    // 3. Dispatch
    auto& reg = OperatorRegistry::instance();
    reg.invokeRolling(op_, inputView, output, window_);

    return nullptr;
}
```

**UnaryExpr** / **ColumnRef** / **Scalar** — 无需临时缓冲区，直接用默认实现（忽略 pool）。

### 4.3 改动范围总结

| 节点 | 内部临时缓冲区 | 改动 |
|------|---------------|------|
| `ColumnRef` | 无 | 无需改动 |
| `Scalar` | 无 | 无需改动 |
| `UnaryExpr` | 无（in-place） | 无需改动 |
| `BinaryExpr` | `rhsBuf_` (double) + `combinedMask_` (uint64_t) | 新增 pool 重载，优先从池分配 |
| `RollingExpr` | `childBuf_` (double) | 新增 pool 重载，优先从池分配 |
| `RedExpr` | `childBuf_` (double) | 新增 pool 重载，优先从池分配 |
| `CsExpr` | `childBuf_` (double) | 新增 pool 重载，优先从池分配 |

保留旧的 `mutable std::vector<double>` 成员作为 pool==nullptr 时的 fallback。

---

## 五、实施顺序

```
Step 1: BufferHandle.h         (独立，无依赖)
                                  ↓
Step 2: BufferPool.h/.cpp       (依赖 BufferHandle 前向声明)
                                  ↓
Step 3: EngineMetrics.h         (独立，可并行)
                                  ↓
Step 4: ExprNode.h — 新增 evaluate() 重载  (依赖 BufferPool 前向声明)
                                  ↓
Step 5: 表达式节点适配            (依赖 Step 4)
   ├── BinaryExpr.h  — pool 优先路径
   ├── RollingExpr.h — pool 优先路径
   ├── RedExpr.h     — pool 优先路径
   └── CsExpr.h      — pool 优先路径
                                  ↓
Step 6: ExecutionEngine.h/.cpp  (依赖 Step 1-5)
                                  ↓
Step 7: test_buffer_pool.cpp + test_engine.cpp (依赖 Step 1-6)
                                  ↓
Step 8: 编译 + 全量测试验证
```

**预估代码量**：

| 文件 | 操作 | 行数 |
|------|------|------|
| `engine/BufferHandle.h` | **新建** | ~105 |
| `engine/BufferPool.h` | **新建** | ~85 |
| `engine/BufferPool.cpp` | **新建** | ~130 |
| `engine/EngineMetrics.h` | **新建** | ~70 |
| `engine/ExecutionEngine.h` | **新建** | ~55 |
| `engine/ExecutionEngine.cpp` | **新建** | ~60 |
| `expression/ExprNode.h` | 修改 — 新增 pool 重载 | +12 |
| `expression/BinaryExpr.h` | 修改 — pool 优先路径 | +35 |
| `expression/RollingExpr.h` | 修改 — pool 优先路径 | +20 |
| `expression/RedExpr.h` | 修改 — pool 优先路径 | +20 |
| `expression/CsExpr.h` | 修改 — pool 优先路径 | +20 |
| `tests/unit/test_buffer_pool.cpp` | **新建** | ~130 |
| `tests/unit/test_engine.cpp` | **新建** | ~130 |
| **合计** | | **~870 行** |

---

## 六、测试计划

### test_buffer_pool.cpp（BufferPool + BufferHandle 独立测试）

| 测试类别 | 测试用例 | 数量 |
|----------|---------|------|
| BufferHandle 基本 | 构造/析构、移动语义、release 幂等、detach 后手动管理、wrap 外部内存 | 5 |
| BufferPool 基本 | allocate→deallocate 循环、对齐验证（地址 % 64 == 0）、数据完整性（读写） | 3 |
| BufferPool 多级 | Tiny/Small/Medium/Large 各级分配、跨级分配不交叉污染 | 3 |
| BufferPool 溢出 | >256KB 分配走 aligned_alloc/free 直通路径 | 1 |
| BufferPool 压力 | 多轮 alloc/free 后 slab 复用率 | 1 |

### test_engine.cpp（ExecutionEngine + 节点 pool 适配测试）

| 测试类别 | 测试用例 | 数量 |
|----------|---------|------|
| Engine 基本 | 简单表达式求值、空输入（n=0）、单元素输入 | 3 |
| Engine 复合 | 多节点复合表达式（同 test_expression 的 Composite 测试） | 3 |
| Engine 不同数据 | clone 后的树在不同 MarketData 上求值 | 1 |
| Engine 指标 | evaluate 后 metrics 非零、reset 后归零 | 2 |
| Pool 路径 vs 旧路径 | 同一表达式分别走 pool 和 fallback 路径，结果一致 | 2 |
| 所有旧测试 | Phase 1 的 57 个测试全部通过 | 57 |

---

## 七、风险点

| 风险 | 级别 | 缓解措施 |
|------|------|---------|
| BufferHandle 移动后 `data_==nullptr` 导致父节点读取空指针 | 高 | `combinedNull` 指向 maskHandle.data() 时，确保 maskHandle 生命周期覆盖 operator 调用 |
| 查找 ptr 所属 Slab 是 O(N) | 低 | Phase 2 先用线性扫描（slab 数量通常 < 50）；后续用区间树优化 |
| `evaluate(..., pool)` 和 `evaluate(...)` 双接口维护负担 | 低 | 默认实现转发到旧版；仅在 4 个需要临时缓冲区的节点中 override |
| 所有旧测试的向后兼容 | 低 | 旧接口签名不变；新增 pool 重载带默认参数 `nullptr`（回退到 vector 路径） |

---

## 八、与后续 Phase 的关系

| Phase | 依赖 Phase 2 的什么 |
|-------|---------------------|
| Phase 3 (Lexer + Parser) | 需要 ExecutionEngine 将 AST 求值为 `Column<double>` |
| Phase 4 (FusedLoopGenerator) | 需要 BufferPool 提供对齐内存；Engine 提供融合循环的注入点 |
