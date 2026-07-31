// EngineMetrics.h — lightweight performance counters for ExecutionEngine
// Phase: 二期必实现
//
// Tracks:
//   - Evaluation count and aggregate latency
//   - Rows processed and node count
//   - Buffer allocation / deallocation counts and current/peak memory
//
// All counters are monotonic except currentBytes_.  Use reset() to zero
// everything between benchmarks.
#pragma once

#include <cstddef>

namespace quantcore {

class EngineMetrics {
public:
    // ============================================================
    // Reset
    // ============================================================

    void reset() noexcept {
        evalCount_       = 0;
        totalUsec_       = 0;
        totalRows_       = 0;
        totalNodeCount_  = 0;
        allocCount_      = 0;
        deallocCount_    = 0;
        currentBytes_    = 0;
        peakBytes_       = 0;
        totalAllocBytes_ = 0;
    }

    // ============================================================
    // Record events (called by ExecutionEngine / BufferPool)
    // ============================================================

    /// Record one completed expression evaluation.
    void recordEvaluation(std::size_t usec, std::size_t rows,
                          std::size_t nodeCount) noexcept {
        ++evalCount_;
        totalUsec_      += usec;
        totalRows_      += rows;
        totalNodeCount_ += nodeCount;
    }

    /// Record a buffer allocation.
    void recordAllocation(std::size_t bytes) noexcept {
        ++allocCount_;
        currentBytes_     += bytes;
        totalAllocBytes_  += bytes;
        if (currentBytes_ > peakBytes_) {
            peakBytes_ = currentBytes_;
        }
    }

    /// Record a buffer deallocation.
    void recordDeallocation(std::size_t bytes) noexcept {
        ++deallocCount_;
        if (currentBytes_ >= bytes) {
            currentBytes_ -= bytes;
        } else {
            currentBytes_ = 0;  // defensive: shouldn't happen
        }
    }

    // ============================================================
    // Query interface
    // ============================================================

    std::size_t evaluationCount()  const noexcept { return evalCount_;       }
    std::size_t totalUsec()        const noexcept { return totalUsec_;       }
    std::size_t totalRows()        const noexcept { return totalRows_;       }
    std::size_t totalNodeCount()   const noexcept { return totalNodeCount_;  }
    std::size_t allocationCount()  const noexcept { return allocCount_;      }
    std::size_t deallocationCount() const noexcept { return deallocCount_;   }
    std::size_t peakBytes()        const noexcept { return peakBytes_;       }
    std::size_t currentBytes()     const noexcept { return currentBytes_;    }

    /// Average microseconds per evaluation, or 0 if no evaluations.
    std::size_t avgUsecPerEval() const noexcept {
        return evalCount_ ? totalUsec_ / evalCount_ : 0;
    }

    /// Average rows per evaluation.
    std::size_t avgRowsPerEval() const noexcept {
        return evalCount_ ? totalRows_ / evalCount_ : 0;
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
