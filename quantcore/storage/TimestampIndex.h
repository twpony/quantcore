// TimestampIndex.h — time-axis index with date-aware queries
// Phase: 一期必实现
//
// TimestampIndex stores a sequence of Unix timestamps (seconds since epoch)
// together with their YYYYMMDD date representation for fast date-range lookups.
//
// Trading-day logic: without a full holiday calendar, weekdays (Mon-Fri) are
// treated as trading days.  A future extension point (holidayCalendar) is
// reserved so that a proper exchange calendar can be plugged in.
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "quantcore/storage/Column.h"

namespace quantcore {

class TimestampIndex {
public:
    // ============================================================
    // Construction
    // ============================================================

    TimestampIndex() = default;

    /// Build the index from a contiguous array of Unix timestamps.
    /// Dates (YYYYMMDD) are derived automatically.
    explicit TimestampIndex(const int64_t* timestamps, std::size_t size);

    /// Build from a Column<int64_t> of timestamps.
    explicit TimestampIndex(const Column<int64_t>& timestamps);

    // ============================================================
    // Basic access
    // ============================================================

    std::size_t size() const noexcept;

    /// Unix timestamp at row i.
    int64_t operator[](std::size_t i) const;

    /// Raw timestamps column.
    const Column<int64_t>& timestamps() const noexcept;

    /// Dates column (YYYYMMDD int64_t).
    const Column<int64_t>& dates() const noexcept;

    // ============================================================
    // Date-aware queries
    // ============================================================

    /// Date at row i in YYYYMMDD format.
    int64_t dateAt(std::size_t i) const;

    /// True when row i and row i-1 are on consecutive trading days.
    /// A trading day is defined as Mon-Fri (excludes weekends).
    /// Row 0 always returns false.
    bool isConsecutive(std::size_t i) const;

    /// Number of trading days (Mon-Fri) between rows `from` and `to`
    /// (inclusive of `from`, exclusive of `to`).
    int64_t tradingDaysBetween(std::size_t from, std::size_t to) const;

    /// Find the [start, end) row range for a date interval.
    /// Returns {0, 0} if no rows fall within [startDate, endDate].
    std::pair<std::size_t, std::size_t> dateRange(int64_t startDate,
                                                   int64_t endDate) const;

    // ============================================================
    // Lookup
    // ============================================================

    /// Find the row index for the first timestamp >= `ts`.
    /// Returns size() if all timestamps are < ts.
    std::size_t lowerBound(int64_t ts) const;

    /// Find the row index for the first timestamp > `ts`.
    std::size_t upperBound(int64_t ts) const;

    // ============================================================
    // Holiday calendar extension point (远期)
    // ============================================================

    // Future: void setHolidayCalendar(const HolidayCalendar& cal);
    // Future: bool isTradingDay(int64_t date) const;

private:
    /// Derive YYYYMMDD date from a Unix timestamp.
    static int64_t unixToYYYYMMDD(int64_t unixSeconds);

    /// Check if a YYYYMMDD date falls on a weekday (Mon=1 .. Fri=5).
    static bool isWeekday(int64_t yyyymmdd);

    Column<int64_t> timestamps_;
    Column<int64_t> dates_;
};

}  // namespace quantcore
