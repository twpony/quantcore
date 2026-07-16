// TimestampIndex.cpp — time-axis index implementation
// Phase: 一期必实现
#include "TimestampIndex.h"

#include <algorithm>
#include <ctime>
#include <stdexcept>

namespace quantcore {

// ============================================================
// Unix timestamp → YYYYMMDD conversion
// ============================================================

int64_t TimestampIndex::unixToYYYYMMDD(int64_t unixSeconds) {
    // Use standard library gmtime_r for UTC-based date extraction.
    // For local exchange time, this should be adjusted upstream.
    std::time_t t = static_cast<std::time_t>(unixSeconds);
    std::tm tm_buf{};
    if (!gmtime_r(&t, &tm_buf)) {
        return 0;  // Invalid timestamp
    }
    return static_cast<int64_t>(tm_buf.tm_year + 1900) * 10000 +
           static_cast<int64_t>(tm_buf.tm_mon + 1) * 100 +
           static_cast<int64_t>(tm_buf.tm_mday);
}

// ============================================================
// Weekday detection from YYYYMMDD
// ============================================================

bool TimestampIndex::isWeekday(int64_t yyyymmdd) {
    // Convert YYYYMMDD to a tm struct to get day-of-week via mktime.
    int year  = static_cast<int>(yyyymmdd / 10000);
    int month = static_cast<int>((yyyymmdd / 100) % 100);
    int day   = static_cast<int>(yyyymmdd % 100);

    if (month < 1 || month > 12 || day < 1 || day > 31) return false;

    std::tm tm_buf{};
    tm_buf.tm_year = year - 1900;
    tm_buf.tm_mon  = month - 1;
    tm_buf.tm_mday = day;
    tm_buf.tm_isdst = -1;  // Let mktime determine DST

    // mktime normalizes the tm struct and fills in tm_wday
    std::time_t _ = std::mktime(&tm_buf);
    (void)_;  // We only care about the normalized tm_wday

    // tm_wday: 0=Sun, 1=Mon, ..., 6=Sat
    return tm_buf.tm_wday >= 1 && tm_buf.tm_wday <= 5;
}

// ============================================================
// Construction
// ============================================================

TimestampIndex::TimestampIndex(const int64_t* timestamps, std::size_t size)
    : timestamps_(timestamps, size)
    , dates_(size)
{
    for (std::size_t i = 0; i < size; ++i) {
        dates_[i] = unixToYYYYMMDD(timestamps[i]);
    }
}

TimestampIndex::TimestampIndex(const Column<int64_t>& timestamps)
    : timestamps_(timestamps)
    , dates_(timestamps.size())
{
    for (std::size_t i = 0; i < timestamps.size(); ++i) {
        dates_[i] = unixToYYYYMMDD(timestamps[i]);
    }
}

// ============================================================
// Basic access
// ============================================================

std::size_t TimestampIndex::size() const noexcept {
    return timestamps_.size();
}

int64_t TimestampIndex::operator[](std::size_t i) const {
    return timestamps_[i];
}

const Column<int64_t>& TimestampIndex::timestamps() const noexcept {
    return timestamps_;
}

const Column<int64_t>& TimestampIndex::dates() const noexcept {
    return dates_;
}

int64_t TimestampIndex::dateAt(std::size_t i) const {
    return dates_[i];
}

// ============================================================
// Date-aware queries
// ============================================================

bool TimestampIndex::isConsecutive(std::size_t i) const {
    if (i == 0) return false;
    int64_t d1 = dates_[i - 1];
    int64_t d2 = dates_[i];

    // Same day → not consecutive (different bars within same day?)
    if (d1 == d2) return false;

    // Walk forward from d1 + 1 day, skipping weekends, to see if d2 is
    // the very next trading day.
    // Strategy: count the number of weekdays strictly between d1 and d2.
    // If it's 0, they are consecutive trading days (ignoring holidays,
    // which are a future extension).

    // Fast path: if d2 == next calendar day and d2 is a weekday and d1 is
    // also a weekday, they're consecutive.
    // But we need to handle Friday→Monday correctly: Friday is weekday,
    // Monday is weekday, Saturday/Sunday are between them.  So the fast
    // path only works for non-Friday weekdays.

    // Simpler approach: convert both to time_t, iterate day by day,
    // counting weekdays between them.

    // Actually the simplest correct check: count the weekdays in
    // [d1+1day, d2-1day].  If count == 0, they are consecutive.
    // This handles Friday→Monday automatically because Saturday and
    // Sunday are not weekdays, so d1=Friday, d2=Monday → between them
    // there are Sat/Sun (non-weekdays) → count == 0 → consecutive.

    // Convert to days-since-epoch for simple iteration.
    // We'll use a minimal approach: iterate through YYYYMMDD dates.

    // Break d1 into components
    int y1 = static_cast<int>(d1 / 10000);
    int m1 = static_cast<int>((d1 / 100) % 100);
    int d_1 = static_cast<int>(d1 % 100);

    int y2 = static_cast<int>(d2 / 10000);
    int m2 = static_cast<int>((d2 / 100) % 100);
    int d_2 = static_cast<int>(d2 % 100);

    // Create tm and advance day by day
    std::tm tm_buf{};
    tm_buf.tm_year = y1 - 1900;
    tm_buf.tm_mon  = m1 - 1;
    tm_buf.tm_mday = d_1;
    tm_buf.tm_isdst = -1;

    std::time_t t1 = std::mktime(&tm_buf);
    constexpr std::time_t kSecPerDay = 86400;
    std::time_t t2 = t1;

    // Advance one day at a time from d1 toward d2
    int weekdayCount = 0;
    while (true) {
        t2 += kSecPerDay;
        std::tm* next_tm = std::localtime(&t2);
        int next_y  = next_tm->tm_year + 1900;
        int next_m  = next_tm->tm_mon + 1;
        int next_d  = next_tm->tm_mday;
        int64_t next_date = static_cast<int64_t>(next_y) * 10000 +
                            static_cast<int64_t>(next_m) * 100 +
                            static_cast<int64_t>(next_d);

        if (next_date >= d2) break;

        int wday = next_tm->tm_wday;  // 0=Sun, 6=Sat
        if (wday >= 1 && wday <= 5) {
            ++weekdayCount;
        }
    }

    return weekdayCount == 0;
}

int64_t TimestampIndex::tradingDaysBetween(std::size_t from,
                                             std::size_t to) const {
    if (from >= to) return 0;
    if (to > size()) return 0;

    int64_t count = 0;
    for (std::size_t i = from; i < to; ++i) {
        if (isWeekday(dates_[i])) {
            ++count;
        }
    }
    return count;
}

std::pair<std::size_t, std::size_t>
TimestampIndex::dateRange(int64_t startDate, int64_t endDate) const {
    std::size_t start = 0;
    std::size_t end   = 0;

    // Find first row with date >= startDate
    for (start = 0; start < dates_.size(); ++start) {
        if (dates_[start] >= startDate) break;
    }

    if (start >= dates_.size()) return {0, 0};

    // Find first row with date > endDate (or end of data)
    for (end = start; end < dates_.size(); ++end) {
        if (dates_[end] > endDate) break;
    }

    return {start, end};
}

// ============================================================
// Binary search helpers
// ============================================================

std::size_t TimestampIndex::lowerBound(int64_t ts) const {
    auto it = std::lower_bound(timestamps_.cbegin(), timestamps_.cend(), ts);
    return static_cast<std::size_t>(std::distance(timestamps_.cbegin(), it));
}

std::size_t TimestampIndex::upperBound(int64_t ts) const {
    auto it = std::upper_bound(timestamps_.cbegin(), timestamps_.cend(), ts);
    return static_cast<std::size_t>(std::distance(timestamps_.cbegin(), it));
}

}  // namespace quantcore
