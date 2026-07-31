// CsvLoader.h — bulk factor registration from CSV files
//
// CSV format (3 columns, with header):
//   name,expression,description
//
// The loader reads columns 1 (name) and 2 (expression); column 3
// (description) is stored but not used for registration.
//
// Fields containing commas MUST be enclosed in double quotes:
//   ma5,"rolling_mean(close, 5)",5-day moving average
//
// Lines starting with '#' are treated as comments and skipped.
//
// Usage:
//   #include "factors/CsvLoader.h"
//   FactorCalculator calc;
//   loadFactorsFromCsv(calc, "factors/example_factors.csv");
//   auto result = calc.evaluate("alpha_0001", marketData);
#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "quantcore/core/FactorCalculator.h"

namespace quantcore {
namespace factors {

// ============================================================
// Data types
// ============================================================

/// A single record parsed from a factor CSV row.
struct CsvFactorRecord {
    std::string name;         // column 1 — factor name
    std::string expression;   // column 2 — formula expression
    std::string description;  // column 3 — human-readable description
};

// ============================================================
// Internal: CSV field splitting with quote support
// ============================================================

namespace detail {

/// Split a single CSV line into fields, respecting double-quote escaping.
/// Fields not enclosed in quotes are trimmed of surrounding whitespace.
/// Quoted fields have their quotes removed and internal commas preserved.
inline std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t i = 0;
    std::size_t n = line.size();

    while (i < n) {
        // Skip leading whitespace before a field
        while (i < n && (line[i] == ' ' || line[i] == '\t'))
            ++i;

        if (i >= n) break;

        std::string field;

        if (line[i] == '"') {
            // Quoted field — read until closing quote
            ++i;  // skip opening quote
            while (i < n) {
                if (line[i] == '"') {
                    if (i + 1 < n && line[i + 1] == '"') {
                        // Escaped quote: "" → "
                        field += '"';
                        i += 2;
                    } else {
                        // Closing quote
                        ++i;  // skip closing quote
                        break;
                    }
                } else {
                    field += line[i];
                    ++i;
                }
            }
        } else {
            // Unquoted field — read until comma or end
            while (i < n && line[i] != ',') {
                field += line[i];
                ++i;
            }
            // Trim trailing whitespace from unquoted field
            std::size_t a = field.find_first_not_of(" \t\r");
            std::size_t b = field.find_last_not_of(" \t\r");
            if (a == std::string::npos) {
                field.clear();
            } else {
                field = field.substr(a, b - a + 1);
            }
        }

        fields.push_back(std::move(field));

        // Skip comma separator
        if (i < n && line[i] == ',') ++i;
    }

    return fields;
}

}  // namespace detail

// ============================================================
// CSV parsing
// ============================================================

/// Parse a factor CSV file.
///
/// Expected header:  name,expression,description
/// Each data row must have at least 2 columns (name, expression);
/// the third column (description) is optional.
///
/// Fields containing commas (e.g. "rolling_mean(close, 5)") MUST be
/// enclosed in double quotes.
///
/// Lines whose first non-whitespace character is '#' are comments.
/// Empty lines are skipped.
///
/// @throws std::runtime_error if the file cannot be opened.
inline std::vector<CsvFactorRecord> parseFactorCsv(
        const std::string& csvPath) {
    std::vector<CsvFactorRecord> records;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        throw std::runtime_error("CsvLoader: cannot open file '" +
                                 csvPath + "'");
    }

    std::string line;

    // Skip header line
    if (!std::getline(file, line)) {
        return records;  // empty file
    }

    while (std::getline(file, line)) {
        // Trim leading whitespace
        std::size_t start = line.find_first_not_of(" \t\r");
        if (start == std::string::npos) continue;   // blank line
        if (line[start] == '#') continue;            // comment

        auto fields = detail::splitCsvLine(line.substr(start));

        if (fields.size() < 2) continue;       // need at least name + expr
        if (fields[0].empty() || fields[1].empty()) continue;

        std::string desc = (fields.size() >= 3) ? std::move(fields[2])
                                                 : std::string{};

        records.push_back({std::move(fields[0]), std::move(fields[1]),
                           std::move(desc)});
    }

    return records;
}

// ============================================================
// Batch registration
// ============================================================

/// Load all factors from a CSV file and register them with @p calc.
///
/// This is the primary entry point.  It parses the CSV and calls
/// calc.registerFormula() for every row.
///
/// @returns the number of factors registered.
/// @throws std::runtime_error if the file cannot be opened or if
///         any expression fails to parse.
inline std::size_t loadFactorsFromCsv(FactorCalculator& calc,
                                       const std::string& csvPath) {
    auto records = parseFactorCsv(csvPath);
    for (auto& r : records) {
        calc.registerFormula(r.name, r.expression);
    }
    return records.size();
}

}  // namespace factors
}  // namespace quantcore
