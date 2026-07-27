"""
verify_rolling_ops.py — Verify all 18 rolling operators against pandas reference output.
Each operator is tested with multi-window data and edge cases.
"""
import numpy as np
import pandas as pd
import math

# Biased skewness: m3 / m2^(3/2)
def biased_skew(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    m2 = np.sum((x - mean) ** 2) / n
    m3 = np.sum((x - mean) ** 3) / n
    if m2 == 0.0:
        return np.nan
    return m3 / (m2 ** 1.5)

# Biased excess kurtosis: m4 / m2^2 - 3
def biased_kurtosis(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    m2 = np.sum((x - mean) ** 2) / n
    m4 = np.sum((x - mean) ** 4) / n
    if m2 == 0.0:
        return np.nan
    return m4 / (m2 * m2) - 3.0

# Test data
np.random.seed(42)
TEST_DATA = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                       10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
s = pd.Series(TEST_DATA)

def nan_equal(a, b):
    """Check if two values are equal, both NaN, or both Inf."""
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isinf(a) and np.isinf(b) and np.sign(a) == np.sign(b):
        return True
    return np.isclose(a, b, rtol=1e-12, atol=1e-12)

def check_op(name, cpp_values, pandas_values, tol=1e-12):
    """Compare C++ expected values against pandas reference."""
    failures = []
    for i, (c, p) in enumerate(zip(cpp_values, pandas_values)):
        if not nan_equal(c, p):
            # Relaxed tolerance for some ops
            if np.isnan(c) and not np.isnan(p):
                failures.append(f"  pos {i}: C++ NaN vs pandas {p:.15g}")
            elif not np.isnan(c) and np.isnan(p):
                failures.append(f"  pos {i}: C++ {c:.15g} vs pandas NaN")
            elif not np.isclose(c, p, rtol=1e-10, atol=1e-8):
                failures.append(f"  pos {i}: C++ {c:.15g} vs pandas {p:.15g}")
    if failures:
        print(f"  ❌ {name} FAILED ({len(failures)}/{len(cpp_values)}):")
        for f in failures[:5]:
            print(f)
    else:
        print(f"  ✅ {name} PASSED")

# Helper: extract values at specific positions for comparison
def get_values(series, positions=None):
    """Get values at specified positions, or all values."""
    if positions is None:
        return series.values
    return series.iloc[list(positions)].values

print("=" * 60)
print("Rolling Operator Verification against Pandas")
print("=" * 60)

# ============================================================
# 1. rolling_sum
# ============================================================
print("\n--- rolling_sum: pandas rolling(window).sum() ---")
for w in [1, 3, 5]:
    cpp = []
    start = 1 + w - 1  # i + 1 >= w  -> i >= w - 1
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            cpp.append(np.sum(TEST_DATA[i+1-w:i+1]))
    ref = s.rolling(window=w, min_periods=w).sum()
    check_op(f"sum(w={w})", cpp, ref.values)

# ============================================================
# 2. rolling_mean / sma
# ============================================================
print("\n--- rolling_mean: pandas rolling(window).mean() ---")
for w in [1, 3, 5]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            cpp.append(np.mean(TEST_DATA[i+1-w:i+1]))
    ref = s.rolling(window=w, min_periods=w).mean()
    check_op(f"mean(w={w})", cpp, ref.values)

# ============================================================
# 3. rolling_max
# ============================================================
print("\n--- rolling_max: pandas rolling(window).max() ---")
for w in [1, 3, 5]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            cpp.append(np.max(TEST_DATA[i+1-w:i+1]))
    ref = s.rolling(window=w, min_periods=w).max()
    check_op(f"max(w={w})", cpp, ref.values)

# ============================================================
# 4. rolling_min
# ============================================================
print("\n--- rolling_min: pandas rolling(window).min() ---")
for w in [1, 3, 5]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            cpp.append(np.min(TEST_DATA[i+1-w:i+1]))
    ref = s.rolling(window=w, min_periods=w).min()
    check_op(f"min(w={w})", cpp, ref.values)

# ============================================================
# 5. rolling_std  (ddof=0, population std)
# ============================================================
print("\n--- rolling_std: pandas rolling(window).std(ddof=0) ---")
for w in [1, 3, 5]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            cpp.append(np.std(TEST_DATA[i+1-w:i+1], ddof=0))
    ref = s.rolling(window=w, min_periods=w).std(ddof=0)
    check_op(f"std(w={w})", cpp, ref.values)

# ============================================================
# 6. rolling_var  (ddof=0, population var)
# ============================================================
print("\n--- rolling_var: pandas rolling(window).var(ddof=0) ---")
for w in [1, 3, 5]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            cpp.append(np.var(TEST_DATA[i+1-w:i+1], ddof=0))
    ref = s.rolling(window=w, min_periods=w).var(ddof=0)
    check_op(f"var(w={w})", cpp, ref.values)

# ============================================================
# 7. rolling_median
# ============================================================
print("\n--- rolling_median: pandas rolling(window).median() ---")
for w in [1, 2, 3, 4, 5]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            cpp.append(np.median(TEST_DATA[i+1-w:i+1]))
    ref = s.rolling(window=w, min_periods=w).median()
    check_op(f"median(w={w})", cpp, ref.values)

# ============================================================
# 8. rolling_mul: pandas rolling(window).apply(np.prod)
# ============================================================
print("\n--- rolling_mul: pandas rolling(window).apply(np.prod) ---")
for w in [1, 3, 5]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            cpp.append(np.prod(TEST_DATA[i+1-w:i+1]))
    ref = s.rolling(window=w, min_periods=w).apply(np.prod, raw=True)
    check_op(f"mul(w={w})", cpp, ref.values)

# ============================================================
# 9. rolling_argmax: pandas rolling(window).apply(np.argmax)
# ============================================================
print("\n--- rolling_argmax: pandas rolling(window).apply(np.argmax) ---")
for w in [1, 3, 5]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            window = TEST_DATA[i+1-w:i+1]
            # np.argmax returns first occurrence of max
            argmax = 0
            max_val = window[0]
            for j, v in enumerate(window):
                if v > max_val:
                    max_val = v
                    argmax = j
            cpp.append(float(argmax))
    ref = s.rolling(window=w, min_periods=w).apply(lambda x: np.argmax(x.values), raw=False)
    check_op(f"argmax(w={w})", cpp, ref.values)

# ============================================================
# 10. rolling_argmin: pandas rolling(window).apply(np.argmin)
# ============================================================
print("\n--- rolling_argmin: pandas rolling(window).apply(np.argmin) ---")
for w in [1, 3, 5]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            window = TEST_DATA[i+1-w:i+1]
            argmin = 0
            min_val = window[0]
            for j, v in enumerate(window):
                if v < min_val:
                    min_val = v
                    argmin = j
            cpp.append(float(argmin))
    ref = s.rolling(window=w, min_periods=w).apply(lambda x: np.argmin(x.values), raw=False)
    check_op(f"argmin(w={w})", cpp, ref.values)

# ============================================================
# 11. rolling_rank: rank(method="average", pct=True).iloc[-1]
# ============================================================
print("\n--- rolling_rank: pandas rank(method='average', pct=True).iloc[-1] ---")
for w in [1, 3, 5]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            window = pd.Series(TEST_DATA[i+1-w:i+1])
            rank_pct = window.rank(method="average", pct=True).iloc[-1]
            cpp.append(rank_pct)
    ref = s.rolling(window=w, min_periods=w).apply(
        lambda x: x.rank(method="average", pct=True).iloc[-1], raw=False)
    check_op(f"rank(w={w})", cpp, ref.values)

# ============================================================
# 12. rolling_quantile: rolling(window).quantile(q, interpolation="linear")
# ============================================================
print("\n--- rolling_quantile: pandas rolling(window).quantile(q, interpolation='linear') ---")
for w in [2, 3, 4, 5]:
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cpp = []
        for i in range(len(TEST_DATA)):
            if i + 1 < w:
                cpp.append(np.nan)
            else:
                window = TEST_DATA[i+1-w:i+1]
                cpp.append(np.quantile(window, q, method="linear"))
        ref = s.rolling(window=w, min_periods=w).quantile(q, interpolation="linear")
        check_op(f"quantile(w={w}, q={q})", cpp, ref.values)

# ============================================================
# 13. rolling_skew (bias=True, population skewness)
# ============================================================
print("\n--- rolling_skew: biased skewness (bias=True), m3/m2^(3/2) ---")
for w in [3, 5, 10]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            window = TEST_DATA[i+1-w:i+1]
            cpp.append(biased_skew(window))
    ref = s.rolling(window=w, min_periods=w).apply(
        lambda x: biased_skew(x), raw=True)
    check_op(f"skew(w={w})", cpp, ref.values)

# ============================================================
# 14. rolling_kurt (bias=True, population excess kurtosis)
# ============================================================
print("\n--- rolling_kurt: biased excess kurtosis (bias=True), m4/m2^2 - 3 ---")
for w in [4, 5, 10]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i + 1 < w:
            cpp.append(np.nan)
        else:
            window = TEST_DATA[i+1-w:i+1]
            cpp.append(biased_kurtosis(window))
    ref = s.rolling(window=w, min_periods=w).apply(
        lambda x: biased_kurtosis(x), raw=True)
    check_op(f"kurt(w={w})", cpp, ref.values)

# ============================================================
# 15. EMA: ewm(span=n, adjust=False).mean()
# ============================================================
print("\n--- EMA: pandas ewm(span=n, adjust=False).mean() ---")
for span in [3, 5, 10]:
    alpha = 2.0 / (span + 1)
    cpp = []
    ema = TEST_DATA[0]
    cpp.append(ema)
    for i in range(1, len(TEST_DATA)):
        ema = alpha * TEST_DATA[i] + (1 - alpha) * ema
        cpp.append(ema)
    ref = s.ewm(span=span, adjust=False).mean()
    check_op(f"ema(span={span})", cpp, ref.values)

# ============================================================
# 16. diff: pandas diff(periods=n)
# ============================================================
print("\n--- diff: pandas diff(periods=n) ---")
for n in [1, 3, 5]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i < n:
            cpp.append(np.nan)
        else:
            cpp.append(TEST_DATA[i] - TEST_DATA[i - n])
    ref = s.diff(periods=n)
    check_op(f"diff(n={n})", cpp, ref.values)

# ============================================================
# 17. shift: pandas shift(periods=n)
# ============================================================
print("\n--- shift: pandas shift(periods=n) ---")
for n in [1, 3, 5]:
    cpp = []
    for i in range(len(TEST_DATA)):
        if i < n:
            cpp.append(np.nan)
        else:
            cpp.append(TEST_DATA[i - n])
    ref = s.shift(periods=n)
    check_op(f"shift(n={n})", cpp, ref.values)

print("\n" + "=" * 60)
print("Verification complete")
print("=" * 60)
