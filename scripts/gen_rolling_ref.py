"""
Generate reference values for C++ cross-validation.
Outputs C++ array initializer code for each operator.
"""
import numpy as np
import pandas as pd
import math

TEST_DATA = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                       10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
N = len(TEST_DATA)

def biased_skew(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    m2 = np.sum((x - mean) ** 2) / n
    m3 = np.sum((x - mean) ** 3) / n
    if m2 == 0.0: return np.nan
    return m3 / (m2 ** 1.5)

def biased_kurtosis(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    m2 = np.sum((x - mean) ** 2) / n
    m4 = np.sum((x - mean) ** 4) / n
    if m2 == 0.0: return np.nan
    return m4 / (m2 * m2) - 3.0

def fmt(x):
    """Format a float as C++ literal or NAN."""
    if np.isnan(x):
        return "NAN"
    return f"{x:.15g}"

def emit_rolling(name, w, values):
    print(f"// {name}(window={w})")
    print(f"static const double k{name.capitalize()}W{w}[] = {{", end="")
    print(", ".join(fmt(v) for v in values), end="")
    print("};")
    print()

# Generate reference for each operator with specific windows
print("// Auto-generated reference values for C++ rolling operator verification")
print("// Test data: 1..10, 10..1 (20 elements)")
print(f"static constexpr std::size_t kN = {N};")
print(f"static const double kInput[] = {{", end="")
print(", ".join(f"{x:.15g}" for x in TEST_DATA), end="")
print("};")
print()

# rolling_sum
emit_rolling("sum", 3, [np.nan if i+1 < 3 else np.sum(TEST_DATA[i+1-3:i+1]) for i in range(N)])
emit_rolling("sum", 5, [np.nan if i+1 < 5 else np.sum(TEST_DATA[i+1-5:i+1]) for i in range(N)])

# rolling_mean
emit_rolling("mean", 3, [np.nan if i+1 < 3 else np.mean(TEST_DATA[i+1-3:i+1]) for i in range(N)])

# rolling_var
emit_rolling("var", 3, [np.nan if i+1 < 3 else np.var(TEST_DATA[i+1-3:i+1], ddof=0) for i in range(N)])

# rolling_std
emit_rolling("std", 3, [np.nan if i+1 < 3 else np.std(TEST_DATA[i+1-3:i+1], ddof=0) for i in range(N)])

# rolling_max
emit_rolling("max", 3, [np.nan if i+1 < 3 else np.max(TEST_DATA[i+1-3:i+1]) for i in range(N)])

# rolling_min
emit_rolling("min", 3, [np.nan if i+1 < 3 else np.min(TEST_DATA[i+1-3:i+1]) for i in range(N)])

# rolling_median
emit_rolling("median", 3, [np.nan if i+1 < 3 else np.median(TEST_DATA[i+1-3:i+1]) for i in range(N)])
emit_rolling("median", 4, [np.nan if i+1 < 4 else np.median(TEST_DATA[i+1-4:i+1]) for i in range(N)])

# rolling_mul
emit_rolling("mul", 3, [np.nan if i+1 < 3 else np.prod(TEST_DATA[i+1-3:i+1]) for i in range(N)])

# rolling_argmax
def argmax_first(arr):
    m = arr[0]; pos = 0
    for j, v in enumerate(arr):
        if v > m: m = v; pos = j
    return float(pos)
emit_rolling("argmax", 3, [np.nan if i+1 < 3 else argmax_first(TEST_DATA[i+1-3:i+1]) for i in range(N)])

# rolling_argmin
def argmin_first(arr):
    m = arr[0]; pos = 0
    for j, v in enumerate(arr):
        if v < m: m = v; pos = j
    return float(pos)
emit_rolling("argmin", 3, [np.nan if i+1 < 3 else argmin_first(TEST_DATA[i+1-3:i+1]) for i in range(N)])

# rolling_skew
emit_rolling("skew", 5, [np.nan if i+1 < 5 else biased_skew(TEST_DATA[i+1-5:i+1]) for i in range(N)])

# rolling_kurt
emit_rolling("kurt", 5, [np.nan if i+1 < 5 else biased_kurtosis(TEST_DATA[i+1-5:i+1]) for i in range(N)])

# rolling_rank
def rank_pct(arr):
    s = pd.Series(arr)
    return s.rank(method="average", pct=True).iloc[-1]
emit_rolling("rank", 3, [np.nan if i+1 < 3 else rank_pct(TEST_DATA[i+1-3:i+1]) for i in range(N)])

# rolling_quantile
for q in [0.25, 0.5, 0.75]:
    emit_rolling(f"quantile_q{int(q*100)}", 4,
        [np.nan if i+1 < 4 else np.quantile(TEST_DATA[i+1-4:i+1], q, method="linear") for i in range(N)])

# ema (span=5, i+1 < window → NaN for first window-1 positions)
alpha = 2.0 / (5 + 1)
ema = TEST_DATA[0]
ema_vals = [np.nan]  # i=0: i+1 < 5
for i in range(1, N):
    ema = alpha * TEST_DATA[i] + (1 - alpha) * ema
    if i + 1 < 5:
        ema_vals.append(np.nan)
    else:
        ema_vals.append(ema)
emit_rolling("rolling_ema", 5, ema_vals)

# sma
emit_rolling("rolling_sma", 3, [np.nan if i+1 < 3 else np.mean(TEST_DATA[i+1-3:i+1]) for i in range(N)])

# diff
emit_rolling("rolling_diff", 3, [np.nan if i < 3 else TEST_DATA[i] - TEST_DATA[i-3] for i in range(N)])

# shift
emit_rolling("rolling_shift", 3, [np.nan if i < 3 else TEST_DATA[i-3] for i in range(N)])
