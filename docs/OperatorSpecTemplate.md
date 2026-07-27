# 算子规范模版

> 使用方式：为每个算子复制一份下方模版，填写后即可按 `docs/HowToAddOperator.md` 中的步骤实现。

---

## 模版

```markdown
## [算子名称] (英文简称)

- **分类**: Unary / Binary
- **枚举**: `UnaryOpCode::XXX` / `BinaryOpCode::XXX`
- **数学形式**: 用 LaTeX 或伪代码描述

  ```
  result[i] = f(input[i])
  result[i] = f(lhs[i], rhs[i])
  ```

- **输入类型**: `ColView<double>` / `ColView<double> + double`
- **输出类型**: `Column<double>`
- **空值传播**:

  | 条件 | 行为 |
  |------|------|
  | 输入位置 i 为 null | 输出位置 i 标记为 null |
  | （二元）lhs[i] 或 rhs[i] 任一为 null | 输出位置 i 标记为 null |
  | 其他边界条件 | 描述 |

- **定义域/值域约束**:

  | 输入约束 | 处理方式 |
  |---------|---------|
  | x <= 0 时 log(x) | 输出 NaN，写 QC_WARN 日志 |
  | 除零 | 输出 NaN |
  | 溢出 | 输出 ±Inf |

- **首元素/越界处理** (仅 Diff / Shift 类算子需要):

  | 位置 | 行为 |
  |------|------|
  | i=0 (Diff) | 输出 null |
  | i < offset (Shift) | 输出 null |

- **数值稳定性注意事项**: 有无特殊防溢出/防精度损失的实现技巧
- **SIMD 优化要点**: 有无可利用的 SIMD 指令 (如 FMA)，有无需要特殊处理的分支
- **对标参考**: 与哪些常见库的计算结果需要一致 (如 numpy/pandas/Talib)
- **测试用例**:

  | 用例 | 输入 | 期望输出 |
  |------|------|---------|
  | 正常值 | ... | ... |
  | 零值 | 0 | 0 |
  | 负值 | -1 | NaN (或正常) |
  | 空值 | null | null |
  | 极大值 | 1e308 | Inf |
  | 极小值 | -1e308 | -Inf |
  | NaN | NaN | NaN |
  | 混合 null+正常 | [1.0, null, 3.0] | [f(1.0), null, f(3.0)] |
```

---

## 填写示例

以下是一个已填写好的一元算子示例，供参考格式。

```markdown
## Abs — 绝对值

- **分类**: Unary
- **枚举**: `UnaryOpCode::ABS`
- **数学形式**:

  ```
  result[i] = |input[i]|
  ```

- **输入类型**: `ColView<double>`
- **输出类型**: `Column<double>`
- **空值传播**:

  | 条件 | 行为 |
  |------|------|
  | 输入位置 i 为 null | 输出位置 i 标记为 null |

- **定义域/值域约束**:

  | 输入约束 | 处理方式 |
  |---------|---------|
  | 无特殊约束 | 直接计算，std::abs 对 NaN/Inf 行为已定义 |

- **首元素/越界处理**: 不适用
- **数值稳定性注意事项**: 直接调用 `std::abs`，无可担心的精度问题
- **SIMD 优化要点**: 可用 `_mm256_andnot_pd` (AVX2) 清除符号位，无需分支
- **对标参考**: 与 `numpy.abs`、`pandas.Series.abs()` 结果一致
- **测试用例**:

  | 用例 | 输入 | 期望输出 |
  |------|------|---------|
  | 正数 | 3.14 | 3.14 |
  | 负数 | -5.0 | 5.0 |
  | 零 | 0.0 | 0.0 |
  | 负零 | -0.0 | 0.0 |
  | 空值 | null | null |
  | +Inf | +∞ | +∞ |
  | -Inf | -∞ | +∞ |
  | NaN | NaN | NaN |
  | 混合 | [1.0, null, -3.0] | [1.0, null, 3.0] |
```

---

## 一期待实现算子清单

以下是一期需要实现的全部算子，可逐个按模版填写规范。已填写示例的打了 ✓。

### 一元算子 (Unary)

| 状态 | 算子 | 枚举 | 文件 |
|------|------|------|------|
| ✓ 示例 | Abs | `UnaryOpCode::ABS` | `operators/unary/Abs.h` |
| ✗ | Log | `UnaryOpCode::LOG` | `operators/unary/Log.h` |
| ✗ | Log10 | `UnaryOpCode::LOG10` | `operators/unary/Log10.h` |
| ✗ | Sqrt | `UnaryOpCode::SQRT` | `operators/unary/Sqrt.h` |
| ✗ | Neg | `UnaryOpCode::NEG` | `operators/unary/Neg.h` |
| ✗ | Diff | `RollingOpCode::DIFF` | `operators/rolling/Diff.h` |
| ✗ | Shift | `RollingOpCode::SHIFT` | `operators/rolling/Shift.h` |
| ✗ | Sign | `UnaryOpCode::SIGN` | `operators/unary/Sign.h` |
| ✗ | Square | `UnaryOpCode::SQUARE` | `operators/unary/Square.h` |
| ✗ | Exp | `UnaryOpCode::EXP` | `operators/unary/Exp.h` |

### 二元算子 (Binary)

| 状态 | 算子 | 枚举 | 文件 |
|------|------|------|------|
| ✗ | Add | `BinaryOpCode::ADD` | `operators/binary/Add.h` |
| ✗ | Sub | `BinaryOpCode::SUB` | `operators/binary/Sub.h` |
| ✗ | Mul | `BinaryOpCode::MUL` | `operators/binary/Mul.h` |
| ✗ | Div | `BinaryOpCode::DIV` | `operators/binary/Div.h` |
| ✗ | Pow | `BinaryOpCode::POW` | `operators/binary/Pow.h` |
| ✗ | Max | `BinaryOpCode::MAX` | `operators/binary/Max.h` |
| ✗ | Min | `BinaryOpCode::MIN` | `operators/binary/Min.h` |
| ✗ | Gt | `BinaryOpCode::GT` | `operators/binary/Cmp.h` |
| ✗ | Lt | `BinaryOpCode::LT` | `operators/binary/Cmp.h` |
| ✗ | Eq | `BinaryOpCode::EQ` | `operators/binary/Cmp.h` |
| ✗ | Neq | `BinaryOpCode::NEQ` | `operators/binary/Cmp.h` |

### 滚动窗口算子 (Rolling) — 远期

| 状态 | 算子 | 枚举 | 窗口参数 | 文件 |
|------|------|------|---------|------|
| ✗ | Sma | `RollingOpCode::SMA` | window | `operators/rolling/Sma.h` |
| ✗ | Ema | `RollingOpCode::EMA` | window (alpha = 2/(window+1)) | `operators/rolling/Ema.h` |
| ✗ | RollingMax | `RollingOpCode::ROLLING_MAX` | window | `operators/rolling/RollingMax.h` |
| ✗ | RollingMin | `RollingOpCode::ROLLING_MIN` | window | `operators/rolling/RollingMin.h` |
| ✗ | RollingStd | `RollingOpCode::ROLLING_STD` | window | `operators/rolling/RollingStd.h` |
| ✗ | RollingSum | `RollingOpCode::ROLLING_SUM` | window | `operators/rolling/RollingSum.h` |
| ✗ | RollingRank | `RollingOpCode::ROLLING_RANK` | window | `operators/rolling/RollingRank.h` |

### 横截面算子 (CrossSection) — 远期

| 状态 | 算子 | 枚举 | 参数 | 文件 |
|------|------|------|------|------|
| ✗ | Rank | `CrossSectionOpCode::RANK` | — | `operators/cross_section/Rank.h` |
| ✗ | ZScore | `CrossSectionOpCode::ZSCORE` | — | `operators/cross_section/ZScore.h` |
| ✗ | Quantile | `CrossSectionOpCode::QUANTILE` | numBuckets | `operators/cross_section/Quantile.h` |

---

## Rolling 算子规范模版

```markdown
## [算子名称] (英文简称)

- **分类**: Rolling
- **枚举**: `RollingOpCode::XXX`
- **窗口参数**: window (运行时参数, size_t, >= 1)
- **数学形式**: 用 LaTeX 或伪代码描述

  ```
  result[i] = statistic(input[i-window+1 ... i]),  for i >= window-1
  result[i] = null,                                   for i < window-1
  ```

- **输入类型**: `ColView<double>` + `window` (size_t)
- **输出类型**: `Column<double>`
- **前 window-1 位置**: 标记为 null
- **min_periods**: 窗口内最少有效观测数 (默认 = window)
- **空值处理**:

  | 条件 | 行为 |
  |------|------|
  | i < window-1 | 输出位置 i 标记为 null |
  | 窗口内有效观测数 < min_periods | 输出位置 i 标记为 null |
  | 窗口内 null 值 | 不计入有效观测，不传播 |

- **数值稳定性注意事项**: 如使用 online 算法(Welford)、前缀和差分等
- **时间复杂度**: 标量参考 O(n*window)，远期优化目标 (如 O(n))
- **测试用例**:

  | 用例 | 输入 | window | 期望输出 |
  |------|------|--------|---------|
  | 正常值 | [1.0, 2.0, 3.0, 4.0, 5.0] (SMA) | 3 | [null, null, 2.0, 3.0, 4.0] |
  | 含空值 | [1.0, null, 3.0, 4.0, 5.0] (SMA) | 3 | [null, null, 2.0, 3.5, 4.0] (若 min_periods=2) |
  | 长度不足 | [1.0, 2.0] (SMA) | 5 | [null, null] |
```

---

## CrossSection 算子规范模版

```markdown
## [算子名称] (英文简称)

- **分类**: CrossSection
- **枚举**: `CrossSectionOpCode::XXX`
- **参数**: 如 numBuckets（运行时参数）
- **数学形式**: 用 LaTeX 或伪代码描述

  ```
  For each time t:
    values = [asset_0[t], asset_1[t], ..., asset_{N-1}[t]]
    result_i[t] = statistic(values, i)   // per-asset cross-sectional stat
  ```

- **输入类型**: N 个资产的数值数组（单时间点）
- **输出类型**: 每个资产 `Column<double>`
- **null 处理**:

  | 条件 | 行为 |
  |------|------|
  | 资产在时间 t 为 null | 该资产不参与截面统计计算 |
  | 所有资产均为 null | 全部输出 null |
  | 只有一个有效资产 | std=0 → zscore 输出 0.0；rank 输出 0.5 |

- **时间复杂度**: 标量参考 O(N log N) per time point（排序类）或 O(N)（均值和标准差类）
- **依赖**: MarketDataBundle（远期）
- **测试用例**:

  | 用例 | 输入 (5 assets @ time t) | 期望输出 |
  |------|--------------------------|---------|
  | ZScore 正常 | [1.0, 2.0, 3.0, 4.0, 5.0] | [-1.41, -0.71, 0.0, 0.71, 1.41] (约) |
  | Rank 正常 | [3.0, 1.0, 5.0, 2.0, 4.0] | [3.0, 1.0, 5.0, 2.0, 4.0] |
  | 含 null | [1.0, null, 3.0, null, 5.0] | null 资产输出不参与计算 |
```
