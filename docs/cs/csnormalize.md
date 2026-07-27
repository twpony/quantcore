## CSNORMALIZE

- **分类**: Cs

- **枚举**: `CsOpCode::CS_NORMALIZE`

- **数学形式**:  $normalize(X)$；对每个时间截面（如每个交易日）的数据进行 Min-Max 归一化，将数据缩放到一个固定的数值区间，通常是 `[0, 1]`。

- **输入类型**: $X: ColView<double>;$  

- **输出类型**: `Column<double>`

- **空值传播**: 输入为null时，输出也为null

- **定义域/值域约束**: 需处理值越界的情形

- **首元素/越界处理**: 需处理值越界的情形

- **数值稳定性注意事项**: 输出数据继续保持double类型

- **SIMD 优化要点**:  使用函数能够使用SIMD优化

- **对标参考**: normalized_value(t, i) = (X(t, i) - min(t)) / (max(t) - min(t))；`X(t, i)` 是资产 `i` 在时间 `t` 的原始值；`min(t)` 是时间 `t` 上所有有效值的最小值；`max(t)` 是时间 `t` 上所有有效值的最大值。

  
