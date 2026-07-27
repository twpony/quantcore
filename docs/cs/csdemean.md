## CSDEMEAN

- **分类**: Cs

- **枚举**: `CsOpCode::CS_DEMEAN`

- **数学形式**:  $demean(X)$

- **输入类型**: $X: ColView<double>;$  

- **输出类型**: `Column<double>`

- **空值传播**: 输入为null时，输出也为null

- **定义域/值域约束**: 需处理值越界的情形

- **首元素/越界处理**: 需处理值越界的情形

- **数值稳定性注意事项**: 输出数据继续保持double类型

- **SIMD 优化要点**:  使用函数能够使用SIMD优化

- **对标参考**: demeaned_value(t, i) = X(t, i) - mean(t)，`mean(t)` 是时间 `t` 上所有有效资产值的均值；每个时间点减去截面均值。

  
