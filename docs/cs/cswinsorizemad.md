## CSWINSORIZEMAD

- **分类**: Cs

- **枚举**: `CsOpCode::CS_winsorize_mad`

- **数学形式**:  $winsorize(X，n)$；**使用中位数（Median）和中位数绝对偏差（MAD，m-median）来定义数据的“正常”范围，然后将超出该范围的值“拉回”到边界值**，$M=median(\{x_i\})$，$MAD=median(\{x_i-M\}_i)$

- **输入类型**: $X: ColView<double>;$ ，n为边界系数，double类型  

- **输出类型**: `Column<double>`

- **空值传播**: 输入为null时，输出也为null

- **定义域/值域约束**: 需处理值越界的情形

- **首元素/越界处理**: 需处理值越界的情形

- **数值稳定性注意事项**: 输出数据继续保持double类型

- **SIMD 优化要点**:  使用函数能够使用SIMD优化

- **对标参考**: 下限等于$median-k\times1.4826\times{MAD}$；上限等于$median+k\times1.4826\times{MAD}$；若 `X(t, i) < 下限`，则 `X(t, i)` 被替换为**下限值**。若 `X(t, i) > 上限`，则 `X(t, i)` 被替换为**上限值**。

  
