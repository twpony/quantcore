## CSNORMALIZEL2

- **分类**: Cs

- **枚举**: `CsOpCode::CS_NORMALIZEL2`

- **数学形式**:  $normalizel1(X)$；对每个时间截面（如每个交易日）的数据进行 平方和归一化。

- **输入类型**: $X: ColView<double>;$  

- **输出类型**: `Column<double>`

- **空值传播**: 输入为null时，输出也为null

- **定义域/值域约束**: 需处理值越界的情形

- **首元素/越界处理**: 需处理值越界的情形

- **数值稳定性注意事项**: 输出数据继续保持double类型

- **SIMD 优化要点**:  使用函数能够使用SIMD优化

- **对标参考**: $y_i=\frac{x_i}{\sqrt{\sum_j{x_j}^2}}$,  满足 $\sum_i{y_i}^2=1$

  
