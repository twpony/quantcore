## EMA

- **分类**: Rolling
- **枚举**: `RollingOpCode::ROLLING_EMA`
- **数学形式**:  $ema(X,n)$
- **输入类型**: $X: ColView<double>;$   n: int64_t，n是整形，窗口长度; 
- **输出类型**: `Column<double>`
- **空值传播**: 输入为null时，输出也为null
- **定义域/值域约束**: 需处理值越界的情形
- **首元素/越界处理**: 需处理值越界的情形
- **数值稳定性注意事项**: 输出数据继续保持double类型
- **SIMD 优化要点**:  使用函数能够使用SIMD优化
- **对标参考**: 指数移动平均，$EMA_t=αX_t+(1−α)EMA_{t−1}, \ α = \frac{2}{(n+1)}$; 对标$pandas.ewm(span).mean()$
