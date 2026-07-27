## SMA

- **分类**: Rolling
- **枚举**: `RollingOpCode::ROLLING_SMA`
- **数学形式**:  $sma(X,n)$
- **输入类型**: $X: ColView<double>;$   n: int64_t，n是整形，窗口长度; 
- **输出类型**: `Column<double>`
- **空值传播**: 输入为null时，输出也为null
- **定义域/值域约束**: 需处理值越界的情形
- **首元素/越界处理**: 需处理值越界的情形
- **数值稳定性注意事项**: 输出数据继续保持double类型
- **SIMD 优化要点**:  使用函数能够使用SIMD优化
- **对标参考**: 简单移动平均，X序列中过去n个元素值的算式平均值；对标$pandas.rolling(window).mean()$
