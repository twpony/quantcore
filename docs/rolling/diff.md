## DIFF

- **分类**: Rolling
- **枚举**: `RollingOpCode::ROLLING_DIFF`
- **数学形式**:  $diff(X,n)$
- **输入类型**: $X: ColView<double>;$   n: int64_t，n是整形，窗口长度; 
- **输出类型**: `Column<double>`
- **空值传播**: 输入为null时，输出也为null
- **定义域/值域约束**: 需处理值越界的情形
- **首元素/越界处理**: 需处理值越界的情形
- **数值稳定性注意事项**: 输出数据继续保持double类型
- **SIMD 优化要点**:  使用函数能够使用SIMD优化
- **对标参考**: 返回X序列某个元素的值与该元素前n个序号的值的差，$x_k-x_{k-n}$；k是序列X某个元素的序列号；对标pandas.shift(periods)
