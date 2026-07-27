## SQURE

- **分类**: Unary
- **枚举**: `UnaryOpCode::SQURE`
- **数学形式**:  $y_i={x_i}^2$
- **输入类型**: `ColView<double>`
- **输出类型**: `Column<double>`
- **空值传播**: 输入为null时，输出也为null
- **定义域/值域约束**: 需要处理值超过边界的情况
- **首元素/越界处理**: 不适用
- **数值稳定性注意事项**: 输出数据继续保持double类型
- **SIMD 优化要点**:  使用函数能够使用SIMD优化
- **对标参考**: 参考`numpy.square`的结果
