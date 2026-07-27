## ADD

- **分类**: Binary
- **枚举**: `BinaryOpCode::ADD`
- **数学形式**:  $ADD(X,Y)$
- **输入类型**: 1)`X: ColView<double>, Y: ColView<double>`;  2) `X: ColView<double>, Y: double`;  3) `X: double, Y: double`
- **输出类型**: 1)`Column<double>`;  2) `Column<double>`;  3) `double`; 
- **空值传播**: 输入为null时，输出也为null
- **定义域/值域约束**: 需处理值越界的情形
- **首元素/越界处理**: 不适用
- **数值稳定性注意事项**: 输出数据继续保持double类型
- **SIMD 优化要点**:  使用函数能够使用SIMD优化
- **对标参考**: 1) 当X和Y都是向量时，结果是向量Z，其每个元素$z_i=x_i+y_i$；2）当X是向量，Y是标量，则结果是向量Z，其每个元素$z_i=x_i+Y$；3）当X是标量，Y是标量，则结果是标量Z，$Z=X+Y$
- **注意要点**：虽然有3种输入方式，但请使用统一的$Z=ADD(X,Y)$接口
