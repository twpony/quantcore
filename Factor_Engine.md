

QuantCore 高性能 C++ 量化因子引擎设计规范

（适用于交付 AI 生成完整工程代码）

文档基本信息

文档名称：QuantCore C++ High Performance Quant Factor Library 设计规范

适用场景：基于 7 类 K 线行情字段开发时序因子计算库，交由 AI 生成全套可编译 C++ 工程

开发阶段：一期基础版本（仅实现 CPU SIMD 基础能力，GPU、遗传规划等为远期预留）

一、项目整体目标

基于现代 C++ 构建列式存储高性能量化因子计算引擎 QuantCore，面向股票 K 线时序因子计算，分层架构规划如下：

1. 一期落地功能：列式存储数据结构、一元向量算子、二元标量 / 向量算术算子、执行调度引擎、全局对齐内存缓冲池；仅基于OPEN/HIGH/LOW/CLOSE/VOLUME/AMOUNT/VWAP7     个基础行情字段完成表达式计算；
2. 远期扩展规划：多线程并行、DAG 表达式优化、CUDA/HIP/OpenCL GPU 算子后端、横截面因子、滚动窗口聚合算子、遗传规划因子生成模块；
3. 底层核心特性：列式存储、零拷贝、SIMD 向量化批量计算、解耦硬件后端、编译期低开销表达式系统。

二、顶层强制设计原则

1. 面向数据设计（Data-Oriented Design）：禁止逐根 K 线循环计算，全部采用批量向量运算；
2. 列式存储 SoA 架构：行情字段分离数组存储，禁用 AOS 单 K 线结构体数组；
3. 零拷贝（Zero Copy）：视图结构访问列数据，避免内存频繁拷贝；
4. SIMD 优先：所有数值计算优先向量化实现，单元素循环仅作兜底；
5. 表达式驱动（Expression Oriented）：支持多层嵌套复合数学表达式构建因子；
6. 硬件后端无关（Backend Independent）：算子业务逻辑与 CPU/GPU 硬件解耦；
7. 零运行时开销（Zero Runtime Overhead）：大量使用编译期模板、常量枚举，减少运行时分支判断。

三、底层数据存储模型规范

3.1 固定基础行情字段

统一 7 类浮点时序序列：OPEN、HIGH、LOW、CLOSE、VOLUME、AMOUNT、VWAP

3.2 核心数据结构定义

1. Column<double>：单列连续内存容器，统一使用     double 浮点类型；自定义AlignedAllocator实现 64 字节内存对齐，适配 AVX2/AVX512 向量化读写；
2. MarketData：顶层行情数据集容器，内部持有 7 个独立 Column；内置长度校验逻辑，保证全部时序列长度对齐；支持切片生成子行情数据集；
3. ColView<double>：零拷贝只读列视图，仅保存原始列指针、起止下标，不持有内存所有权，用于切片、滑动窗口截取，完全实现零拷贝访问。

四、表达式系统需求

4.1 支持运算形式

1. 一元数学变换：LOG、ABS、SQRT、取负等；
2. 向量 - 标量四则运算：加减乘除；
3. 向量 - 向量二元算术运算：CLOSE-OPEN、VOLUME*VWAP、分式复合计算；
4. 多层嵌套复合表达式：ABS (LOG (CLOSE)-LOG (VWAP))。

4.2 分阶段实现要求

1. 一期：提供链式 API 手动拼接算子构建表达式，无需完整 AST 抽象语法树；
2. 远期迭代：新增完整 Expression Tree、字符串公式解析模块。

五、算子分层体系

算子统一输入输出Column<double>时序序列，分为五大分类，一期仅实现 Unary、Binary 两类基础算子：

1. UnaryOperator 一元向量算子：单序列输入数学变换（Log/Abs/Sqrt/Diff/Shift     等）；
2. BinaryOperator 二元算子：细分向量 + 标量、向量 + 向量两类算术运算；
3. RollingOperator 滚动窗口算子（远期）：SMA、EMA、滚动最大 / 最小 /     标准差；
4. CrossSectionOperator 横截面算子（远期）：多标的截面统计计算；
5. ReductionOperator 归约聚合算子（远期）：全局求和、均值、极值等。

六、SIMD 向量化编码强制规范

1. 全部批量数值计算开启编译器自动向量化；
2. 循环头部统一添加向量化制导：#pragma omp simd，兼容 GCC ivdep、Clang vectorize 备选指令；
3. 指针参数添加__restrict__关键字消除内存别名，提升向量化效率；
4. 所有计算缓冲区强制 64 字节内存对齐，适配 AVX 系列寄存器批量读写。

七、ExecutionEngine 执行引擎模块

全局统一调度核心，一期实现职责：

1. 管理全局 BufferPool 内存块申请与回收；
2. 接收链式算子表达式，串行调度批量计算流程；
3. 统一管理列内存生命周期、视图合法性校验；
4. 预留硬件后端抽象接口，兼容后续 GPU 算子扩展。

八、BufferPool 全局对齐缓冲池

1. 预分配大批量 64 字节对齐连续内存块，统一复用；
2. 规避频繁 new/malloc、vector 扩容带来的内存分配性能损耗；
3. 算子中间计算内存统一从缓冲池申请、计算完成后归还。

九、硬件后端分层架构

分层解耦，一期仅完整实现 CPU SIMD 后端：

1. 默认底层：CPU SIMD 后端（一期完整落地）；
2. 预留扩展后端：CUDA / HIP / OpenCL GPU 算子抽象基类（一期不实现具体逻辑）。

十、OperatorRegistry 算子注册中心

1. 统一注册所有一元、二元算子，建立算子枚举映射关系；
2. 远期预留接口：公式字符串解析、遗传规划批量生成因子；
3. 一期仅实现基础算子注册映射逻辑。

十一、CMake 编译构建规范

1. C++ 标准：最低 C++17，推荐启用 C++20；
2. Release 版本编译参数：-O3 -march=native     -DNDEBUG；
3. 可选优化参数：-ffast-math，浮点允许微小精度损耗换取 SIMD 加速；
4. 工程采用 CMake 管理，区分核心库、单元测试、性能基准测试目录，提供编译开关控制测试模块启用 / 关闭。

十二、测试与性能基准规范

1. 单元测试：基于 GoogleTest，覆盖边界值、NaN 空值、时序长度不匹配、非法窗口等异常场景；
2. 性能基准：基于 GoogleBenchmark，对全部一元、二元算子进行 SIMD 性能压测；
3. 标准测试数据源：使用包含 OPEN/HIGH/LOW/CLOSE/VOLUME/AMOUNT/VWAP 七字段的标准 K 线时序数据构造用例。

十三、AI 代码交付硬性要求

基于本文档规范输出完整可独立编译运行的 C++ QuantCore 工程代码，包含以下内容：

1. 分层头文件目录：存储层、算子层、执行引擎、缓冲池、表达式链式构建接口；
2. 仅落地一期规定功能：MarketData 列式存储、ColView 零拷贝视图、Unary/Binary 基础算子、SIMD 批量计算、BufferPool、ExecutionEngine、链式表达式 API；远期功能仅保留抽象预留接口，不填充实现；
3. 提供 main 示例代码：演示加载 7 列 K 线数据、构造多层复合因子表达式、输出计算结果；
4. 配套完整 CMakeLists.txt，内置 GTest、GBenchmark 编译开关；
5. 代码注释完整清晰，严格区分「一期必实现逻辑」和「远期扩展预留接口」，完全遵循本文档全部设计约束。