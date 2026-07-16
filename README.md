# QuantCore — 高性能 C++ 量化因子引擎

[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![Build](https://img.shields.io/badge/build-CMake-green.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

QuantCore 是一个基于现代 C++ 构建的**列式存储高性能量化因子计算引擎**，面向股票 K 线时序因子计算。一期实现 CPU SIMD 向量化、表达式模板惰性求值、算子融合等核心能力。

---

## 目录

- [架构概览](#架构概览)
- [目录结构](#目录结构)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [数据模型](#数据模型)
- [表达式系统](#表达式系统)
- [算子体系](#算子体系)
- [SIMD 向量化](#simd-向量化)
- [API 使用示例](#api-使用示例)
- [构建选项](#构建选项)
- [数据 I/O 层](#数据-io-层)
- [测试](#测试)
- [性能基准](#性能基准)
- [远期规划](#远期规划)
- [设计原则](#设计原则)

---

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                     ExecutionEngine                      │
│  表达式求值 · 算子融合 · SIMD 派发 · BufferPool 管理      │
├─────────────────────────────────────────────────────────┤
│                     表达式系统 (ET)                       │
│  ExprNode → UnaryExpr / BinaryExpr → FusedLoopGenerator │
├──────────────┬────────────────┬─────────────────────────┤
│   UnaryOps   │   BinaryOps    │     数据 I/O 层 ★        │
│  Abs Log     │  Add Sub Mul   │  Csv / Parquet / HDF5     │
│  Sqrt Diff   │  Div Max Min   │  Binary mmap 零拷贝        │
│  Shift Exp   │  Pow Cmp       │  pybind11 预留             │
├──────────────┴────────────────┴─────────────────────────┤
│                     数据存储层                            │
│  Column<T> → ColView<T> → TimestampIndex                 │
│  MarketData → MarketDataView                             │
├─────────────────────────────────────────────────────────┤
│                  基础设施                                 │
│  AlignedAllocator · Logger · ErrorHandling · SIMD Disp   │
└─────────────────────────────────────────────────────────┘
```

### 数据流

```
CSV / Parquet / HDF5 / 二进制文件
      │
      ▼
  ParquetDailyReader / HDF5Reader / CsvReader
      │
      ▼
  MarketData (内存)
      │
      ▼
  链式表达式 API 构建因子
  close + volume → abs(log(close) - log(vwap)) * volume
      │
      ▼
  ExecutionEngine.evaluate()
      ├── 表达式树展开
      ├── 算子融合 (单次遍历)
      ├── SIMD 内核派发 (AVX-512 / AVX2 / SSE4.2 / Scalar)
      └── 返回 Column<double>
```

---

## 目录结构

```
QuantCore/
├── CMakeLists.txt                  # 顶层 CMake
├── README.md                       # 本文件
├── QuantCore_Design_v2.md          # 详细设计规范
├── Factor_Engine.md                # 因子引擎设计文档
│
├── quantcore/                      # 核心库
│   ├── core/                       # 基础类型与工具
│   │   ├── Types.h                 # Field 枚举、算子枚举、常量
│   │   ├── AlignedAllocator.h      # 64字节对齐分配器
│   │   ├── Logger.h / .cpp         # 分级日志系统
│   │   └── ErrorHandling.h         # 异常体系与断言宏
│   │
│   ├── storage/                    # ★ 存储层 (已实现)
│   │   ├── Column.h / .cpp         # 泛型列式容器
│   │   ├── ColView.h               # 零拷贝只读视图
│   │   ├── TimestampIndex.h / .cpp # 时间轴索引
│   │   ├── MarketData.h / .cpp     # 单资产多字段数据集
│   │   └── MarketDataView.h        # 零拷贝切片视图
│   │
│   ├── expression/                 # 表达式系统
│   │   ├── ExprNode.h              # CRTP 基类
│   │   ├── ColumnRef.h             # 列引用叶节点
│   │   ├── Scalar.h                # 标量叶节点
│   │   ├── UnaryExpr.h             # 一元表达式节点
│   │   ├── BinaryExpr.h            # 二元表达式节点
│   │   ├── ExprTraits.h            # 类型萃取
│   │   └── FusedLoopGenerator.h    # 融合循环生成器
│   │
│   ├── operators/                  # 算子层
│   │   ├── UnaryOperator.h         # 一元算子接口
│   │   ├── BinaryOperator.h        # 二元算子接口
│   │   ├── unary/                  # 10 个一元算子
│   │   │   ├── Abs.h  Log.h  Sqrt.h  Neg.h  Sign.h
│   │   │   ├── Diff.h  Shift.h  Square.h  Exp.h  Log10.h
│   │   └── binary/                 # 11 个二元算子
│   │       ├── Add.h  Sub.h  Mul.h  Div.h  Pow.h
│   │       ├── Max.h  Min.h  Cmp.h (GT/LT/EQ/NEQ)
│   │
│   ├── simd/                       # SIMD 向量化
│   │   ├── SimdDispatcher.h/.cpp   # 运行时 CPU 特性派发
│   │   ├── SimdTraits.h            # SIMD 类型萃取
│   │   ├── avx512/                 # AVX-512 intrinsics (8 doubles)
│   │   ├── avx2/                   # AVX2 intrinsics    (4 doubles)
│   │   └── sse42/                  # SSE4.2 intrinsics  (2 doubles)
│   │
│   ├── engine/                     # 执行引擎
│   │   ├── ExecutionEngine.h/.cpp  # 核心调度器
│   │   ├── BufferPool.h/.cpp       # 分级 Slab 缓冲池
│   │   ├── BufferHandle.h          # RAII 内存句柄
│   │   └── EngineMetrics.h         # 性能指标收集
│   │
│   ├── io/                         # ★ 数据 I/O (多格式支持)
│   │   ├── CsvReader.h / .cpp      # CSV 行情数据解析
│   │   ├── BinaryWriter.h / .cpp   # 二进制列存写入
│   │   ├── BinaryReader.h / .cpp   # mmap 零拷贝读取
│   │   ├── ParquetReader.h / .cpp  # ★ Parquet 日线数据读取 (Arrow)
│   │   └── HDF5Reader.h / .cpp     # ★ HDF5 日线数据读取
│   │
│   └── registry/                   # 算子注册中心
│       └── OperatorRegistry.h/.cpp # 算子名称 ↔ 枚举映射
│
├── tests/                          # 测试
│   ├── unit/                       # 单元测试 (65%+)
│   │   ├── test_column.cpp         # ★ Column/ColView 测试
│   │   ├── test_market_data.cpp    # ★ MarketData/Timestamp 测试
│   │   ├── test_unary_ops.cpp      # 一元算子测试
│   │   ├── test_binary_ops.cpp     # 二元算子测试
│   │   ├── test_expression.cpp     # 表达式系统测试
│   │   ├── test_buffer_pool.cpp    # 缓冲池测试
│   │   ├── test_simd_cross_validate.cpp  # SIMD 交叉验证
│   │   ├── test_registry.cpp       # 算子注册测试
│   │   └── test_csv_reader.cpp     # CSV 解析测试
│   ├── integration/                # ★ 集成测试 (真实数据)
│   │   ├── test_full_pipeline.cpp  # 端到端：加载→计算→验证
│   │   ├── test_csv_roundtrip.cpp  # CSV 往返测试
│   │   ├── test_parquet_reader.cpp # ★ Parquet 真实数据测试
│   │   └── test_hdf5_reader.cpp   # ★ HDF5 真实数据测试
│   └── regression/                 # 回归测试
│       └── golden_factors.cpp      # Golden File 数值验证
│
├── benchmarks/                     # 性能基准
│   ├── bench_unary_ops.cpp         # 一元算子吞吐量
│   ├── bench_binary_ops.cpp        # 二元算子吞吐量
│   ├── bench_expression_fusion.cpp # 融合 vs 非融合对比
│   └── bench_buffer_pool.cpp       # 缓冲池性能
│
├── examples/                       # 示例
│   └── demo_basic_factors.cpp      # 基础因子计算演示
│
├── python/                         # Python 绑定 (预留)
├── scripts/                        # 构建脚本
│   ├── build.sh                    # Release 构建
│   └── build_debug.sh              # Debug 构建
└── cmake/                          # CMake 模块
    ├── CompilerSettings.cmake      # 编译器配置
    ├── FindGTest.cmake             # GTest 发现/下载
    ├── FindGBenchmark.cmake        # GBenchmark 发现/下载
    ├── FindArrow.cmake             # ★ Arrow + Parquet C++ 发现
    └── FindHDF5.cmake              # ★ HDF5 C/C++ 发现
```

★ 标记 = 本期已完整实现

---

## 核心特性

### 数据存储层

| 特性 | 说明 |
|------|------|
| **SoA 列式存储** | 每字段独立连续数组，禁用 AOS 结构体数组 |
| **64 字节对齐** | 适配 AVX-512 缓存行，使用 `AlignedAllocator<T, 64>` |
| **泛型支持** | `Column<double>` / `Column<int64_t>` / `Column<uint8_t>` (量额已改为 double) |
| **空值 bitmask** | 惰性分配，仅存在空值时占用内存，NaN/Null 语义分离 |
| **零拷贝视图** | `ColView<T>` 不持有内存，`MarketDataView` 零拷贝切片 |
| **mmap 加载** | 二进制格式内存布局与 `Column<T>` 一致，支持直接映射 |
| **类型异构** | 所有 7 个字段统一使用 `Column<double>`，`std::variant` 保留扩展能力 |

### 表达式系统

| 特性 | 说明 |
|------|------|
| **表达式模板** | CRTP 静态多态，编译期类型编码，零虚函数开销 |
| **惰性求值** | 链式 API 仅构建表达式树，调用 `.evaluate()` 才计算 |
| **算子融合** | 嵌套表达式编译为单次循环遍历，零中间临时列 |
| **公共子表达式消除** | 自动识别重复子表达式并复用结果 |
| **类型安全** | 编译期检测算子输入/输出类型不匹配 |

### SIMD 向量化

| 级别 | 指令集 | 寄存器 | doubles/cycle | 状态 |
|------|--------|--------|---------------|------|
| L3 | AVX-512F | 512-bit | 8 | 预留 |
| L2 | AVX2 + FMA | 256-bit | 4 | 预留 |
| L1 | SSE4.2 | 128-bit | 2 | 预留 |
| L0 | Scalar | — | 1 | 已实现 |

- 运行时 CPUID 检测 + 函数指针派发
- 每条 SIMD 路径必须有对标量路径的交叉验证测试
- 循环尾部不足向量宽度时自动降级标量

### 内存管理

```
BufferPool (分级 Slab Allocator)
  ├── Slab 64B    →  8 个 double
  ├── Slab 256B   →  32 个 double
  ├── Slab 1KB    →  128 个 double  (常用)
  ├── Slab 4KB    →  512 个 double
  ├── Slab 16KB   →  2048 个 double
  ├── Slab 64KB   →  8192 个 double
  ├── Slab 256KB  →  32768 个 double
  └── Slab 1MB    →  131072 个 double
```

- RAII `BufferHandle` 析构自动归还
- 全局 64 字节对齐保证
- 峰值/命中率统计

---

## 快速开始

### 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|---------|---------|
| C++ 编译器 | GCC 11 / Clang 15 | GCC 11+ / Clang 15+ |
| CMake | 3.20 | 3.25+ |
| C++ 标准 | C++20 | C++20 |
| Google Test | 1.12 | (CMake 自动下载) |
| Google Benchmark | 1.8 | (CMake 自动下载) |
| Apache Arrow | 25.0.0 | (conda: `libarrow libparquet`) |
| HDF5 | 2.1.0 | (conda: `hdf5`) |

### 安装依赖 (conda)

```bash
# 安装 Arrow/Parquet 和 HDF5 C++ 库 (无需 sudo)
conda install -c conda-forge libarrow=25.0.0 libparquet=25.0.0 hdf5
```

### 构建

```bash
git clone https://github.com/twpony/quantcore.git
cd quantcore

# Release 构建 + 测试 + 示例
bash scripts/build.sh

# Debug 构建 (含 AddressSanitizer + UBSan)
bash scripts/build_debug.sh
```

### 手动 CMake 构建

```bash
mkdir build && cd build

# 基本构建
cmake .. -DCMAKE_BUILD_TYPE=Release

# 完整构建选项
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DQUANTCORE_BUILD_TESTS=ON \
    -DQUANTCORE_BUILD_BENCHMARKS=ON \
    -DQUANTCORE_BUILD_EXAMPLES=ON \
    -DQUANTCORE_ENABLE_AVX512=ON \
    -DQUANTCORE_ENABLE_AVX2=ON \
    -DQUANTCORE_ENABLE_SSE42=ON \
    -DQUANTCORE_ENABLE_FAST_MATH=OFF

cmake --build . -j $(nproc)
```

### 运行测试

```bash
cd build
ctest --output-on-failure
# 或单独运行
./tests/test_column
./tests/test_market_data
./tests/test_parquet_reader   # 需本地 Parquet 数据
./tests/test_hdf5_reader      # 需本地 HDF5 数据
```

### 使用真实数据测试

```bash
# 默认数据目录: /home/twpony/quant/twpony/data_files/daily
./tests/test_parquet_reader

# 自定义数据目录
QC_DATA_DIR=/path/to/your/daily  ./tests/test_parquet_reader
QC_HDF5_DATA_DIR=/path/to/hdf5   ./tests/test_hdf5_reader
```


---

## 数据模型

### 字段定义

| 枚举 | 字段 | 存储类型 | 说明 |
|------|------|---------|------|
| `Field::OPEN` | 开盘价 | `double` | — |
| `Field::HIGH` | 最高价 | `double` | — |
| `Field::LOW` | 最低价 | `double` | — |
| `Field::CLOSE` | 收盘价 | `double` | — |
| `Field::VOLUME` | 成交量 | `double` | 保留原始精度，避免 llround 截断 |
| `Field::AMOUNT` | 成交额 | `double` | 保留原始精度，避免 llround 截断 |
| `Field::VWAP` | 均价 | `double` | 成交量加权平均价 |

### 数据层次

```
Column<T>                     — 单列连续内存 (含空值 bitmask)
    │
    ├── TimestampIndex        — 时间轴索引 (日期感知、交易日判断)
    │
    └── MarketData            — 单资产 7 字段数据集
           │
           ├── MarketDataView — 零拷贝切片视图
           │
           └── (远期) MarketDataBundle — 多资产对齐面板
```

### Column\<T\> 核心 API

```cpp
// 构造
Column<double> col(1000);                    // 定长
Column<double> col({1.0, 2.0, 3.0});        // 初始化列表
auto col = Column<double>::fromMmap(ptr, n); // mmap 零拷贝

// 访问
double v = col[42];
double* p = col.data();
size_t n  = col.size();

// 空值管理
col.setNull(5);       // 标记空值 (首次触发 bitmask 分配)
bool n = col.isNull(5);
size_t c = col.nullCount();
const uint64_t* mask = col.nullMaskData();

// 内存
bool aligned = col.isAligned();   // 是否 64 字节对齐
size_t bytes = col.memoryBytes();
```

### MarketData 核心 API

```cpp
// 构造
MarketData md("000001.SZ", timestampIndex);
md.allocateAllFields();  // 分配全部 7 个字段

// 字段访问
auto& close  = md.column<double>(Field::CLOSE);
auto& volume = md.column<double>(Field::VOLUME);
close[0] = 10.5;

// 元数据
md.assetId();           // "000001.SZ"
md.rowCount();          // 行数
md.allColumnsAligned(); // 校验列长度一致

// 零拷贝切片
MarketDataView view = md.slice(100, 200);       // 行切片
MarketDataView view = md.sliceByDate(20240101, 20240331);  // 日期切片

// 视图访问
auto closeView = view.column<double>(Field::CLOSE);
const auto& tsView = view.timestamps();
```

---

## 表达式系统

### 链式 API

```cpp
auto close   = marketData.column<double>(Field::CLOSE);
auto vwap    = marketData.column<double>(Field::VWAP);
auto volume  = marketData.column<double>(Field::VOLUME);

// 构建表达式 — 仅构建树，不计算
auto factorExpr = abs(log(close) - log(vwap)) * volume.toDouble();

// 求值 — 融合为单次循环
Column<double> result = engine.evaluate(factorExpr);

// 等价于:
// for i in 0..N:
//   result[i] = abs(log(close[i]) - log(vwap[i])) * double(volume[i])
```

### 算子融合效果

```
传统 (5 次循环, 4 个临时列):          融合 (1 次循环, 0 个临时列):
tmp1 = log(close)      // 循环1       result[i] = abs(
tmp2 = log(vwap)       // 循环2           log(close[i])
tmp3 = tmp1 - tmp2     // 循环3         - log(vwap[i])
tmp4 = abs(tmp3)       // 循环4       ) * double(volume[i])
res  = tmp4 * vol      // 循环5       // 单次遍历, 零中间内存
```

---

## 算子体系

### 一元算子 (UnaryOperator)

| 算子 | 函数 | 数学形式 | 空值处理 |
|------|------|---------|---------|
| Abs | `abs(x)` | \|x\|| 传播空值 |
| Log | `log(x)` | ln(x) | x≤0 → NaN |
| Log10 | `log10(x)` | log₁₀(x) | x≤0 → NaN |
| Sqrt | `sqrt(x)` | √x | x<0 → NaN |
| Exp | `exp(x)` | eˣ | 传播空值 |
| Neg | `-x` | -x | 传播空值 |
| Sign | `sign(x)` | sign(x) | 传播空值 |
| Square | `square(x)` | x² | 传播空值 |
| Diff | `diff(x)` | x[i] - x[i-1] | 首元素标记空值 |
| Shift | `shift(x, n)` | x[i-n] | 越界标记空值 |

### 二元算子 (BinaryOperator)

| 算子 | 函数 | 操作数 | 说明 |
|------|------|--------|------|
| Add | `a + b` | Col+Col / Col+Scalar | — |
| Sub | `a - b` | Col+Col / Col+Scalar | — |
| Mul | `a * b` | Col+Col / Col+Scalar | — |
| Div | `a / b` | Col+Col / Col+Scalar | 除零 → NaN |
| Pow | `a^b` | Col+Scalar | — |
| Max | `max(a,b)` | Col+Col / Col+Scalar | — |
| Min | `min(a,b)` | Col+Col / Col+Scalar | — |
| GT | `a > b` | Col+Col / Col+Scalar | 返回 bool |
| LT | `a < b` | Col+Col / Col+Scalar | 返回 bool |
| EQ | `a == b` | Col+Col / Col+Scalar | 返回 bool |
| NEQ | `a != b` | Col+Col / Col+Scalar | 返回 bool |

---

## SIMD 向量化

### 运行时派发

```cpp
// 启动时自动检测 CPU 能力
SimdLevel level = SimdDispatcher::bestLevel();
// 返回: SCALAR / SSE42 / AVX2 / AVX512

// 算子执行时自动选择最优内核
template <typename Op>
auto kernel = SimdDispatcher::dispatch<Op>(level);
```

### 交叉验证

每个算子的 SIMD 路径必须通过标量路径交叉验证：

```cpp
// Debug 模式下自动执行
TEST(UnaryOpSimd, AbsCrossValidate) {
    auto input   = generateTestData(1024);  // 含 NaN/Inf/边界值
    auto scalar  = evaluateScalar<AbsOp>(input);
    auto avx2    = evaluateAvx2<AbsOp>(input);
    EXPECT_COLUMNS_NEAR(scalar, avx2, /*maxUlp=*/1);
}
```

---

## API 使用示例

### 示例 1: 构建并计算因子

```cpp
#include "quantcore/storage/MarketData.h"
#include "quantcore/engine/ExecutionEngine.h"

using namespace quantcore;

int main() {
    // 1. 加载数据
    auto timestamps = loadTimestamps();  // 用户实现
    TimestampIndex tsIdx(timestamps.data(), timestamps.size());
    MarketData md("000001.SZ", std::move(tsIdx));
    md.allocateAllFields();

    // 填充数据 (实际场景从 CSV 或数据库加载)
    for (size_t i = 0; i < md.rowCount(); ++i) {
        md.column<double>(Field::CLOSE)[i]  = /* ... */;
        md.column<double>(Field::VWAP)[i]   = /* ... */;
        md.column<double>(Field::VOLUME)[i] = /* ... */;
    }

    // 2. 构建因子表达式
    ExecutionEngine engine;
    auto& close  = md.column<double>(Field::CLOSE);
    auto& vwap   = md.column<double>(Field::VWAP);
    auto& volume = md.column<double>(Field::VOLUME);

    // 日内振幅
    auto amplitude = (md.column<double>(Field::HIGH) -
                      md.column<double>(Field::LOW)) /
                      md.column<double>(Field::CLOSE);

    // 量价复合因子
    auto factor = abs(log(close) - log(vwap)) * volume.toDouble();

    // 3. 计算
    Column<double> ampResult  = engine.evaluate(amplitude);
    Column<double> facResult  = engine.evaluate(factor);

    // 4. 查看结果
    for (size_t i = 0; i < std::min(size_t(10), ampResult.size()); ++i) {
        printf("row %zu: amplitude=%.6f  factor=%.2f\n",
               i, ampResult[i], facResult[i]);
    }

    return 0;
}
```

### 示例 2: 空值处理

```cpp
Column<double> close = {10.0, 10.5, 10.3, 10.8, 10.6};

// 标记停牌日为空值
close.setNull(2);  // 第3天停牌

// 计算收益率时会自动传播空值
auto ret = log(close) - log(shift(close, 1));
// ret[1] = log(10.5) - log(10.0)  → 正常
// ret[2] = null                    → 空值传播 (输入 null)
// ret[3] = log(10.8) - log(10.5)  → 正常 (shift 跳过了空值)
```

### 示例 3: 数据切片

```cpp
// 按日期范围获取子视图
auto view2024Q1 = md.sliceByDate(20240101, 20240331);

// 视图零拷贝 — 修改原数据会反映到视图中
md.column<double>(Field::CLOSE)[0] = 999.0;
// view2024Q1.column<double>(Field::CLOSE)[0] 也会变为 999.0

// 链式计算在子视图上
auto q1Factor = engine.evaluate(
    log(view2024Q1.column<double>(Field::CLOSE))
);
```

---

## 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `QUANTCORE_BUILD_TESTS` | ON | 编译单元测试 |
| `QUANTCORE_BUILD_BENCHMARKS` | ON | 编译性能基准 |
| `QUANTCORE_BUILD_EXAMPLES` | ON | 编译示例程序 |
| `QUANTCORE_ENABLE_AVX512` | ON | 启用 AVX-512 intrinsics |
| `QUANTCORE_ENABLE_AVX2` | ON | 启用 AVX2 intrinsics |
| `QUANTCORE_ENABLE_SSE42` | ON | 启用 SSE4.2 intrinsics |
| `QUANTCORE_ENABLE_PARQUET` | ON | 启用 Parquet I/O（需 Apache Arrow） |
| `QUANTCORE_ENABLE_HDF5` | ON | 启用 HDF5 I/O（需 HDF5 库） |
| `QUANTCORE_ENABLE_FAST_MATH` | **OFF** | 启用 `-ffast-math`（可能损失精度） |

> **重要**: `-ffast-math` 默认关闭。金融计算优先保证数值正确性和可复现性。

### 编译模式

| 模式 | 优化 | 断言 | 交叉验证 | 适用场景 |
|------|------|------|---------|---------|
| Debug | `-O0 -g` | 全量 | ✓ 逐元素对比 | 开发调试 |
| Release | `-O3 -march=native` | 关闭 | ✗ | 生产运行 |
| ReleaseFast | `-O3 -ffast-math` | 关闭 | ✗ | 性能极致 (有风险) |

---

## 测试

### 测试金字塔

```
         ┌─────────────┐
         │  集成测试     │  ~10%  — 端到端链路
         │  Integration │
         ├─────────────┤
         │  回归测试     │  ~15%  — Golden File 数值验证
         │  Regression  │
         ├─────────────┤
         │  单元测试     │  ~65%  — 边界值/空值/异常
         │  Unit Tests  │
         └─────────────┘
```

### 测试覆盖

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_column.cpp` | Column: 构造/拷贝/移动/空值/aligned/mmap/大容量/比较 |
| `test_market_data.cpp` | MarketData/TimestampIndex/MarketDataView 全部 API |
| `test_unary_ops.cpp` | 10 种一元算子: 正确性/NaN/Inf/空值传播 |
| `test_binary_ops.cpp` | 11 种二元算子: Col+Col/Col+Scalar/除零/空值 |
| `test_expression.cpp` | 嵌套表达式/融合等价性/CSE 正确性 |
| `test_buffer_pool.cpp` | 分配/归还/RAII/多级 slab/峰值/对齐 |
| `test_simd_cross_validate.cpp` | SIMD vs 标量逐元素交叉验证 |
| `test_csv_reader.cpp` | CSV 解析/错误行跳过/日期格式 |
| `test_registry.cpp` | 算子注册/查询/重复注册检测 |
| `test_parquet_reader.cpp` | ★ Parquet 真实数据: 文件读取/OHLC 校验/因子计算 |
| `test_hdf5_reader.cpp` | ★ HDF5 读取: 配置/异常/多格式兼容 |

### 集成测试 (基于真实本地数据)

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_parquet_reader.cpp` | ★ 单文件读取、字段验证、多日期、OHLC 完整性、TimestampIndex、因子计算、股票筛选、内存对齐 |
| `test_hdf5_reader.cpp` | ★ HDF5 配置、自定义 schema、已存在文件读取、异常处理、统计追踪 |
| `test_full_pipeline.cpp` | CSV 加载 → 因子计算 → 结果验证 |
| `test_csv_roundtrip.cpp` | CSV 读 → Binary 写 → Binary mmap 读 → 对比 |

### 回归测试

```cpp
// Golden File: 固定输入 → 固定输出, 保证重构不改变数值
TEST(RegressionTest, StandardFactorSet) {
    auto data = loadGoldenMarketData("test/data/golden_input.csv");
    auto engine = ExecutionEngine();
    auto momentum  = engine.evaluate(close / shift(close, 20) - 1.0);
    auto amplitude = engine.evaluate((high - low) / close);
    assertNearGolden("momentum_20d.csv", momentum, /*tolerance=*/1e-12);
}
```

---

## 性能基准

| 基准测试 | 数据规模 | 指标 |
|---------|---------|------|
| `bench_unary_ops` | 1K / 10K / 100K / 1M | SIMD vs Scalar 加速比 |
| `bench_binary_ops` | 同上 | Col+Col / Col+Scalar 吞吐量 |
| `bench_expression_fusion` | 同上 | 融合 vs 非融合性能对比 |
| `bench_buffer_pool` | 1000 次操作 | 分配/归还吞吐量、命中率 |
| `bench_csv_load` | 1MB / 10MB / 100MB | CSV 解析吞吐量 |
| `bench_binary_load` | 同上 | mmap 加载吞吐量 |

```bash
# 运行所有基准测试
cd build
./benchmarks/bench_unary_ops
./benchmarks/bench_expression_fusion
```

---

## 数据 I/O 层

QuantCore 支持多种数据格式的读取，通过统一的 I/O 层接口将不同格式的数据源转换为 `MarketData` 对象。

### 支持的格式

| 格式 | 读取器 | 库依赖 | 状态 |
|------|--------|--------|------|
| **Parquet** | `ParquetDailyReader` | Apache Arrow C++ 25+ | ✅ 已实现 |
| **HDF5** | `HDF5Reader` | HDF5 C++ API | ✅ 已实现 |
| CSV | `CsvMarketDataReader` | 标准库 | 🔧 接口就绪 |
| 二进制列存 | `BinaryMarketDataReader` | 标准库 + mmap | 🔧 接口就绪 |

### ParquetDailyReader

读取按日期分片的 Parquet 日线数据，每文件包含当日全市场股票。

**文件命名**: `YYYYMMDD.parquet`（如 `20260629.parquet`）

**数据 Schema**:
| 列名 | 类型 | 映射字段 |
|------|------|---------|
| `ts_code` | string | 资产代码 |
| `trade_date` | string/int64 | 交易日期 (YYYYMMDD) |
| `open` | double | `Field::OPEN` |
| `high` | double | `Field::HIGH` |
| `low` | double | `Field::LOW` |
| `close` | double | `Field::CLOSE` |
| `vol` | double | `Field::VOLUME` (直接存储 double) |
| `amount` | double | `Field::AMOUNT` (直接存储 double) |
| (计算) | double | `Field::VWAP` = amount × 10 / vol |

```cpp
#include "quantcore/io/ParquetReader.h"

// 读取单个日期文件
ParquetDailyReader reader;
auto result = reader.readFile("/data/daily/20260629.parquet");

// result.assets     → std::vector<MarketData> (每股票一个)
// result.tradeDate  → 20260629
// result.rowsRead   → 5510

// 遍历结果
for (const auto& md : result.assets) {
    std::cout << md.assetId() << ": CLOSE="
              << md.column<double>(Field::CLOSE)[0] << "\n";
}

// 按市场筛选
int shCount = 0;
for (const auto& md : result.assets) {
    if (md.assetId().ends_with(".SH")) ++shCount;
}

// 批量读取整个目录 (按股票聚合)
auto stockMap = reader.readDirectory("/data/daily/");
// → map<string, vector<pair<date, MarketData>>>
// stockMap["000001.SZ"] = [(20260626, md_day1), (20260627, md_day2), ...]
```

### HDF5Reader

读取 HDF5 格式的日线数据，支持列式存储布局。

```cpp
#include "quantcore/io/HDF5Reader.h"

HDF5ReaderConfig cfg;
cfg.datasetPath = "/data";      // HDF5 dataset 路径
cfg.openColumn  = "open";       // 可自定义列名
cfg.closeColumn = "close";

HDF5Reader reader(cfg);
auto result = reader.readFile("/data/market_2026.h5");
// result.assets → 同 ParquetDailyReader 的输出格式
```

### 自定义列名映射

```cpp
ParquetReaderConfig cfg;
cfg.tsCodeColumn     = "symbol";       // 不叫 ts_code
cfg.openColumn       = "OPEN_PRICE";   // 不叫 open
cfg.volumeColumn     = "volume_shares";
cfg.skipNullRows     = false;          // 保留空值行

ParquetDailyReader reader(cfg);
auto result = reader.readFile("custom_schema.parquet");
```

### 条件编译

当库不可用时，阅读器编译为安全的 stub 实现，调用时输出警告日志：

```cmake
# 禁用 Parquet 支持
cmake .. -DQUANTCORE_ENABLE_PARQUET=OFF

# 禁用 HDF5 支持
cmake .. -DQUANTCORE_ENABLE_HDF5=OFF
```

代码中可通过宏判断：
```cpp
#if QUANTCORE_HAS_PARQUET
    // Parquet 可用
#endif
```

---

## 错误处理

### 异常层级

```
std::exception
  └── QuantCoreError (含文件/行号/函数上下文)
        ├── ConfigError      — 配置错误 (列长度不匹配、字段不存在)
        ├── DataError        — 数据错误 (NaN、空值异常)
        ├── ResourceError    — 资源错误 (BufferPool 耗尽)
        └── InternalError    — 内部错误 (违反不变量)
```

### 空值传播语义

| 场景 | 行为 |
|------|------|
| 一元算子输入 null | 输出 = null |
| 二元 Col+Col 任一 null | 输出 = null |
| 二元 Col+Scalar Col 为 null | 输出 = null |
| Diff 首元素 | 输出 = null |
| Shift 越界 | 输出 = null |

---

## 远期规划

| 优先级 | 功能 | 说明 |
|--------|------|------|
| P1 | 多线程并行 | per-stock 数据并行 + per-operator 任务并行 |
| P1 | RollingOperator | SMA/EMA/滚动最大最小/标准差/排名 |
| P1 | 完整 SIMD 路径 | AVX2/AVX-512 intrinsics 手写实现 |
| P2 | 字符串公式解析 | `"ABS(LOG(CLOSE)-LOG(VWAP))*VOLUME"` → 表达式树 |
| P2 | CrossSectionOperator | 多标的截面统计 (排名、分位数、标准化) |
| P2 | ReductionOperator | 全局求和/均值/极值/标准差 |
| P2 | Python 绑定 | pybind11 完整暴露核心 API |
| P3 | GPU 后端 | CUDA/HIP/OpenCL 算子加速 |
| P3 | MarketDataBundle | 多资产对齐面板，支持截面操作 |
| P3 | 遗传规划 | 自动因子生成与筛选 |
| P3 | Arrow 完整兼容 | Arrow 列存格式互操作 (Parquet 读取已完成) |

---

## 设计原则

1. **正确性优先** — 任何优化不得牺牲数值正确性
2. **面向数据设计** — 禁止逐根 K 线循环，全部向量批量运算
3. **SoA 列式存储** — 字段分离数组，禁用 AOS 结构体数组
4. **惰性求值** — 链式 API 构建 DAG，evaluate 时融合单次遍历
5. **零拷贝** — ColView 视图访问，BufferPool 复用中间结果
6. **显式 SIMD** — intrinsics 优先，编译器向量化 fallback
7. **分层可验证** — 每层标量参考实现，SIMD 路径交叉验证
8. **表达式驱动** — 支持多层嵌套复合数学表达式

```
正确性 > 可复现性 > 性能 > 扩展性 > 代码简洁度
```

---

## 开发状态

| 模块 | 状态 | 完成度 |
|------|------|--------|
| 存储层 (Column/ColView/MarketData) | ✅ 已实现 | 100% |
| 日志系统 (Logger) | ✅ 已实现 | 100% |
| 错误处理 (Exception + Assert) | ✅ 已实现 | 100% |
| 对齐分配器 (AlignedAllocator) | ✅ 已实现 | 100% |
| 基础类型 (Types) | ✅ 已实现 | 100% |
| 表达式系统 | 🔧 接口就绪 | ~20% |
| 算子层 (Unary/Binary) | 🔧 接口就绪 | ~10% |
| SIMD 向量化 | 🔧 框架就绪 | ~5% |
| 执行引擎 (Engine/BufferPool) | 🔧 接口就绪 | ~5% |
| 数据 I/O (CSV/Binary) | 🔧 接口就绪 | ~5% |
| 数据 I/O (ParquetReader) | ✅ 已实现 | 100% |
| 数据 I/O (HDF5Reader) | ✅ 已实现 | 100% |
| 算子注册中心 | 🔧 接口就绪 | ~10% |
| 单元测试 | 🔧 存储+IO层完成 | ~35% |

---

## License

MIT License. 详见 [LICENSE](LICENSE) 文件。

---

## 相关文档

- [QuantCore 设计规范 v2.0](QuantCore_Design_v2.md) — 完整架构设计
- [Factor Engine 设计](Factor_Engine.md) — 因子引擎早期设计文档
