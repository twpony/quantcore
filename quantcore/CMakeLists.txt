# QuantCore core library
add_library(quantcore STATIC)

target_sources(quantcore PRIVATE
    # core
    core/Logger.cpp

    # storage
    storage/Column.cpp
    storage/TimestampIndex.cpp
    storage/MarketData.cpp

    # engine
    engine/ExecutionEngine.cpp
    engine/BufferPool.cpp

    # simd
    simd/SimdDispatcher.cpp

    # io
    io/CsvReader.cpp
    io/BinaryWriter.cpp
    io/BinaryReader.cpp
    io/ParquetReader.cpp
    io/HDF5Reader.cpp

    # registry
    registry/OperatorRegistry.cpp
)

target_include_directories(quantcore PUBLIC ${PROJECT_SOURCE_DIR})

# SIMD preprocessor defines
if(QUANTCORE_ENABLE_AVX512)
    target_compile_definitions(quantcore PUBLIC QUANTCORE_HAS_AVX512=1)
endif()
if(QUANTCORE_ENABLE_AVX2)
    target_compile_definitions(quantcore PUBLIC QUANTCORE_HAS_AVX2=1)
endif()
if(QUANTCORE_ENABLE_SSE42)
    target_compile_definitions(quantcore PUBLIC QUANTCORE_HAS_SSE42=1)
endif()

# Fast math (optional, OFF by default)
if(QUANTCORE_ENABLE_FAST_MATH)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(quantcore PUBLIC -ffast-math)
    elseif(MSVC)
        target_compile_options(quantcore PUBLIC /fp:fast)
    endif()
    target_compile_definitions(quantcore PUBLIC QUANTCORE_FAST_MATH=1)
endif()

# Apache Arrow / Parquet I/O
if(QUANTCORE_ENABLE_PARQUET)
    include(../cmake/FindArrow.cmake)
    if(Arrow_FOUND AND PARQUET_FOUND)
        target_compile_definitions(quantcore PUBLIC QUANTCORE_HAS_PARQUET=1)
        target_include_directories(quantcore PUBLIC ${ARROW_INCLUDE_DIRS})
        target_link_libraries(quantcore PUBLIC Arrow::Arrow Parquet::Parquet)
        message(STATUS "QuantCore: Parquet I/O enabled (Arrow ${ARROW_INCLUDE_DIR})")
    else()
        set(QUANTCORE_ENABLE_PARQUET OFF)
        message(WARNING "QuantCore: Arrow/Parquet not found — Parquet I/O disabled")
    endif()
endif()

# HDF5 I/O
if(QUANTCORE_ENABLE_HDF5)
    include(../cmake/FindHDF5.cmake)
    if(HDF5_FOUND)
        target_compile_definitions(quantcore PUBLIC QUANTCORE_HAS_HDF5=1)
        target_include_directories(quantcore PUBLIC ${HDF5_INCLUDE_DIRS})
        target_link_libraries(quantcore PUBLIC ${HDF5_LIBRARIES})
        message(STATUS "QuantCore: HDF5 I/O enabled (${HDF5_INCLUDE_DIR})")
    else()
        set(QUANTCORE_ENABLE_HDF5 OFF)
        message(WARNING "QuantCore: HDF5 not found — HDF5 I/O disabled")
    endif()
endif()

# Generate compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
