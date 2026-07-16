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

    # registry
    registry/OperatorRegistry.cpp
)

target_include_directories(quantcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

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

# Generate compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
